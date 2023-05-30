import sys
import copy
import logging
from hashlib import sha256
from pathlib import Path
from datetime import datetime

import toml
import torch
import torchmetrics
from torch.utils.tensorboard.writer import SummaryWriter

import dataloaders
import metrics
import models
import losses


class Trainer:
    def __init__(self, experiemnt: str, config: dict):
        self.config = copy.deepcopy(config)

        # [dataloaders]
        self.dataloader_train = init(dataloaders, config["dataloaders"]["train"])
        self.dataloader_val = init(dataloaders, config["dataloaders"]["val"])

        self.epochs, self.epoch = config.get("epochs", 0), 0
        self.steps, self.step = self.epochs * len(self.dataloader_train), 0
        self.period_val = config.get("validate", self.epochs)
        self.period_log = config.get("log", self.steps)

        self.gpus = list(range(config.get("num_gpus", 0)))
        self.device = torch.device("cuda:0" if self.gpus else "cpu")

        # [patience]
        self.patience = config.get("patience", {})

        # [model]
        self.model = init(models, config["model"]).to(self.device)
        if len(self.gpus) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus)

        # [loss]
        self.loss = init(losses, config["loss"])
        self.loss_train = torchmetrics.MeanMetric().to(self.device)
        self.loss_val = torchmetrics.MetricTracker(
            torchmetrics.MeanMetric().to(self.device),
            maximize=False,
        )

        # [optimizer]
        config["optimizer"]["params"] = self.model.parameters()
        self.optimizer = init(torch.optim, config["optimizer"])

        # [lr_scheduler]
        config["lr_scheduler"]["optimizer"] = self.optimizer
        self.lr_scheduler = init(torch.optim.lr_scheduler, config["lr_scheduler"])

        # [metrics]
        self.metrics_train = torchmetrics.MetricCollection(
            {
                name: init(metrics, metric).to(self.device)
                for name, metric in config["metrics"]["train"].items()
            }
        )
        self.metrics_val = torchmetrics.MetricTracker(
            torchmetrics.MetricCollection(
                {
                    name: init(metrics, metric).to(self.device)
                    for name, metric in config["metrics"]["val"].items()
                }
            )
        )

        # Track and save results
        self.path = Path(config.get("path", ".")) / experiemnt
        self.writer = SummaryWriter(self.path / "runs")
        self.logger = init_logger(self.path / "trainer.log")
        (self.path / "checkpoints").mkdir()
        with open(self.path / "trainer.toml", "w") as f:
            toml.dump(self.config, f)

    def __str__(self):
        return toml.dumps(self.config)

    def _train(self):
        self.model = self.model.train()
        with torch.enable_grad():
            for inputs, targets in self.dataloader_train:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                loss.backward()
                self.metrics_train(outputs, targets)
                self.loss_train(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.step += 1

                if self.step % self.period_log == 0:
                    metrics = self.metrics_train.compute()
                    metrics["loss"] = self.loss_train.compute()
                    self._log("debug", "train", metrics)
                    self.metrics_train.reset()
                    self.loss_train.reset()

                if self.step == self.steps:
                    break

            self.epoch += 1
            self._log("info", "train")
            self.lr_scheduler.step()

    def _val(self):
        self.model = self.model.eval()
        self.metrics_val.increment()
        self.loss_val.increment()
        with torch.no_grad():
            for inputs, targets in self.dataloader_val:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                self.metrics_val(outputs, targets)
                self.loss_val(loss)

            metrics = self.metrics_val.compute()
            metrics["loss"] = self.loss_val.compute()
            self._log("info", "val", metrics)

    def _eta(self) -> str:
        enlapsed = datetime.now() - self.start_time
        eta = (enlapsed / self.step * (self.steps - self.step)).seconds
        return f"{eta // 3600:>2}:{(eta % 3600) // 60:02}:{eta % 60:02}"

    def _log(self, level: str = "info", mode: str = "train", metrics: dict = {}):
        message = (
            f"{mode.capitalize():<5} [{(self.step / self.steps) * 100 :3.0f}%] | "
            f"Epoch [{self.epoch}/{self.epochs}] | "
            f"Step [{self.step}/{self.steps}] | "
            f"ETA {self._eta()} | "
            f"{', '.join([f'{n} {m.item():.3f}' for n, m in metrics.items()])}"
        )
        getattr(self.logger, level)(message)

        for name, value in metrics.items():
            self.writer.add_scalar(f"{mode}/{name}", value, self.step)

    def _save(self, name: str):
        syml = self.path / "checkpoints" / name
        syml.unlink(missing_ok=True)  # remove old symlink
        path = self.path / "checkpoints" / f"{self.step}.pt"
        if not path.exists():
            checkpoint = {
                "config": self.config,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
            }
            torch.save(checkpoint, path)
        syml.symlink_to(f"{self.step}.pt")

    def _clean(self):
        path = self.path / "checkpoints"
        active = {p.resolve() for p in path.glob("*.pt") if p.is_symlink()}
        [p.unlink() for p in path.glob("*.pt") if p.resolve() not in active]

    def _load(self):
        raise NotImplementedError("This method is not implemented yet.")

    def train(self):
        self.start_time = datetime.now()

        while True:
            self._train()

            if self.epoch % self.period_val == 0:
                self._val()

                # Find the index (from the end) of the best step for loss_val
                _, loss_best_step = self.loss_val.best_metric(return_step=True)
                loss_best_k_last = self.loss_val.n_steps - loss_best_step

                # Find the index (from the end) of the bests steps for metrics_val
                _, metrics_best_steps = self.metrics_val.best_metric(return_step=True)
                metrics_best_k_last = {
                    metric: self.metrics_val.n_steps - metric_best_step
                    for metric, metric_best_step in metrics_best_steps.items()
                }

                # Save checkpoint with best loss_val
                if loss_best_k_last == 1:
                    self._save("loss_val.pt")

                # Save checkpoints with best metrics_val
                for metric, metric_best_k_last in metrics_best_k_last.items():
                    if metric_best_k_last == 1:
                        self._save(f"{metric}.pt")

                # Remove old checkpoints that have no symlink that points to them
                self._clean()

                # Trigger early stopping for loss_val
                if loss_best_k_last - 1 > self.patience.get("loss", self.epochs):
                    self.logger.info("Early stopping trigger by 'loss'.")
                    break

                # Trigger early stopping for metrics_val
                early_stopping = False
                for metric, metric_best_k_last in metrics_best_k_last.items():
                    if metric_best_k_last - 1 > self.patience.get(metric, self.epochs):
                        self.logger.info(f"Early stopping trigger by '{metric}'.")
                        early_stopping = True
                        break
                if early_stopping:
                    break

            if self.step == self.steps:
                break

        # Load last checkpoint
        self._save("last.pt")


def init(module: object, name: dict):
    class_name = name["class"]
    del name["class"]
    class_args = name
    return getattr(module, class_name)(**class_args)


def init_logger(path: Path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(levelname)8s - %(asctime)s - %(message)s")
    fh = logging.FileHandler(path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def generate_experiment_name(config: dict):
    now = datetime.now()
    hash = sha256(toml.dumps(config).encode()).hexdigest()[:8]
    return f"{now.strftime('%m%d_%H%M')}_{hash}"


if __name__ == "__main__":
    config = toml.load(Path(sys.argv[1]))
    experiemnt = generate_experiment_name(config)
    trainer = Trainer(experiemnt, config)

    print(trainer)
    print(f"Progress at {trainer.path / 'trainer.log'}")
    print("Training ...")

    trainer.train()
