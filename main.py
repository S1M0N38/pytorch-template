import copy
import logging
import sys
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import toml
import torch
import torchmetrics
from torch.utils.tensorboard.writer import SummaryWriter

import dataloaders
import losses
import metrics
import models


class Trainer:
    def __init__(self, config: dict, experiemnt: str):
        cfg = copy.deepcopy(config)
        self.config = config

        # [dataloaders]
        self.dataloader_train = init(dataloaders, cfg["dataloaders"]["train"])
        self.dataloader_val = init(dataloaders, cfg["dataloaders"]["val"])

        self.epochs, self.epoch = cfg.get("epochs", 0), 0
        self.steps, self.step = self.epochs * len(self.dataloader_train), 0
        self.period_val = cfg.get("validate", self.epochs)
        self.period_log = cfg.get("log", self.steps)

        self.gpus = list(range(cfg.get("num_gpus", 0)))
        self.device = torch.device("cuda:0" if self.gpus else "cpu")

        # [patience]
        self.patience = cfg.get("patience", {})

        # [model]
        self.model = init(models, cfg["model"]).to(self.device)
        if len(self.gpus) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus)

        # [loss]
        self.loss = init(losses, cfg["loss"])
        self.loss_train = torchmetrics.MeanMetric().to(self.device)
        self.loss_val = torchmetrics.MetricTracker(
            torchmetrics.MeanMetric().to(self.device),
            maximize=False,
        )

        # [optimizer]
        cfg["optimizer"]["params"] = self.model.parameters()
        self.optimizer = init(torch.optim, cfg["optimizer"])

        # [lr_scheduler]
        cfg["lr_scheduler"]["optimizer"] = self.optimizer
        self.lr_scheduler = init(torch.optim.lr_scheduler, cfg["lr_scheduler"])

        # [metrics]
        self.metrics_train = torchmetrics.MetricCollection(
            {
                name: init(metrics, metric).to(self.device)
                for name, metric in cfg["metrics"]["train"].items()
            }
        )
        self.metrics_val = torchmetrics.MetricTracker(
            torchmetrics.MetricCollection(
                {
                    name: init(metrics, metric).to(self.device)
                    for name, metric in cfg["metrics"]["val"].items()
                }
            )
        )

        # Track and save results
        self.path = Path(cfg.get("path", ".")) / experiemnt
        self.writer = SummaryWriter(self.path / "runs")
        self.logger = init_logger(self.path / "trainer.log")
        (self.path / "checkpoints").mkdir()
        with open(self.path / "config.toml", "w") as f:
            toml.dump(self.config, f)

        self.logger.info(f"Trainer initialized using {self.path / 'trainer.toml'}")
        self.logger.debug(f"Using {self.device} as device.")

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
            self.logger.debug(f"Update LR: {self.lr_scheduler.get_last_lr()[0]:.3e}")

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
                _, step = self.loss_val.best_metric(return_step=True)  # type: ignore
                loss_best_k_last = self.loss_val.n_steps - step  # type: ignore

                # Find the index (from the end) of the bests steps for metrics_val
                _, step = self.metrics_val.best_metric(return_step=True)  # type: ignore
                metrics_best_k_last = {
                    metric: self.metrics_val.n_steps - metric_best_step  # type: ignore
                    for metric, metric_best_step in step.items()  # type: ignore
                }

                # Save checkpoint with best loss_val
                if loss_best_k_last == 1:
                    self._save("loss_val.pt")
                    self.logger.debug("Saved checkpoint with best 'loss_val'.")

                # Save checkpoints with best metrics_val
                for metric, metric_best_k_last in metrics_best_k_last.items():
                    if metric_best_k_last == 1:
                        self._save(f"{metric}.pt")
                        self.logger.debug(f"Saved checkpoint with best '{metric}'.")

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
                self.logger.info("Training completed.")
                break

        # Load last checkpoint
        self._save("last.pt")
        self.logger.debug("Saved checkpoint with last step.")


class Tester:
    def __init__(self, config: dict, experiemnt: str) -> None:
        cfg = copy.deepcopy(config)
        self.config = config

        # [dataloaders]
        self.dataloader_test = init(dataloaders, cfg["dataloaders"]["test"])

        self.gpus = list(range(cfg.get("num_gpus", 0)))
        self.device = torch.device("cuda:0" if self.gpus else "cpu")

        # [model]
        self.model = init(models, cfg["model"]).to(self.device)
        if len(self.gpus) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus)

        # [loss]
        self.loss = init(losses, cfg["loss"]).to(self.device)
        self.loss_test = torchmetrics.MeanMetric().to(self.device)

        # [metrics]
        self.metrics_test = torchmetrics.MetricCollection(
            {
                name: init(metrics, metric).to(self.device)
                for name, metric in cfg["metrics"]["test"].items()
            }
        )

        # Track and save results
        self.path = Path(cfg.get("path", ".")) / experiemnt
        self.logger = init_logger(self.path / "tester.log")
        self.logger.info(f"Tester initialized using {self.path / 'config.toml'}")
        self.logger.debug(f"Using {self.device} as device.")

    def __str__(self) -> str:
        return toml.dumps(self.config)

    def load(self, path: Path):
        checkpoint = torch.load(path.resolve())
        self.model.load_state_dict(checkpoint["model"])
        return self

    def test(self) -> dict[str, torch.Tensor]:
        self.model = self.model.eval()
        self.logger.info("Start testing.")
        with torch.no_grad():
            for inputs, targets in self.dataloader_test:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                self.metrics_test(outputs, targets)
                self.loss_test(loss)

            metrics = self.metrics_test.compute()
            metrics["loss"] = self.loss_test.compute()

        self.logger.info("Testing completed.")
        return metrics


def init(module: object, class_args: dict):
    class_name = class_args.pop("class")
    return getattr(module, class_name)(**class_args)


def init_logger(path: Path):
    logger = logging.getLogger(path.stem)
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
    experiement = generate_experiment_name(config)

    trainer = Trainer(config, experiement)
    print(trainer)
    print(f"Progress at {trainer.path.parent / '*' / 'trainer.log'}")
    print("Training ...")
    trainer.train()

    tester = Tester(config, experiement)
    tester.load(tester.path / "checkpoints" / "last.pt")
    print(f"Progress at {trainer.path.parent / '*' / 'tester.log'}")
    print("Testing ...")
    results = tester.test()
    print(results)
