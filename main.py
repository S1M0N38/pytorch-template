import sys
import logging
from hashlib import sha256
from pathlib import Path
from datetime import datetime

import toml
import torch
import torchmetrics
from torch.utils.tensorboard import SummaryWriter

import dataloaders
import metrics
import models
import losses


def init(module: object, name: dict):
    class_name = name["class"]
    del name["class"]
    class_args = name
    return getattr(module, class_name)(**class_args)


def logger(path: Path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(levelname)8s - %(asctime)s - %(message)s")
    fh = logging.FileHandler(path)
    sh = logging.StreamHandler()
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class Learner:
    def __init__(self, config: Path):
        now = datetime.now()
        self.config = toml.load(config)
        hash = sha256(toml.dumps(self.config).encode()).hexdigest()[:8]
        self.name = f"{now.strftime('%m%d_%H%M')}_{hash}"

        # [dataloaders]
        self.train_dataloader = init(dataloaders, self.config["dataloaders"]["train"])
        self.val_dataloader = init(dataloaders, self.config["dataloaders"]["val"])
        self.test_dataloader = init(dataloaders, self.config["dataloaders"]["test"])
        self.num_classes = len(self.train_dataloader.dataset.classes)
        self.batches = batches = len(self.train_dataloader)

        # [learner]
        self.epochs, self.epoch = self.config["learner"]["epochs"], 0
        self.steps, self.step = self.epochs * batches, 0
        self.val_period = self.config["learner"].get("val_period", self.epochs)
        self.save_period = self.config["learner"].get("save_period", self.epochs)
        self.log_period = self.config["learner"].get("log_period", self.steps)
        self.save_path = Path(self.config["learner"]["save_path"]) / self.name
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.load_path = self.config["learner"].get("load_path")
        self.writer = SummaryWriter(self.save_path / "runs")

        # [device]
        self.gpus = list(range(self.config["device"]["num_gpus"]))
        self.device = torch.device("cuda:0" if self.gpus else "cpu")

        # [model]
        self.config["model"]["num_classes"] = self.num_classes
        self.model = init(models, self.config["model"]).to(self.device)
        if len(self.gpus) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus)

        # [loss]
        self.loss = init(losses, self.config["loss"])
        self.train_loss = torchmetrics.MeanMetric().to(self.device)
        self.val_loss = torchmetrics.MeanMetric().to(self.device)
        self.test_loss = torchmetrics.MeanMetric().to(self.device)

        # [optimizer]
        self.config["optimizer"]["params"] = self.model.parameters()
        self.optimizer = init(torch.optim, self.config["optimizer"])

        # [lr_scheduler]
        self.config["lr_scheduler"]["optimizer"] = self.optimizer
        self.lr_scheduler = init(torch.optim.lr_scheduler, self.config["lr_scheduler"])

        # [metrics]
        self.train_metrics = {}
        for name, metric in self.config["metrics"]["train"].items():
            metric["num_classes"] = self.num_classes
            self.train_metrics[name] = init(metrics, metric).to(self.device)
        self.train_metrics = torchmetrics.MetricCollection(self.train_metrics)
        self.val_metrics = {}
        for name, metric in self.config["metrics"]["val"].items():
            metric["num_classes"] = self.num_classes
            self.val_metrics[name] = init(metrics, metric).to(self.device)
        self.val_metrics = torchmetrics.MetricCollection(self.val_metrics)
        self.test_metrics = {}
        for name, metric in self.config["metrics"]["test"].items():
            metric["num_classes"] = self.num_classes
            self.test_metrics[name] = init(metrics, metric).to(self.device)
        self.test_metrics = torchmetrics.MetricCollection(self.test_metrics)

        # Hyperparameters for logging
        self.hparams = {
            "model": self.model.__class__.__name__,
            "loss": self.loss.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__,
            "lr_scheduler": self.lr_scheduler.__class__.__name__,
            "init_lr": self.optimizer.param_groups[0]["lr"],
            "batch_size": self.train_dataloader.batch_size,
            "epochs": self.epochs,
            "steps": self.steps,
        }

        # Init logger
        self.logger = logger(self.save_path / "learner.log")

        # Save config
        with config.open("rb") as src_file:
            with (self.save_path / "config.toml").open("wb") as dst_file:
                dst_file.write(src_file.read())

        # Load model and update steps
        if self.load_path:
            self.load_path = Path(self.load_path)
            self.step = int(self.load_path.stem)
            self.steps += int(self.load_path.stem)
            self._load(self.load_path)

    def _train(self) -> bool:
        self.model = self.model.train()
        with torch.enable_grad():
            for inputs, targets in self.train_dataloader:
                if self.step == self.steps:
                    return True
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                loss.backward()
                self.train_metrics(outputs, targets)
                self.train_loss(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.step += 1

                if self.step % self.log_period == 0:
                    metrics = self.train_metrics.compute()
                    metrics["loss"] = self.train_loss.compute()
                    self._log("debug", "train", metrics, add_scalar=True)
                    self.train_metrics.reset()
                    self.train_loss.reset()

            self.epoch += 1
            self._log("info", "train")
            self.lr_scheduler.step()

    def _val(self):
        self.model = self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.val_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                self.val_metrics(outputs, targets)
                self.val_loss(loss)

            metrics = self.val_metrics.compute()
            metrics["loss"] = self.val_loss.compute()
            self._log("info", "val", metrics, add_scalar=True)
            self.val_metrics.reset()
            self.val_loss.reset()

    def _test(self):
        self.model = self.model.eval()
        with torch.no_grad():
            for inputs, targets in self.test_dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, targets)
                self.test_metrics(outputs, targets)
                self.test_loss(loss)

            metrics = self.test_metrics.compute()
            metrics["loss"] = self.test_loss.compute()
            self._log("info", "test", metrics, add_hparams=True)
            self.test_metrics.reset()
            self.test_loss.reset()

    def _save(self, path: Path):
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)
        self.logger.info(f"Save  {path.relative_to('.')}")

    def _load(self, path: Path):
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        self.logger.info(f"Load  {path.relative_to('.')}")

    def _log(
        self,
        level: str = "info",
        mode: str = "train",
        metrics: dict = {},
        add_scalar: bool = False,
        add_hparams: bool = False,
    ):
        getattr(self.logger, level)(
            (
                f"{mode.capitalize():<5} [{(self.step / self.steps) * 100 :3.0f}%] | "
                f"Epoch [{self.epoch}/{self.epochs}] | "
                f"Step [{self.step}/{self.steps}] | "
                f"ETA {self._eta()} | "
                f"{', '.join([f'{n} {m.item():.3f}' for n, m in metrics.items()])}"
            )
        )

        if add_scalar:
            for name, value in metrics.items():
                self.writer.add_scalar(f"{mode}/{name}", value, self.step)

        if add_hparams:
            metrics = {f"{mode}/{k}": v.item() for k, v in metrics.items()}
            self.writer.add_hparams(self.hparams, metrics, run_name="../runs")

    def _eta(self):
        enlapsed = datetime.now() - self.start_time
        eta = (enlapsed / self.step * (self.steps - self.step)).seconds
        return f"{eta // 3600:>2}:{(eta % 3600) // 60:02}:{eta % 60:02}"

    def train(self):
        self.start_time = datetime.now()
        while True:
            finished = self._train()
            if self.epoch % self.val_period == 0 or self.step == self.steps:
                self._val()
            if self.epoch % self.save_period == 0 or self.step == self.steps:
                self._save(self.save_path / "checkpoints" / f"{self.step}.pt")
            if finished or self.step == self.steps:
                break
        self.load_path = self.save_path / "checkpoints" / f"{self.step}.pt"

    def test(self):
        self._test()


if __name__ == "__main__":
    config = Path(sys.argv[1])
    learner = Learner(config)
    learner.train()
    learner.test()
