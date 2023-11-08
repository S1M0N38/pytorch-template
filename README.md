<div align="center">
  <h1>[🔥]</h1>
  <p><em><a href="https://pytorch.org/">PyTorch</a> template with <a href="https://toml.io/">TOML</a></em></p>

<a href="https://github.com/S1M0N38/pytorch-template/actions/workflows/main.yml">
    <img alt="Status" src="https://img.shields.io/github/actions/workflow/status/S1M0N38/pytorch-template/main.yml?label=train&amp;style=for-the-badge">
  </a>
  <a>
    <img alt="Python" src="https://img.shields.io/badge/python-3.10-blue?style=for-the-badge&amp;logo=python">
  </a>
  <a href="https://github.com/S1M0N38/pytorch-template/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/S1M0N38/pytorch-template?style=for-the-badge&amp;color=ff69b4">
  </a>
  <a href="https://discord.com/users/S1M0N38#0317">
    <img alt="Discord" src="https://img.shields.io/static/v1?label=DISCORD&amp;message=DM&amp;color=blueviolet&amp;style=for-the-badge">
  </a>
</div>

______________________________________________________________________

There are many components involved in training a PyTorch model, including model
architectures, loss functions, hyperparameters, optimizers, dataloaders, and
all of their arguments.

A standard training loop requires boilerplate code to connect all of
these components, including training and validation steps, saving and loading
checkpoints, and tracking metrics.

To simplify this process, a \[🔥\] template can be used to specify the aforementioned
components using a TOML file, while implementing the remaining ones as a minimal class
in a single PyTorch file.

## Usage

1. Define the configuration in a TOML file (e.g. `configs/example.toml`)
1. Train, Validate and Test the model with `python main.py configs/example.toml`

## How it works

A TOML file is read by the `Trainer` class (the unique class that implements all
the boilerplate code for training) and dynamically loads classes and their
arguments using the `init` function:

```python
def init(module: object, class_args: dict):
    class_name = class_args.pop("class")
    return getattr(module, class_name)(**class_args)
```

Suppose the following TOML configuration for the optimizer:

```TOML
[optimizer]
class = "Adam"
lr = 1e-3
weight_decay=0
```

From the `[optimizer]` section, \[🔥\] uses the `class` to create a new instance
of a `torch.optim.Adam` optimizer and passes all other values as arguments to the
new object (here `lr` and `weight_decay`). Optimizer also has `parameters` as a
positional argument but this is already provided by the code in the `Learner`
class.

You can also initialize from TOML your custom classes

```TOML
[model]
class = "LeNet5"
num_classes = 10
```

This configuration section will initialize `LeNet5` class defined in `models/models.py`
as the model architecture.

You can easily understand how a TOML file is loaded by `Trainer` and `Tester` by
comparing `configs/example.toml` and `__init__()` methods in `main.py`.

## Q&A

- **Is \[🔥\] stable?**
  *No, I'm tweaking this template based on my experience and needs, so expect
  breaking changes. Nevertheless, this is a template so you might have to heavily modify
  it to fit your needs.*

- **Why the name \[🔥\]?**
  *It's a combination of the PyTorch flame and the square brackets defining
  sections in a TOML file.*

- **Why TOML?**
  *I think it's simpler than [YAML](https://en.wikipedia.org/wiki/YAML) and better
  than [JSON](https://en.wikipedia.org/wiki/JSON) for configuration. Moreover, the
  Python ecosystem starts to embrace it:
  [tomllib](https://docs.python.org/3/library/tomllib.html) in the standard
  library and
  [pyproject.toml](https://snarky.ca/what-the-heck-is-pyproject-toml/) for Python
  project configuration.*

## Similar projects

- [victoresque/pytorch-template](https://github.com/victoresque/pytorch-template):
  PyTorch deep learning projects made easy.
- [moemen95/Pytorch-Project-Template](https://github.com/moemen95/Pytorch-Project-Template):
  A scalable template for PyTorch projects, with examples in Image Segmentation,
  Object classification, GANs and Reinforcement Learning.
