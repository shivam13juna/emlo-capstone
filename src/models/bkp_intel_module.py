from typing import Any, List, Dict
import os
os.environ['HYDRA_FULL_ERROR'] = '1'
import timm
import torch
import torchvision
from PIL import Image
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import torchmetrics
import io
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import itertools
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import logging

# configure logging at the root level of Lightning
# logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# # configure logging on module level, redirect to file
# logger = logging.getLogger("pytorch_lightning.core")

# from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1

class IntHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text

class INTELLitModule(LightningModule):
    """Example of LightningModule for INTEL classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model_name: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        n_labels = 6
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.n_labels = 6
        self.net = timm.create_model(model_name=model_name, pretrained=True, num_classes=self.n_labels)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=self.n_labels)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.n_labels)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.n_labels)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        # self.writer = SummaryWriter()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()
        # print("What are hparams: ", self.hparams)
        # self.logger.log_hyperparams(self.hparams, {"hp/precision": 0, "hp/recall": 0, "hp/f1_score": 0})
        # self.logger.log_hyperparams(self.hparams, {"hp/metric": 0})

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on intel
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outs: List[Any]):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

        # log confusion matrix
        # tb = self.writer  # noqa

        # outputs = torch.cat([tmp['preds'] for tmp in outs])
        # labels = torch.cat([tmp['targets'] for tmp in outs])
        # confusion = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.n_labels).to(outputs.get_device())
        # # Save confusion matrix to tensorboard with following labels ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        # computed_confusion = confusion(outputs, labels)
        # # tb.add_figure("val_confusion_matrix", plot_confusion_matrix(computed_confusion, ["buildings", "forest", "glacier", "mountain", "sea", "street"]), global_step=self.current_epoch)        

        

        # # also Log precision, recall and F1score

        # precision = torchmetrics.Precision(task="multiclass", num_classes=self.n_labels).to(outputs.get_device())
        # value = precision(outputs, labels)
        # # tb.add_scalar("val_precision", value, global_step=self.current_epoch)
        # # self.log("hp/precision", value)
        # # self.log("hp/metric", value)
        # # self.log("hp_metric", value)

        # recall = torchmetrics.Recall(task="multiclass", num_classes=self.n_labels).to(outputs.get_device())
        # value = recall(outputs, labels)
        # # tb.add_scalar("val_recall", value, global_step=self.current_epoch)
        # # self.log("hp/recall", value)

        # f1_score = torchmetrics.classification.MulticlassF1Score(task="multiclass", num_classes=self.n_labels).to(outputs.get_device())
        # value = f1_score(outputs, labels)
        # # tb.add_scalar("val_f1_score", value, global_step=self.current_epoch)
        # # self.log("hp/f1_score", value)



    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(16, 16))
    cm = cm.cpu().numpy()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "intel.yaml")
    _ = hydra.utils.instantiate(cfg)
