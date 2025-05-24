import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torchmetrics

from losses.dice_loss import DiceLoss
from losses.focal_loss import FocalLoss
from losses.dice_focal_loss import DiceFocalLoss
from losses.loss_wrapper import LossWrapper

from metrics.dice_metrics import DiceMetric
from metrics.iou_metrics import IoUMetric

from configs.default_config import FOCAL_LOSS_WEIGHTS, FOCAL_LOSS_GAMMA

class SegmentationModel(pl.LightningModule):
    def __init__(self,
                model_name,
                encoder_name,
                encoder_weights,
                in_channels,
                classes,
                loss_function,
                lr,
                weight_decay,
                scheduler_type,
                original_size=(600, 960)
                ):
        super().__init__()

        self._model_name = model_name
        self._encoder_name = encoder_name
        self._encoder_weights = encoder_weights
        self._in_channels = in_channels
        self._classes = classes
        self._loss_function = loss_function
        self._lr = lr
        self._weight_decay = weight_decay
        self._scheduler_type = scheduler_type
        
        # Initialize variables to store outputs for on_*_epoch_end hooks
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        if self._model_name == "Unet":
            self._model = smp.Unet(
                encoder_name=self._encoder_name,
                encoder_weights=self._encoder_weights,
                in_channels=self._in_channels,
                classes=len(self._classes),
                activation=None
            )
        elif self._model_name == "DeepLabV3Plus":
            self._model = smp.DeepLabV3Plus(
                encoder_name=self._encoder_name,
                encoder_weights=self._encoder_weights,
                in_channels=self._in_channels,
                classes=len(self._classes),
                activation=None
            )
        elif self._model_name == "FPN":
            self._model = smp.FPN(
                encoder_name=self._encoder_name,
                encoder_weights=self._encoder_weights,
                in_channels=self._in_channels,
                classes=len(self._classes),
                activation=None
            )
        elif self._model_name == "DPT":
            self._model = smp.DPT(
                encoder_name=self._encoder_name,
                encoder_weights=self._encoder_weights,
                in_channels=self._in_channels,
                classes=len(self._classes),
                activation=None
            )
        elif self._model_name == "Segformer":
            self._model = smp.Segformer(
                encoder_name=self._encoder_name,
                encoder_weights=self._encoder_weights,
                in_channels=self._in_channels,
                classes=len(self._classes),
                activation=None
            )
        else:
            raise ValueError(f"Model {self._model_name} is not supported.")
        
        if self._loss_function == "DiceLoss":
            base_loss = DiceLoss()
            self._loss = LossWrapper(base_loss, original_size=original_size)
        elif self._loss_function == "FocalLoss":
            base_loss = FocalLoss(weights=FOCAL_LOSS_WEIGHTS, gamma=FOCAL_LOSS_GAMMA)
            self._loss = LossWrapper(base_loss, original_size=original_size)
        elif self._loss_function == "DiceFocalLoss":
            base_loss = DiceFocalLoss(weights=FOCAL_LOSS_WEIGHTS, gamma=FOCAL_LOSS_GAMMA)
            self._loss = LossWrapper(base_loss, original_size=original_size)
        else:
            raise ValueError(f"Loss function {self._loss_function} is not supported.")
        
        metrics = torchmetrics.MetricCollection([
            DiceMetric(include_background=True, num_classes=len(classes), original_size=original_size),
            IoUMetric(include_background=True, num_classes=len(classes), original_size=original_size)
        ])
        self._train_metrics = metrics.clone(prefix="train_")
        self._val_metrics = metrics.clone(prefix="val_")
        self._test_metrics = metrics.clone(prefix="test_")

        self.save_hyperparameters(
            "model_name",
            "encoder_name",
            "encoder_weights",
            "in_channels",
            "classes",
            "loss_function",
            "lr"
        )

    def forward(self, x):
        return self._model.forward(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y)
        
        if torch.isinf(loss):
            return None
        
        # Log metrics per step with on_step=True for debugging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Calculate and log metrics
        metrics = self._train_metrics(y_pred, y)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        
        # Store output for epoch_end processing
        output = {"loss": loss, "train_loss": loss.detach()}
        self.train_step_outputs.append(output)
        
        return output
    
    def on_train_epoch_end(self):
        if not self.train_step_outputs:
            return
            
        # Aggregate the metrics from each batch
        avg_loss = torch.stack([x["train_loss"] for x in self.train_step_outputs if "train_loss" in x]).mean()
        self.log("train_loss_epoch", avg_loss, prog_bar=True)
        
        # Clear the outputs list
        self.train_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y)
        
        if torch.isinf(loss):
            return None
        
        # Log validation loss explicitly
        # Note: sync_dist=True is important for distributed training
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Calculate and store metrics for epoch_end
        val_metrics = self._val_metrics(y_pred, y)
        self.log_dict(val_metrics, on_step=False, on_epoch=True, sync_dist=True)
        
        # Store output for epoch_end processing
        output = {"val_loss": loss.detach()}
        self.validation_step_outputs.append(output)
        
        return output
    
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
            
        # Aggregate the metrics from each batch
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs if "val_loss" in x]).mean()
        self.log("val_loss_epoch", avg_loss, prog_bar=True)
        
        # Clear the outputs list
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self._loss(y_pred, y)
        
        if torch.isinf(loss):
            return None
        
        # Log test loss
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        
        # Calculate and log metrics
        metrics = self._test_metrics(y_pred, y)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        
        # Store output for epoch_end processing
        output = {"test_loss": loss.detach()}
        self.test_step_outputs.append(output)
        
        return output
    
    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
            
        # Aggregate the metrics from each batch
        avg_loss = torch.stack([x["test_loss"] for x in self.test_step_outputs if "test_loss" in x]).mean()
        self.log("test_loss_epoch", avg_loss)
        
        # Clear the outputs list
        self.test_step_outputs.clear()

    def on_train_epoch_start(self):
        self._train_metrics.reset()
        
    def on_validation_epoch_start(self):
        self._val_metrics.reset()
        
    def on_test_epoch_start(self):
        self._test_metrics.reset()

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)

        if self._scheduler_type == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        elif self._scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        elif self._scheduler_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5, verbose=True)
            scheduler_config = {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        elif self._scheduler_type == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        elif self._scheduler_type == "CyclicLR":
            max_lr = self._lr * 10
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, 
                base_lr=self._lr, 
                max_lr=max_lr,
                step_size_up=100,
                mode='triangular'
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        else:
            raise ValueError(f"Scheduler type {self._scheduler_type} is not supported.")
        return [self.optimizer], [scheduler_config]