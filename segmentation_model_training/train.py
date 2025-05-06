import onnx
import torch
import typer
from typing import Optional, Tuple, List
from pathlib import Path
from datetime import datetime

from pytorch_lightning import LightningModule, seed_everything, Trainer, Callback, LightningDataModule, loggers
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from configs.default_config import *
from datamodules.seg_datamodule import SegmentationDataModule
from models.segmentation_model import SegmentationModel

def train(
    # Training parameters (matching config order)
    num_epochs: Optional[int] = typer.Option(None, "--num-epochs", "-e", help="Number of epochs"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Batch size"),
    learning_rate: Optional[float] = typer.Option(None, "--learning-rate", "-lr", help="Learning rate"),
    weight_decay: Optional[float] = typer.Option(None, "--weight-decay", help="Weight decay for optimizer"),
    device: Optional[str] = typer.Option(None, "--device", help="Device (cuda or cpu)"),
    num_workers: Optional[int] = typer.Option(None, "--num-workers", "-w", help="Number of workers"),
    loss_fn: Optional[str] = typer.Option(None, "--loss-fn", help="Loss function"),
    
    # Data parameters
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Path to images directory"),
    masks_dir: Optional[Path] = typer.Option(None, "--masks-dir", "-m", help="Path to masks directory"),
    train_val_test_split: Optional[List[float]] = typer.Option(None, "--split", help="Train/val/test split ratio"),
    random_state: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    img_size: Optional[Tuple[int, int]] = typer.Option(None, "--img-size", help="Image size (height, width)"),
    
    # Loss parameters
    focal_loss_weights: Optional[List[float]] = typer.Option(None, "--fl-weights", help="Weights for focal loss"),
    focal_loss_gamma: Optional[float] = typer.Option(None, "--fl-gamma", help="Gamma for focal loss"),
    
    # Model parameters
    encoder_name: Optional[str] = typer.Option(None, "--encoder", help="Encoder backbone name"),
    encoder_weights: Optional[str] = typer.Option(None, "--encoder-weights", help="Pretrained encoder weights"),
    model_name: Optional[str] = typer.Option(None, "--model", help="Model architecture"),
    
    # Scheduler parameters
    scheduler_type: Optional[str] = typer.Option(None, "--scheduler", help="Learning rate scheduler type"),
    
    # Other parameters
    checkpoint_path: Optional[Path] = typer.Option(None, "--checkpoint", "-c", help="Path to load checkpoint"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory for model checkpoints"),
    log_dir: Optional[Path] = typer.Option(None, "--log-dir", help="Directory for logging"),
    model_save_path: Optional[Path] = typer.Option(None, "--save-path", help="Path to save final model"),
    logger: Optional[bool] = typer.Option(None, "--logger", help="Enable logging to WandB"),
    export_to_onnx: Optional[bool] = typer.Option(None, "--onnx", help="Export model to ONNX format"),
    project_name: Optional[str] = typer.Option(None, "--project", help="Project name for wandb"),
    experiment_name: Optional[str] = typer.Option(None, "--name", help="Experiment name for logging"),
    ):

    # Training parameters
    _num_epochs = num_epochs or NUM_EPOCHS
    _batch_size = batch_size or BATCH_SIZE
    _learning_rate = learning_rate or LEARNING_RATE
    _weight_decay = weight_decay or WEIGHT_DECAY
    _device = device or DEVICE
    _num_workers = num_workers or NUM_WORKERS
    _loss_fn = loss_fn or LOSS_FN
    
    # Data parameters
    _data_dir = data_dir or DATA_DIR
    _masks_dir = masks_dir or MASKS_DIR
    _train_val_test_split = train_val_test_split or TRAIN_VAL_TEST_SPLIT
    _random_state = random_state or RANDOM_STATE
    _img_size = img_size or IMG_SIZE
    _original_size = ORIGINAL_IMG_SIZE
    
    # Loss parameters
    _focal_loss_weights = focal_loss_weights or FOCAL_LOSS_WEIGHTS
    _focal_loss_gamma = focal_loss_gamma or FOCAL_LOSS_GAMMA
    
    # Model parameters
    _encoder_name = encoder_name or ENCODER_NAME
    _encoder_weights = encoder_weights or ENCODER_WEIGHTS  
    _model_name = model_name or MODEL_NAME
    
    # Scheduler parameters
    _scheduler_type = scheduler_type or SCHEDULER_TYPE
    
    # Other parameters
    _checkpoint_path = checkpoint_path or CHECKPOINT_DIR
    _output_dir = output_dir or CHECKPOINT_DIR
    _log_dir = log_dir or LOG_DIR
    _model_save_path = model_save_path or MODEL_SAVE_PATH
    _logger = logger if logger is not None else LOGGER
    _export_to_onnx = export_to_onnx if export_to_onnx is not None else EXPORT_TO_ONNX
    _project_name = project_name or PROJECT_NAME
    _experiment_name = experiment_name or None

    if not _experiment_name:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _experiment_name = f"{_model_name}_{_encoder_name}_{_img_size[0]}x{_img_size[1]}_{current_time}"


    # Set random seed for reproducibility
    seed_everything(_random_state, workers=True)

    # Initialize data module
    data_module = SegmentationDataModule(
        image_dir=_data_dir,
        mask_dir=_masks_dir,
        train_transform=AUGMENTATIONS['train'],
        val_transform=AUGMENTATIONS['val'],
        test_transform=AUGMENTATIONS['test'],
        batch_size=_batch_size,
        num_workers=_num_workers,
        img_size=_img_size,
        train_val_test_split=_train_val_test_split,
        classes=CLASSES
    )

    # Initialize loggers
    loggers = []
    if _logger:
        wandb_logger = WandbLogger(
            name=_experiment_name,
            project=_project_name,
            log_model=True,
            save_dir=_log_dir
        )
        wandb_logger.log_hyperparams({
            # Training parameters
            "num_epochs": _num_epochs,
            "batch_size": _batch_size,
            "learning_rate": _learning_rate,
            "weight_decay": _weight_decay,
            "loss_function": _loss_fn,
            
            # Data parameters
            "img_size": _img_size,
            "train_val_test_split": _train_val_test_split,
            
            # Loss parameters
            "focal_loss_weights": _focal_loss_weights,
            "focal_loss_gamma": _focal_loss_gamma,
            
            # Model parameters
            "model_name": _model_name,
            "encoder_name": _encoder_name,
            "encoder_weights": _encoder_weights,
            
            # Scheduler parameters
            "scheduler_type": _scheduler_type,
        })
        loggers.append(wandb_logger)

        # Add TensorBoard logger as backup
        tensorboard_logger = TensorBoardLogger(
            save_dir=_log_dir,
            name=_experiment_name
        )
        loggers.append(tensorboard_logger)

    # Initialize callbacks
    _callbacks = []

    # Initialize checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_epoch",
        dirpath=_output_dir,
        filename=f"{_experiment_name}" + "-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    _callbacks.append(checkpoint_callback)

    # Add early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor="val_loss_epoch",
        min_delta=0.00,
        patience=20,
        verbose=True,
        mode="min",
        strict=False,
    )
    _callbacks.append(early_stopping_callback)

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=_num_epochs,
        devices=1 if _device == "cuda" else 0,
        logger=loggers,
        callbacks=_callbacks,
        accelerator="gpu" if _device == "cuda" else "cpu",
        precision="16-mixed" if _device == "cuda" else 32,
        log_every_n_steps=1,
    )

    # Load model
    model = SegmentationModel(
        model_name=_model_name,
        encoder_name=_encoder_name,
        encoder_weights=_encoder_weights,
        in_channels=3,
        classes=CLASSES,
        loss_function=_loss_fn,
        lr=_learning_rate,
        weight_decay=_weight_decay,
        scheduler_type=_scheduler_type,
        original_size=_original_size
    )

    # Train the model
    trainer.fit(model, data_module)

    # Save the final model
    trainer.save_checkpoint(_model_save_path)
    print(f"Model saved to {_model_save_path}")

    # Test the model
    print("Evaluating model on test dataset...")
    test_results = trainer.test(model, data_module)
    print(f"Test results: {test_results}")

    # Export to ONNX if specified
    if _export_to_onnx:
        dummy_input = torch.randn(1, 3, _img_size[0], _img_size[1])
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        torch.onnx.export(
            model,
            dummy_input,
            "model.onnx",
            export_params=True,
            opset_version=15,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes
        )
        print("Model exported to ONNX format as model.onnx")

if __name__ == "__main__":
    typer.run(train)

