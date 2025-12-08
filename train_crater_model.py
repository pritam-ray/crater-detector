"""
Lunar Crater Detection Model Training Script
Uses YOLOv8 for crater detection on lunar surface images

Requirements:
- ultralytics (YOLOv8)
- Dataset structure:
  dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
"""

from ultralytics import YOLO
import os


def train_crater_detector():
    """
    Train YOLOv8 model for lunar crater detection
    """
    
    # Model configuration
    MODEL_SIZE = 'yolov8n.pt'  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    EPOCHS = 100
    BATCH_SIZE = 16
    IMAGE_SIZE = 640
    
    # Training parameters
    PATIENCE = 50  # Early stopping patience
    SAVE_PERIOD = 10  # Save checkpoint every N epochs
    
    # Paths
    DATA_CONFIG = 'config.yaml'  # Path to your dataset configuration file
    PROJECT_NAME = 'runs/detect'
    EXPERIMENT_NAME = 'train_100'
    
    print("=" * 60)
    print("Lunar Crater Detection - YOLOv8 Training")
    print("=" * 60)
    print(f"Model: {MODEL_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print("=" * 60)
    
    # Load pretrained YOLOv8 model
    model = YOLO(MODEL_SIZE)
    
    # Train the model
    results = model.train(
        data=DATA_CONFIG,           # Path to dataset YAML config
        epochs=EPOCHS,              # Number of training epochs
        batch=BATCH_SIZE,           # Batch size
        imgsz=IMAGE_SIZE,           # Image size (pixels)
        patience=PATIENCE,          # Early stopping patience
        save=True,                  # Save checkpoints
        save_period=SAVE_PERIOD,    # Save checkpoint every N epochs
        project=PROJECT_NAME,       # Project directory
        name=EXPERIMENT_NAME,       # Experiment name
        pretrained=True,            # Use pretrained weights
        optimizer='auto',           # Optimizer (auto, SGD, Adam, AdamW)
        verbose=True,               # Verbose output
        seed=0,                     # Random seed for reproducibility
        deterministic=True,         # Deterministic training
        single_cls=False,           # Train as single-class (if True)
        rect=False,                 # Rectangular training
        cos_lr=False,               # Use cosine learning rate scheduler
        close_mosaic=10,            # Disable mosaic augmentation for final epochs
        resume=False,               # Resume training from last checkpoint
        amp=True,                   # Automatic Mixed Precision training
        fraction=1.0,               # Train on fraction of data
        profile=False,              # Profile ONNX and TensorRT speeds
        freeze=None,                # Freeze layers (e.g., freeze=10)
        lr0=0.01,                   # Initial learning rate
        lrf=0.01,                   # Final learning rate factor
        momentum=0.937,             # SGD momentum
        weight_decay=0.0005,        # Optimizer weight decay
        warmup_epochs=3.0,          # Warmup epochs
        warmup_momentum=0.8,        # Warmup momentum
        warmup_bias_lr=0.1,         # Warmup bias learning rate
        box=7.5,                    # Box loss gain
        cls=0.5,                    # Class loss gain
        dfl=1.5,                    # DFL loss gain
        pose=12.0,                  # Pose loss gain (for pose models)
        kobj=1.0,                   # Keypoint obj loss gain
        label_smoothing=0.0,        # Label smoothing epsilon
        nbs=64,                     # Nominal batch size
        hsv_h=0.015,                # HSV-Hue augmentation
        hsv_s=0.7,                  # HSV-Saturation augmentation
        hsv_v=0.4,                  # HSV-Value augmentation
        degrees=0.0,                # Rotation augmentation (degrees)
        translate=0.1,              # Translation augmentation
        scale=0.5,                  # Scale augmentation
        shear=0.0,                  # Shear augmentation
        perspective=0.0,            # Perspective augmentation
        flipud=0.0,                 # Flip up-down augmentation probability
        fliplr=0.5,                 # Flip left-right augmentation probability
        mosaic=1.0,                 # Mosaic augmentation probability
        mixup=0.0,                  # Mixup augmentation probability
        copy_paste=0.0,             # Copy-paste augmentation probability
        plots=True,                 # Save training plots
        val=True,                   # Validate during training
    )
    
    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    print(f"Best weights saved at: {PROJECT_NAME}/{EXPERIMENT_NAME}/weights/best.pt")
    print(f"Last weights saved at: {PROJECT_NAME}/{EXPERIMENT_NAME}/weights/last.pt")
    print(f"Results saved in: {PROJECT_NAME}/{EXPERIMENT_NAME}/")
    print("=" * 60)
    
    return results


def validate_model(model_path, data_config):
    """
    Validate trained model on test dataset
    
    Args:
        model_path: Path to trained model weights
        data_config: Path to dataset configuration YAML
    """
    print("\n" + "=" * 60)
    print("Model Validation")
    print("=" * 60)
    
    # Load trained model
    model = YOLO(model_path)
    
    # Validate
    metrics = model.val(
        data=data_config,
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.6,
        max_det=300,
        plots=True
    )
    
    print(f"\nValidation Results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print("=" * 60)
    
    return metrics


def export_model(model_path, export_format='onnx'):
    """
    Export trained model to different formats
    
    Args:
        model_path: Path to trained model weights
        export_format: Export format (onnx, torchscript, coreml, etc.)
    """
    print("\n" + "=" * 60)
    print(f"Exporting Model to {export_format.upper()}")
    print("=" * 60)
    
    model = YOLO(model_path)
    model.export(format=export_format)
    
    print(f"Model exported successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Train the model
    results = train_crater_detector()
    
    # Optional: Validate the trained model
    # Uncomment the lines below to validate after training
    # model_path = 'runs/detect/train_100/weights/best.pt'
    # validate_model(model_path, 'config.yaml')
    
    # Optional: Export the model to ONNX format
    # Uncomment the line below to export
    # export_model('runs/detect/train_100/weights/best.pt', 'onnx')
    
    print("\n✅ All tasks completed successfully!")
