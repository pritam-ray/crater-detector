# Lunar Crater Detection - Training Guide

## Dataset Structure Required

Organize your dataset in the following structure:

```
crater_web_app/
├── dataset/
│   ├── images/
│   │   ├── train/          # Training images
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   ├── val/            # Validation images
│   │   │   ├── image1.jpg
│   │   │   └── ...
│   │   └── test/           # Test images (optional)
│   │       └── ...
│   └── labels/
│       ├── train/          # Training labels (YOLO format)
│       │   ├── image1.txt
│       │   ├── image2.txt
│       │   └── ...
│       ├── val/            # Validation labels
│       │   ├── image1.txt
│       │   └── ...
│       └── test/           # Test labels (optional)
│           └── ...
├── config.yaml             # Dataset configuration
└── train_crater_model.py   # Training script
```

## Label Format (YOLO)

Each `.txt` file should contain one line per crater:
```
class_id center_x center_y width height
```

Example for a crater:
```
0 0.5 0.5 0.2 0.2
```

Where:
- `class_id`: 0 (for crater)
- `center_x`, `center_y`: Normalized center coordinates (0-1)
- `width`, `height`: Normalized width and height (0-1)

## How to Run Training

### 1. Activate Virtual Environment
```powershell
cd d:\python\isro\crater_web_app
.\venv_django\Scripts\Activate.ps1
```

### 2. Update Config File
Edit `config.yaml` and update the dataset path if needed.

### 3. Run Training
```powershell
python train_crater_model.py
```

### 4. Monitor Training
Training progress will be displayed in the terminal. Results will be saved in:
```
runs/detect/train_100/
├── weights/
│   ├── best.pt         # Best model weights
│   └── last.pt         # Last epoch weights
├── results.png         # Training metrics plot
├── confusion_matrix.png
├── F1_curve.png
├── PR_curve.png
├── P_curve.png
├── R_curve.png
└── args.yaml           # Training arguments
```

## Training Parameters

The script is configured with:
- **Epochs**: 100
- **Batch Size**: 16
- **Image Size**: 640x640
- **Model**: YOLOv8n (nano) - fastest
- **Early Stopping**: Patience of 50 epochs
- **Checkpoints**: Saved every 10 epochs

### Model Size Options
You can change `MODEL_SIZE` in the script:
- `yolov8n.pt` - Nano (fastest, smallest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (most accurate)

## After Training

### Use New Weights in Web App
Copy the trained weights to your Django project:
```powershell
Copy-Item "runs/detect/train_100/weights/last.pt" -Destination "lunar_crater_detector/train 100/weights/"
```

Then update `detector/views.py` to use the new weights:
```python
model_path = os.path.join(
    settings.BASE_DIR,
    'train 100', 'weights', 'last.pt'
)
```

### Validate Model
Uncomment validation lines in the script to test on validation set.

### Export Model
Uncomment export lines to convert to ONNX, TorchScript, etc.

## Tips for Better Training

1. **Data Augmentation**: Already enabled (flip, scale, HSV)
2. **Learning Rate**: Adjust `lr0` if loss doesn't decrease
3. **Batch Size**: Increase if you have more GPU memory
4. **Image Size**: Try 1280 for better accuracy (slower)
5. **Resume Training**: Set `resume=True` to continue from last checkpoint

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` to 8 or 4
- Reduce `IMAGE_SIZE` to 416

### Poor Results
- Check label format is correct
- Increase `EPOCHS` to 200-300
- Use larger model (yolov8m or yolov8l)
- Add more training data

### Training Too Slow
- Use smaller model (yolov8n)
- Reduce `IMAGE_SIZE`
- Increase `BATCH_SIZE` if possible

## Monitor with TensorBoard (Optional)

Install TensorBoard:
```powershell
pip install tensorboard
```

Run TensorBoard:
```powershell
tensorboard --logdir runs/detect
```

Open browser at: `http://localhost:6006`
