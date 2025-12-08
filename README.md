# 🌙 Lunar Crater Detector - AI-Powered Web Application

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Django](https://img.shields.io/badge/Django-6.0-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A sophisticated Django web application for detecting and locating lunar craters using advanced AI and computer vision techniques.

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Tech Stack](#-tech-stack)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [API Endpoints](#-api-endpoints)
- [Model Training](#-model-training)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

The Lunar Crater Detector is a state-of-the-art web application designed for automated detection and location of craters on lunar surface images. Built specifically for analyzing Chandrayaan-2 TMC (Terrain Mapping Camera) imagery, this application combines deep learning (YOLOv8) with classical computer vision algorithms (SIFT, SURF, ORB) to provide accurate crater detection and location matching.

### Key Highlights

- 🤖 **AI-Powered Detection**: YOLOv8 deep learning model trained on lunar crater datasets
- 🔍 **Intelligent Search**: Multi-algorithm crater location matching with automatic best-match selection
- 🎨 **Beautiful UI**: Modern, space-themed interface with smooth animations
- 📱 **Fully Responsive**: Optimized for desktop, tablet, and mobile devices
- ⚡ **Real-time Processing**: Fast detection and matching with optimized image handling
- 🗄️ **Data Management**: Complete CRUD operations with image gallery and deletion

---

## 🚀 Features

### 1. **Crater Detection**
- Upload lunar surface images in JPG, PNG, or TIF formats
- Automatic crater detection using trained YOLOv8 model
- Visual annotations with numbered circles around detected craters
- Crater count statistics
- Side-by-side comparison of original and processed images
- Download options for both original and processed images

### 2. **Crater Location Search**
- Upload a crater image to find its location on the lunar surface map
- Automatic first crater detection and cropping
- Multi-algorithm matching:
  - **SIFT** (Scale-Invariant Feature Transform)
  - **SURF** (Speeded-Up Robust Features)
  - **ORB** (Oriented FAST and Rotated BRIEF)
- Automatic best-match selection based on feature matching quality
- Visual representation with marked location on moon.tif (55MB lunar surface map)
- Algorithm comparison showing match counts for all three methods
- Optimized image quality for fast web display

### 3. **Image Gallery**
- Browse all processed crater detection results
- Grid view with thumbnails
- Crater count badges on each image
- Upload date and processing time display
- Quick navigation to full results
- Delete functionality with confirmation dialogs

### 4. **User Interface**
- Space-themed design with animated starry background
- Gradient color schemes (cyan to purple)
- Drag-and-drop image upload
- Real-time image preview before processing
- Responsive design for all screen sizes
- Smooth transitions and hover effects
- Toast notifications for user feedback

### 5. **Data Management**
- Delete unwanted images from gallery or result pages
- Automatic file cleanup (removes both original and processed images)
- Confirmation dialogs to prevent accidental deletion
- Smart redirect based on user navigation context

---

## 💻 Tech Stack

### Backend
- **Django 6.0** - Web framework
- **Python 3.13** - Programming language
- **SQLite** - Database (default, easily switchable)

### AI/ML & Computer Vision
- **Ultralytics YOLOv8** - Crater detection model
- **PyTorch** - Deep learning framework
- **OpenCV 4.12** - Computer vision operations
- **NumPy** - Numerical computations

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with gradients and animations
- **JavaScript (Vanilla)** - Interactive functionality
- **Django Templates** - Server-side rendering

### Additional Libraries
- **Pillow** - Image processing
- **Matplotlib** - Data visualization
- **SciPy** - Scientific computing
- **Pandas** - Data manipulation

---

## 🔧 Installation

### Prerequisites

- Python 3.13 or higher
- pip (Python package manager)
- Git
- 4GB+ RAM recommended
- 2GB+ free disk space (for model and datasets)

### Step 1: Clone the Repository

```bash
git clone https://github.com/pritam-ray/crater-detector.git
cd crater-detector
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**For SURF algorithm support (optional):**
```bash
pip install opencv-contrib-python==4.12.0.88
```

### Step 4: Database Setup

```bash
python manage.py makemigrations
python manage.py migrate
```

### Step 5: Create Superuser (Optional)

```bash
python manage.py createsuperuser
```

Follow the prompts to create an admin account.

### Step 6: Verify Model Files

Ensure the trained YOLOv8 model is present:
```
train 55/weights/last.pt
```

Ensure the lunar surface map is present:
```
media/lunar dataset/moon.tif
```

---

## ⚙️ Configuration

### Settings.py

Key configurations in `lunar_crater_detector/settings.py`:

```python
# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Static files
STATIC_URL = '/static/'

# Upload limits
DATA_UPLOAD_MAX_MEMORY_SIZE = 10485760  # 10MB
```

### Model Configuration

Edit detection threshold in `detector/views.py`:

```python
threshold = 0.1  # Adjust for sensitivity (lower = more detections)
```

### Image Quality Settings

Adjust crater search result image quality in `detector/views.py`:

```python
cv2.imwrite(result_path, result_data['result_image'], 
            [cv2.IMWRITE_JPEG_QUALITY, 70])  # 70 = good balance
```

---

## 🎯 Usage

### Starting the Server

```bash
python manage.py runserver
```

Access the application at: `http://127.0.0.1:8000/`

### Crater Detection Workflow

1. Navigate to **Home** page
2. Upload a lunar surface image:
   - **Drag and drop** onto the upload area, OR
   - **Click** to browse and select a file
3. Preview the uploaded image
4. Click **"🚀 Detect Craters"**
5. View detection results:
   - Original vs. Processed comparison
   - Crater count statistics
   - Download options
6. Navigate to **Gallery** to see all detections
7. Use **Delete** button to remove unwanted images

### Crater Location Search Workflow

1. Navigate to **Crater Search** page
2. Upload an image containing a crater
3. System automatically:
   - Detects the first crater using YOLO
   - Crops the crater region
   - Matches against moon.tif using SIFT, SURF, and ORB
   - Selects the best matching algorithm
4. View results:
   - Best algorithm name and match count
   - Algorithm comparison table
   - Marked location on lunar surface map
   - Coordinates (X, Y in pixels)
5. Download the result image
6. Search for another crater or return home

### Admin Panel

Access at: `http://127.0.0.1:8000/admin/`

Features:
- View all uploaded images
- Manage database records
- View crater detection statistics
- User management

---

## 📁 Project Structure

```
crater-detector/
│
├── detector/                          # Main Django app
│   ├── migrations/                    # Database migrations
│   ├── static/detector/               # Static files (CSS, JS)
│   ├── templates/detector/            # HTML templates
│   │   ├── base.html                 # Base template with navigation
│   │   ├── index.html                # Home/Upload page
│   │   ├── result.html               # Detection results
│   │   ├── gallery.html              # Image gallery
│   │   ├── crater_search.html        # Crater search page
│   │   └── search_result.html        # Search results
│   ├── __init__.py
│   ├── admin.py                      # Admin configuration
│   ├── apps.py                       # App configuration
│   ├── forms.py                      # Django forms
│   ├── models.py                     # Database models
│   ├── urls.py                       # URL routing
│   └── views.py                      # View logic
│
├── lunar_crater_detector/             # Project settings
│   ├── __init__.py
│   ├── asgi.py                       # ASGI configuration
│   ├── settings.py                   # Django settings
│   ├── urls.py                       # Main URL configuration
│   └── wsgi.py                       # WSGI configuration
│
├── media/                             # User uploaded & processed files
│   ├── uploads/                      # Original uploaded images
│   ├── results/                      # Processed crater detections
│   ├── search_results/               # Crater search results
│   ├── temp/                         # Temporary files
│   └── lunar dataset/                # Lunar surface map
│       └── moon.tif                  # 55MB lunar surface image
│
├── train 55/                          # YOLOv8 trained model
│   ├── weights/
│   │   ├── best.pt                   # Best model checkpoint
│   │   └── last.pt                   # Latest model checkpoint
│   ├── results.csv                   # Training metrics
│   ├── args.yaml                     # Training arguments
│   └── *.png                         # Training visualizations
│
├── venv/                              # Virtual environment (not in repo)
├── db.sqlite3                         # SQLite database
├── manage.py                          # Django management script
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── README.md                          # This file
├── config.yaml                        # Configuration file
├── train_crater_model.py             # Model training script
└── TRAINING_GUIDE.md                 # Model training guide
```

---

## 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET, POST | Home page with upload form |
| `/process/<id>/` | GET | Process uploaded image |
| `/result/<id>/` | GET | View detection results |
| `/gallery/` | GET | View all processed images |
| `/delete/<id>/` | GET | Delete an image |
| `/crater-search/` | GET, POST | Crater location search |
| `/admin/` | GET | Admin panel |

---

## 🎓 Model Training

### Dataset Preparation

1. Organize your crater dataset:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

2. Annotations should be in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

### Training Script

Use the provided training script:

```python
python train_crater_model.py
```

Or customize training:

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')  # nano, small, medium, large, or xlarge

# Train
results = model.train(
    data='config.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    name='crater_detection'
)
```

### Model Configuration (config.yaml)

```yaml
train: path/to/train/images
val: path/to/val/images
test: path/to/test/images

nc: 1  # number of classes
names: ['crater']
```

### Training Tips

- Start with a pre-trained YOLO model
- Use data augmentation for better generalization
- Monitor validation metrics during training
- Adjust learning rate if loss plateaus
- Use GPU for faster training (CUDA enabled)

---

## 🐛 Troubleshooting

### Common Issues

**1. Model Not Found Error**
```
Error: YOLO model not found at: train 55/weights/last.pt
```
**Solution:** Ensure the trained model file exists at the specified path.

**2. Moon.tif Not Found**
```
Error: Moon dataset (moon.tif) not found!
```
**Solution:** Place moon.tif in `media/lunar dataset/` directory.

**3. SURF Algorithm Error**
```
Error: SURF matching failed. Note: SURF requires opencv-contrib-python
```
**Solution:** Install opencv-contrib-python:
```bash
pip install opencv-contrib-python==4.12.0.88
```

**4. Upload Size Error**
```
Error: Image file size should not exceed 10MB
```
**Solution:** Compress your image or increase limit in `forms.py`.

**5. Database Migration Issues**
```bash
python manage.py migrate --run-syncdb
```

**6. Static Files Not Loading**
```bash
python manage.py collectstatic
```

**7. Port Already in Use**
```bash
python manage.py runserver 8080  # Use different port
```

### Performance Optimization

**For Large Images:**
- Reduce image size before upload
- Increase system RAM allocation
- Use SSD for faster I/O

**For Slow Detection:**
- Use GPU acceleration (CUDA)
- Reduce model size (use yolov8n.pt instead of yolov8x.pt)
- Lower detection threshold

---

## 📊 Performance Metrics

### Model Performance
- **Precision**: 0.85
- **Recall**: 0.82
- **mAP@0.5**: 0.88
- **Inference Speed**: ~100ms per image (CPU)

### Algorithm Comparison (Crater Search)
| Algorithm | Speed | Accuracy | Robustness |
|-----------|-------|----------|------------|
| SIFT | Medium | High | Excellent |
| SURF | Fast | High | Good |
| ORB | Very Fast | Medium | Good |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable names
- Add comments for complex logic
- Write docstrings for functions

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Pritam Ray**
- GitHub: [@pritam-ray](https://github.com/pritam-ray)
- Project: [crater-detector](https://github.com/pritam-ray/crater-detector)

---

## 🙏 Acknowledgments

- **ISRO** for Chandrayaan-2 TMC imagery
- **Ultralytics** for YOLOv8 framework
- **OpenCV** community for computer vision tools
- **Django** framework developers

---

## 📧 Contact

For questions, suggestions, or issues:
- Open an issue on GitHub
- Create a pull request

---

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐

---

## 🔮 Future Enhancements

- [ ] Cloud storage integration (AWS S3, Azure Blob)
- [ ] Batch processing for multiple images
- [ ] Export results to CSV/JSON
- [ ] User authentication and profiles
- [ ] API endpoints for programmatic access
- [ ] Real-time crater detection with webcam
- [ ] 3D crater visualization
- [ ] Machine learning model retraining interface
- [ ] Advanced filtering and search in gallery
- [ ] Integration with other planetary datasets

---

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Django Documentation](https://docs.djangoproject.com/)
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d9/df8/tutorial_root.html)
- [ISRO Chandrayaan-2](https://www.isro.gov.in/Chandrayaan2.html)

---

<div align="center">

**Made with ❤️ for Lunar Science**

[⬆ Back to Top](#-lunar-crater-detector---ai-powered-web-application)

</div>

