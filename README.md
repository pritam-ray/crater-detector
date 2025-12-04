# 🌙 Lunar Crater Detector Web Application

A beautiful Django web application for detecting lunar craters using YOLOv8 AI technology.

## 🚀 Features

- **AI-Powered Detection**: Uses YOLOv8 trained model for accurate crater detection
- **Beautiful UI**: Modern, space-themed interface with animations
- **Real-time Processing**: Upload images and get instant crater detection results
- **Image Gallery**: Browse all previously processed images
- **Detailed Results**: View original vs. processed images with crater count
- **Multiple Formats**: Supports JPG, PNG, TIF/TIFF formats

## 📋 Requirements

- Python 3.14
- Django 6.0
- Ultralytics YOLOv8
- OpenCV
- Pillow
- NumPy

## 🛠️ Installation & Setup

1. **Virtual Environment** (Already created at `venv_django`)
   ```powershell
   & d:\python\isro\crater_web_app\venv_django\Scripts\Activate.ps1
   ```

2. **All dependencies are already installed:**
   - Django 6.0
   - ultralytics (YOLOv8)
   - opencv-python
   - pillow
   - numpy

3. **Database is ready** (migrations already applied)

## 🎯 Running the Application

1. **Activate the virtual environment:**
   ```powershell
   & d:\python\isro\crater_web_app\venv_django\Scripts\Activate.ps1
   ```

2. **Start the development server:**
   ```powershell
   python manage.py runserver
   ```

3. **Open your browser and navigate to:**
   ```
   http://127.0.0.1:8000/
   ```

## 📂 Project Structure

```
crater_web_app/
├── detector/                      # Main app
│   ├── templates/detector/        # HTML templates
│   │   ├── base.html             # Base template with navigation
│   │   ├── index.html            # Upload page
│   │   ├── result.html           # Detection results
│   │   └── gallery.html          # Image gallery
│   ├── models.py                 # Database models
│   ├── views.py                  # View logic
│   ├── forms.py                  # Upload form
│   ├── urls.py                   # URL routing
│   └── admin.py                  # Admin interface
├── lunar_crater_detector/         # Project settings
│   ├── settings.py               # Django settings
│   ├── urls.py                   # Main URL config
│   └── wsgi.py                   # WSGI config
├── media/                         # Uploaded & processed images
│   ├── uploads/                  # Original images
│   └── results/                  # Processed images
├── venv_django/                   # Virtual environment
└── manage.py                      # Django management script
```

## 🎨 Pages

### 1. Home Page (Upload)
- Drag & drop or browse to upload lunar images
- Preview before processing
- Recent detections display
- Feature cards explaining the technology

### 2. Result Page
- Side-by-side comparison (Original vs. Detected)
- Crater count statistics
- Processing time and date
- Download options for both images

### 3. Gallery Page
- Grid view of all processed images
- Crater count badges
- Upload dates and times
- Click to view full results

## 🤖 YOLO Model Configuration

The app uses your trained YOLOv8 model located at:
```
d:\python\isro\runs\detect\train 55\weights\last.pt
```

- **Threshold**: 0.1 (configurable in `views.py`)
- **Detection**: Draws green circles around detected craters
- **Labeling**: Each crater is numbered sequentially

## 🎯 Usage

1. **Upload an image** on the home page
2. **Click "Detect Craters"** to process
3. **View results** with crater annotations
4. **Download** processed images
5. **Browse gallery** to see all detections

## 🔧 Admin Panel

Access the Django admin panel at `http://127.0.0.1:8000/admin/`

Create a superuser first:
```powershell
python manage.py createsuperuser
```

## 🎨 Customization

### Change Detection Threshold
Edit `detector/views.py`, line with `threshold = 0.1`

### Modify Styles
Edit CSS in template files or add custom CSS files

### Change Upload Limits
Edit `detector/forms.py`, modify `clean_original_image()` method

## 📝 Notes

- Maximum upload size: 10MB
- Supported formats: JPG, JPEG, PNG, TIF, TIFF
- Model path is relative to project parent directory
- Media files are served in development mode only

## 🌟 Features Highlights

- **Animated starry background**
- **Gradient color scheme**
- **Responsive design**
- **Drag & drop upload**
- **Real-time preview**
- **Modern card-based UI**
- **Smooth transitions**

## 🐛 Troubleshooting

**Model not found error:**
- Verify the model path in `views.py` line 38
- Check that `train 55` folder exists in `../runs/detect/`

**Upload errors:**
- Check file size (< 10MB)
- Verify file format (JPG, PNG, TIF)
- Ensure media directories exist

**Import errors:**
- Reactivate virtual environment
- Check all packages are installed

## 📄 License

ISRO Hackathon Project - 2025

---

**Powered by YOLOv8 | Built with Django | ISRO Chandrayaan-2 TMC Compatible**
