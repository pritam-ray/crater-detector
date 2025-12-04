from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.conf import settings
from .models import UploadedImage
from .forms import ImageUploadForm
import os
import cv2
from ultralytics import YOLO
from datetime import datetime
from PIL import Image as PILImage


def index(request):
    """Home page with upload form"""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_image = form.save()
            # Redirect to processing page
            return redirect('process_image', image_id=uploaded_image.id)
    else:
        form = ImageUploadForm()
    
    # Get recent uploads
    recent_uploads = UploadedImage.objects.all()[:6]
    
    return render(request, 'detector/index.html', {
        'form': form,
        'recent_uploads': recent_uploads
    })


def process_image(request, image_id):
    """Process the uploaded image with YOLO crater detection"""
    uploaded_image = get_object_or_404(UploadedImage, id=image_id)
    
    # Path to the trained YOLO model (now in lunar_crater_detector folder)
    model_path = os.path.join(
        settings.BASE_DIR,
        'train 55', 'weights', 'last.pt'
    )
    
    # Check if model exists
    if not os.path.exists(model_path):
        messages.error(request, f'YOLO model not found at: {model_path}')
        return redirect('index')
    
    try:
        # Load YOLO model
        model = YOLO(model_path)
        
        # Get the path to the uploaded image
        image_path = uploaded_image.original_image.path
        
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            messages.error(request, 'Error reading the uploaded image')
            return redirect('index')
        
        # Perform detection
        results = model(frame)[0]
        
        # Detection threshold
        threshold = 0.1
        crater_count = 0
        
        # Process each detection result
        if results:
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                
                if score > threshold:
                    # Calculate the center and radius for the circle
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    radius = int(max((x2 - x1), (y2 - y1)) / 2)
                    
                    crater_count += 1
                    
                    # Draw the circle
                    cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 4)
                    text_to_display = f"CRATER {crater_count}"
                    cv2.putText(frame, text_to_display, (center_x, center_y - radius - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Save the processed image
        result_filename = f"processed_{uploaded_image.id}_{os.path.basename(image_path)}"
        result_path = os.path.join(settings.MEDIA_ROOT, 'results', result_filename)
        
        cv2.imwrite(result_path, frame)
        
        # Update the model instance
        uploaded_image.processed_image = os.path.join('results', result_filename)
        uploaded_image.processed_at = datetime.now()
        uploaded_image.crater_count = crater_count
        uploaded_image.save()
        
        messages.success(request, f'Successfully detected {crater_count} craters!')
        return redirect('result', image_id=uploaded_image.id)
        
    except Exception as e:
        messages.error(request, f'Error processing image: {str(e)}')
        return redirect('index')


def result(request, image_id):
    """Display the processed result"""
    uploaded_image = get_object_or_404(UploadedImage, id=image_id)
    
    return render(request, 'detector/result.html', {
        'uploaded_image': uploaded_image
    })


def gallery(request):
    """Gallery view showing all processed images"""
    images = UploadedImage.objects.filter(processed_image__isnull=False)
    
    return render(request, 'detector/gallery.html', {
        'images': images
    })


def delete_image(request, image_id):
    """Delete an uploaded image and its processed result"""
    uploaded_image = get_object_or_404(UploadedImage, id=image_id)
    
    try:
        # Delete the original image file
        if uploaded_image.original_image and os.path.exists(uploaded_image.original_image.path):
            os.remove(uploaded_image.original_image.path)
        
        # Delete the processed image file
        if uploaded_image.processed_image and os.path.exists(uploaded_image.processed_image.path):
            os.remove(uploaded_image.processed_image.path)
        
        # Delete the database record
        uploaded_image.delete()
        
        messages.success(request, 'Image deleted successfully!')
    except Exception as e:
        messages.error(request, f'Error deleting image: {str(e)}')
    
    # Redirect to gallery or index based on referrer
    referer = request.META.get('HTTP_REFERER', '')
    if 'gallery' in referer:
        return redirect('gallery')
    else:
        return redirect('index')
