from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.conf import settings
from .models import UploadedImage
from .forms import ImageUploadForm, CraterSearchForm
import os
import cv2
import numpy as np
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


def crater_search(request):
    """Search for a crater in moon.tif using SIFT, SURF, and ORB"""
    if request.method == 'POST':
        form = CraterSearchForm(request.POST, request.FILES)
        if form.is_valid():
            search_image = request.FILES['search_image']
            
            # Save uploaded search image temporarily
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, search_image.name)
            
            with open(temp_path, 'wb+') as destination:
                for chunk in search_image.chunks():
                    destination.write(chunk)
            
            # Path to moon.tif
            moon_path = os.path.join(settings.MEDIA_ROOT, 'lunar dataset', 'moon.tif')
            
            if not os.path.exists(moon_path):
                messages.error(request, 'Moon dataset (moon.tif) not found!')
                return redirect('crater_search')
            
            try:
                # Perform matching with all three methods
                results = perform_crater_matching(temp_path, moon_path)
                
                # Save result images
                result_files = {}
                results_dir = os.path.join(settings.MEDIA_ROOT, 'search_results')
                os.makedirs(results_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                for method, result_img in results.items():
                    if result_img is not None:
                        result_filename = f'{method}_match_{timestamp}.jpg'
                        result_path = os.path.join(results_dir, result_filename)
                        cv2.imwrite(result_path, result_img)
                        result_files[method] = os.path.join('search_results', result_filename)
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                return render(request, 'detector/search_result.html', {
                    'results': result_files,
                    'search_image': search_image.name
                })
                
            except Exception as e:
                messages.error(request, f'Error during crater matching: {str(e)}')
                return redirect('crater_search')
    else:
        form = CraterSearchForm()
    
    return render(request, 'detector/crater_search.html', {'form': form})


def perform_crater_matching(query_path, moon_path):
    """Perform crater matching using SIFT, SURF, and ORB algorithms"""
    # Read images
    query_img = cv2.imread(query_path)
    moon_img = cv2.imread(moon_path)
    
    if query_img is None or moon_img is None:
        raise ValueError('Failed to read images')
    
    # Convert to grayscale
    query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    moon_gray = cv2.cvtColor(moon_img, cv2.COLOR_BGR2GRAY)
    
    results = {}
    
    # 1. SIFT (Scale-Invariant Feature Transform)
    try:
        sift = cv2.SIFT_create()
        kp1_sift, des1_sift = sift.detectAndCompute(query_gray, None)
        kp2_sift, des2_sift = sift.detectAndCompute(moon_gray, None)
        
        if des1_sift is not None and des2_sift is not None:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches_sift = bf.knnMatch(des1_sift, des2_sift, k=2)
            
            # Apply ratio test
            good_matches_sift = []
            for match_pair in matches_sift:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches_sift.append(m)
            
            # Draw matches
            result_sift = cv2.drawMatches(query_img, kp1_sift, moon_img, kp2_sift, 
                                         good_matches_sift[:50], None, 
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            # Add text with match count
            cv2.putText(result_sift, f'SIFT: {len(good_matches_sift)} matches', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            results['SIFT'] = result_sift
    except Exception as e:
        print(f'SIFT error: {e}')
        results['SIFT'] = None
    
    # 2. SURF (Speeded-Up Robust Features) - requires opencv-contrib-python
    try:
        surf = cv2.xfeatures2d.SURF_create(400)
        kp1_surf, des1_surf = surf.detectAndCompute(query_gray, None)
        kp2_surf, des2_surf = surf.detectAndCompute(moon_gray, None)
        
        if des1_surf is not None and des2_surf is not None:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches_surf = bf.knnMatch(des1_surf, des2_surf, k=2)
            
            # Apply ratio test
            good_matches_surf = []
            for match_pair in matches_surf:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches_surf.append(m)
            
            # Draw matches
            result_surf = cv2.drawMatches(query_img, kp1_surf, moon_img, kp2_surf, 
                                         good_matches_surf[:50], None, 
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            # Add text with match count
            cv2.putText(result_surf, f'SURF: {len(good_matches_surf)} matches', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            results['SURF'] = result_surf
    except Exception as e:
        print(f'SURF error: {e}')
        results['SURF'] = None
    
    # 3. ORB (Oriented FAST and Rotated BRIEF)
    try:
        orb = cv2.ORB_create(nfeatures=5000)
        kp1_orb, des1_orb = orb.detectAndCompute(query_gray, None)
        kp2_orb, des2_orb = orb.detectAndCompute(moon_gray, None)
        
        if des1_orb is not None and des2_orb is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches_orb = bf.knnMatch(des1_orb, des2_orb, k=2)
            
            # Apply ratio test
            good_matches_orb = []
            for match_pair in matches_orb:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches_orb.append(m)
            
            # Draw matches
            result_orb = cv2.drawMatches(query_img, kp1_orb, moon_img, kp2_orb, 
                                        good_matches_orb[:50], None, 
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            # Add text with match count
            cv2.putText(result_orb, f'ORB: {len(good_matches_orb)} matches', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            results['ORB'] = result_orb
    except Exception as e:
        print(f'ORB error: {e}')
        results['ORB'] = None
    
    return results
