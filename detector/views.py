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
                    text_to_display = f"{crater_count}"
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
    """Search for a crater in moon.tif using SIFT, SURF, and ORB - automatically detects first crater"""
    if request.method == 'POST':
        form = CraterSearchForm(request.POST, request.FILES)
        if form.is_valid():
            search_image = request.FILES['search_image']
            
            # Get selected algorithm
            selected_algorithm = form.cleaned_data.get('algorithm', 'orb')
            
            print("\n" + "="*60)
            print("CRATER SEARCH - DETAILED PROCESS LOG")
            print("="*60)
            print(f"Selected Algorithm: {selected_algorithm.upper()}")
            print(f"Uploaded Image: {search_image.name}")
            
            # Save uploaded search image temporarily
            temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, search_image.name)
            
            with open(temp_path, 'wb+') as destination:
                for chunk in search_image.chunks():
                    destination.write(chunk)
            
            print(f"Temporary file saved: {temp_path}")
            
            # Path to moon.tif
            moon_path = os.path.join(settings.MEDIA_ROOT, 'lunar dataset', 'moon.tif')
            
            if not os.path.exists(moon_path):
                print("ERROR: Moon dataset not found!")
                messages.error(request, 'Moon dataset (moon.tif) not found!')
                return redirect('crater_search')
            
            print(f"Moon dataset loaded: {moon_path}")
            
            try:
                # Detect first crater and perform matching
                result_data = detect_and_match_crater(temp_path, moon_path, selected_algorithm)
                
                if result_data is None:
                    print("ERROR: No crater detected in uploaded image")
                    print("="*60 + "\n")
                    messages.error(request, 'No crater detected in the uploaded image. Please upload an image with at least one crater.')
                    return redirect('crater_search')
                
                print(f"\n✓ Crater matching completed successfully!")
                print(f"Best Algorithm: {result_data['best_method']}")
                print(f"Total Matches: {result_data['match_count']}")
                if result_data['crater_location']:
                    print(f"Crater Location: X={result_data['crater_location'][0]}, Y={result_data['crater_location'][1]}")
                
                # Save result image
                results_dir = os.path.join(settings.MEDIA_ROOT, 'search_results')
                os.makedirs(results_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_filename = f'crater_match_{timestamp}.jpg'
                result_path = os.path.join(results_dir, result_filename)
                
                # Save with reduced quality for web display
                cv2.imwrite(result_path, result_data['result_image'], [cv2.IMWRITE_JPEG_QUALITY, 70])
                print(f"Result saved: {result_path}")
                print("="*60 + "\n")
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                return render(request, 'detector/search_result.html', {
                    'result_image': os.path.join('search_results', result_filename),
                    'best_method': result_data['best_method'],
                    'match_count': result_data['match_count'],
                    'crater_location': result_data['crater_location'],
                    'all_methods': result_data['all_methods'],
                    'search_image': search_image.name
                })
                
            except Exception as e:
                messages.error(request, f'Error during crater matching: {str(e)}')
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return redirect('crater_search')
    else:
        form = CraterSearchForm()
    
    return render(request, 'detector/crater_search.html', {'form': form})


def detect_and_match_crater(query_path, moon_path, selected_algorithm='all'):
    """Detect first crater in query image, crop it, and match using selected algorithm(s)"""
    
    print("\n[STEP 1] Loading YOLO model for crater detection...")
    # Load YOLO model for crater detection
    model_path = os.path.join(settings.BASE_DIR, 'train 55', 'weights', 'last.pt')
    if not os.path.exists(model_path):
        raise ValueError('YOLO model not found')
    
    model = YOLO(model_path)
    print("✓ YOLO model loaded successfully")
    
    print("\n[STEP 2] Reading and analyzing query image...")
    # Read query image and detect craters
    query_img = cv2.imread(query_path)
    if query_img is None:
        raise ValueError('Failed to read query image')
    
    print(f"✓ Query image loaded: {query_img.shape[1]}x{query_img.shape[0]} pixels")
    
    # Perform crater detection
    print("\n[STEP 3] Detecting craters in uploaded image...")
    results = model(query_img)[0]
    threshold = 0.1
    
    # Find first detected crater
    first_crater = None
    crater_count = 0
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            crater_count += 1
            if first_crater is None:
                first_crater = (int(x1), int(y1), int(x2), int(y2))
                print(f"✓ First crater detected at position: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
                print(f"  Confidence: {score:.2%}")
    
    print(f"✓ Total craters detected: {crater_count}")
    
    if first_crater is None:
        print("✗ No craters found above confidence threshold")
        return None
    
    # Crop the first crater
    x1, y1, x2, y2 = first_crater
    crater_img = query_img[y1:y2, x1:x2].copy()
    print(f"✓ Crater cropped: {x2-x1}x{y2-y1} pixels")
    
    print("\n[STEP 4] Loading moon.tif dataset...")
    # Read moon.tif
    moon_img = cv2.imread(moon_path, cv2.IMREAD_UNCHANGED)
    if moon_img is None:
        raise ValueError('Failed to read moon.tif')
    
    print(f"✓ Moon dataset loaded: {moon_img.shape[1]}x{moon_img.shape[0]} pixels")
    
    # Handle different image formats for moon.tif
    if len(moon_img.shape) == 2:
        moon_gray = moon_img
        moon_img_bgr = cv2.cvtColor(moon_img, cv2.COLOR_GRAY2BGR)
        print("  Format: Grayscale (2D)")
    elif len(moon_img.shape) == 3:
        if moon_img.shape[2] == 1:
            moon_gray = moon_img[:, :, 0]
            moon_img_bgr = cv2.cvtColor(moon_gray, cv2.COLOR_GRAY2BGR)
            print("  Format: Grayscale (3D single channel)")
        elif moon_img.shape[2] == 3:
            moon_img_bgr = moon_img.copy()
            moon_gray = cv2.cvtColor(moon_img, cv2.COLOR_BGR2GRAY)
            print("  Format: BGR (3 channels)")
        elif moon_img.shape[2] == 4:
            moon_img_bgr = cv2.cvtColor(moon_img, cv2.COLOR_BGRA2BGR)
            moon_gray = cv2.cvtColor(moon_img_bgr, cv2.COLOR_BGR2GRAY)
            print("  Format: BGRA (4 channels)")
        else:
            moon_gray = moon_img[:, :, 0]
            moon_img_bgr = cv2.cvtColor(moon_gray, cv2.COLOR_GRAY2BGR)
            print(f"  Format: Unknown ({moon_img.shape[2]} channels)")
    else:
        raise ValueError(f'Unexpected moon image shape: {moon_img.shape}')
    
    # Convert crater crop to grayscale
    crater_gray = cv2.cvtColor(crater_img, cv2.COLOR_BGR2GRAY)
    print("✓ Images converted to grayscale for feature matching")
    
    # Dictionary to store results from all methods
    all_matches = {}
    
    print(f"\n[STEP 5] Running feature matching with {selected_algorithm.upper()} algorithm(s)...")
    
    # 1. SIFT Matching
    if selected_algorithm in ['all', 'sift']:
        print("\n  → Running SIFT (Scale-Invariant Feature Transform)...")
        try:
            sift = cv2.SIFT_create()
            kp1_sift, des1_sift = sift.detectAndCompute(crater_gray, None)
            kp2_sift, des2_sift = sift.detectAndCompute(moon_gray, None)
            
            print(f"    Keypoints detected - Query: {len(kp1_sift)}, Moon: {len(kp2_sift)}")
            
            if des1_sift is not None and des2_sift is not None:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                matches_sift = bf.knnMatch(des1_sift, des2_sift, k=2)
                
                good_matches_sift = []
                for match_pair in matches_sift:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches_sift.append(m)
                
                all_matches['SIFT'] = {
                    'matches': good_matches_sift,
                    'keypoints_query': kp1_sift,
                    'keypoints_moon': kp2_sift,
                    'count': len(good_matches_sift)
                }
                print(f"    ✓ SIFT completed: {len(good_matches_sift)} good matches found")
            else:
                all_matches['SIFT'] = {'count': 0}
                print("    ✗ SIFT failed: No descriptors computed")
        except Exception as e:
            print(f'    ✗ SIFT error: {e}')
            all_matches['SIFT'] = {'count': 0}
    
    # 2. SURF Matching
    if selected_algorithm in ['all', 'surf']:
        print("\n  → Running SURF (Speeded-Up Robust Features)...")
        try:
            surf = cv2.xfeatures2d.SURF_create(400)
            kp1_surf, des1_surf = surf.detectAndCompute(crater_gray, None)
            kp2_surf, des2_surf = surf.detectAndCompute(moon_gray, None)
            
            print(f"    Keypoints detected - Query: {len(kp1_surf)}, Moon: {len(kp2_surf)}")
            
            if des1_surf is not None and des2_surf is not None:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                matches_surf = bf.knnMatch(des1_surf, des2_surf, k=2)
                
                good_matches_surf = []
                for match_pair in matches_surf:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches_surf.append(m)
                
                all_matches['SURF'] = {
                    'matches': good_matches_surf,
                    'keypoints_query': kp1_surf,
                    'keypoints_moon': kp2_surf,
                    'count': len(good_matches_surf)
                }
                print(f"    ✓ SURF completed: {len(good_matches_surf)} good matches found")
            else:
                all_matches['SURF'] = {'count': 0}
                print("    ✗ SURF failed: No descriptors computed")
        except Exception as e:
            print(f'    ✗ SURF error: {e}')
            all_matches['SURF'] = {'count': 0}
    
    # 3. ORB Matching
    if selected_algorithm in ['all', 'orb']:
        print("\n  → Running ORB (Oriented FAST and Rotated BRIEF)...")
        try:
            orb = cv2.ORB_create(nfeatures=5000)
            kp1_orb, des1_orb = orb.detectAndCompute(crater_gray, None)
            kp2_orb, des2_orb = orb.detectAndCompute(moon_gray, None)
            
            print(f"    Keypoints detected - Query: {len(kp1_orb)}, Moon: {len(kp2_orb)}")
            
            if des1_orb is not None and des2_orb is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches_orb = bf.knnMatch(des1_orb, des2_orb, k=2)
                
                good_matches_orb = []
                for match_pair in matches_orb:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches_orb.append(m)
                
                all_matches['ORB'] = {
                    'matches': good_matches_orb,
                    'keypoints_query': kp1_orb,
                    'keypoints_moon': kp2_orb,
                    'count': len(good_matches_orb)
                }
                print(f"    ✓ ORB completed: {len(good_matches_orb)} good matches found")
            else:
                all_matches['ORB'] = {'count': 0}
                print("    ✗ ORB failed: No descriptors computed")
        except Exception as e:
            print(f'    ✗ ORB error: {e}')
            all_matches['ORB'] = {'count': 0}
    
    print("\n[STEP 6] Analyzing results and selecting best match...")
    
    # Determine best method based on match count
    if not all_matches:
        print("✗ No algorithms were executed")
        return None
    
    best_method = max(all_matches.items(), key=lambda x: x[1]['count'])
    best_method_name = best_method[0]
    best_data = best_method[1]
    
    print(f"\nResults Summary:")
    for method, data in all_matches.items():
        print(f"  {method}: {data['count']} matches")
    print(f"\n✓ Best performing algorithm: {best_method_name} with {best_data['count']} matches")
    
    if best_data['count'] == 0:
        print("✗ No valid matches found with any algorithm")
        return None
    
    # Calculate crater location in moon image from best matches
    print("\n[STEP 7] Calculating crater location on moon surface...")
    crater_location = None
    if 'matches' in best_data and len(best_data['matches']) > 0:
        # Get average location from top matches
        match_points = []
        for match in best_data['matches'][:10]:  # Use top 10 matches
            pt = best_data['keypoints_moon'][match.trainIdx].pt
            match_points.append(pt)
        
        if match_points:
            avg_x = int(sum(p[0] for p in match_points) / len(match_points))
            avg_y = int(sum(p[1] for p in match_points) / len(match_points))
            crater_location = (avg_x, avg_y)
            print(f"✓ Crater location computed from top {len(match_points)} matches")
    
    print("\n[STEP 8] Preparing visualization...")
    # Resize moon image for web display (reduce to 2000px width max)
    max_width = 2000
    if moon_img_bgr.shape[1] > max_width:
        scale = max_width / moon_img_bgr.shape[1]
        new_width = max_width
        new_height = int(moon_img_bgr.shape[0] * scale)
        moon_display = cv2.resize(moon_img_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
        print(f"✓ Moon image resized: {moon_img_bgr.shape[1]}x{moon_img_bgr.shape[0]} → {new_width}x{new_height}")
        
        # Scale keypoints and location
        if crater_location:
            crater_location = (int(crater_location[0] * scale), int(crater_location[1] * scale))
    else:
        moon_display = moon_img_bgr.copy()
        print("✓ Using original moon image size")
    
    # Draw matched region on moon image
    if crater_location and 'matches' in best_data:
        # Draw circle at detected location
        radius = max(50, int(max(x2-x1, y2-y1) * 0.5))
        if moon_display.shape[1] != moon_img_bgr.shape[1]:
            radius = int(radius * (moon_display.shape[1] / moon_img_bgr.shape[1]))
        
        cv2.circle(moon_display, crater_location, radius, (0, 255, 0), 5)
        cv2.putText(moon_display, f'Matched Location', 
                   (crater_location[0] - 80, crater_location[1] - radius - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    # Add method info overlay
    cv2.rectangle(moon_display, (10, 10), (500, 120), (0, 0, 0), -1)
    cv2.rectangle(moon_display, (10, 10), (500, 120), (0, 255, 0), 2)
    cv2.putText(moon_display, f'Best Match: {best_method_name}', 
               (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(moon_display, f'Matches: {best_data["count"]}', 
               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Prepare method comparison data
    methods_data = {method: data['count'] for method, data in all_matches.items()}
    
    return {
        'result_image': moon_display,
        'best_method': best_method_name,
        'match_count': best_data['count'],
        'crater_location': crater_location,
        'all_methods': methods_data
    }
