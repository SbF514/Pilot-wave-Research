import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from multiprocessing import Pool, cpu_count

print("v1.14")

def create_background_model(video_path, n_frames=20):
    """
    Create a background model by averaging multiple frames from the video.
    
    Args:
        video_path: Path to the input video
        n_frames: Number of frames to sample for background creation
        
    Returns:
        Background model image
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames at regular intervals
    sample_indices = np.linspace(0, frame_count-1, n_frames, dtype=int)
    
    # Collect frames for background model
    frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            frames.append(blurred)
    
    cap.release()
    
    if not frames:
        return None
    
    # Create background model by averaging frames
    background = np.mean(frames, axis=0).astype(np.uint8)
    
    # Enhance the background model visualization
    cv2.imwrite('background_model.png', background)
    
    return background

def detect_droplets_bg_subtraction(frame, background, threshold=25, min_area=15, max_area=3000):
    """
    Detect droplets using background subtraction.
    
    Args:
        frame: Current frame
        background: Background model
        threshold: Threshold for background subtraction (lowered to detect more subtle droplets)
        min_area: Minimum area for droplet detection (lowered to catch smaller droplets)
        max_area: Maximum area for droplet detection (increased for possible larger structures)
        
    Returns:
        List of detected droplets as (x, y, radius) tuples and subtracted image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Subtract background
    subtracted = cv2.absdiff(background, blurred)
    
    # Apply threshold to get binary image
    _, thresholded = cv2.threshold(subtracted, threshold, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and convert to droplet format
    droplets = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            # Find enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y, radius = int(x), int(y), int(radius)
            droplets.append((x, y, radius))
    
    # Enhance visualization of subtracted image
    colormap_subtracted = cv2.applyColorMap(subtracted, cv2.COLORMAP_JET)
    cv2.imwrite('subtracted_example.png', colormap_subtracted)
    
    return droplets, subtracted

def process_video(video_path, output_csv, display=True, sample_rate=1):
    """
    Process video to detect droplets using background subtraction and save data to CSV.
    
    Args:
        video_path: Path to the input video file
        output_csv: Path to save the output CSV file
        display: Whether to display processing results
        sample_rate: Process every Nth frame (for speed)
    """
    # Create background model
    print("Creating background model...")
    background = create_background_model(video_path)
    
    if background is None:
        print("Failed to create background model")
        return
    
    # Display background model
    if display:
        plt.figure(figsize=(10, 8))
        plt.imshow(background, cmap='gray')
        plt.title('Background Model')
        plt.axis('off')
        plt.show()
        
        # Save background model
        cv2.imwrite('background_model.png', background)
        print("Background model saved as 'background_model.png'")
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")
    
    # Initialize data storage
    all_data = []
    
    # Create output directory for subtracted frames if display is enabled
    if display:
        output_dir = "subtracted_frames"
        os.makedirs(output_dir, exist_ok=True)
    
    # Process frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process only every sample_rate frame
        if frame_idx % sample_rate == 0:
            time_sec = frame_idx / fps
            
            # Detect droplets using background subtraction
            droplets, subtracted = detect_droplets_bg_subtraction(frame, background)
            
            # Store data for each droplet
            for idx, (x, y, radius) in enumerate(droplets):
                all_data.append({
                    'frame': frame_idx,
                    'time_sec': time_sec,
                    'droplet_id': idx,
                    'x': x,
                    'y': y,
                    'radius': radius,
                    'area': np.pi * radius * radius
                })
            
            # Visualize detection
            if display:
                # Save subtracted image for the first few frames
                if frame_idx < 10 * sample_rate:
                    subtracted_path = os.path.join(output_dir, f"subtracted_frame_{frame_idx}.png")
                    cv2.imwrite(subtracted_path, subtracted)
                
                # Create a copy for visualization
                vis_frame = frame.copy()
                
                # Draw detected droplets
                for x, y, radius in droplets:
                    cv2.circle(vis_frame, (x, y), radius, (0, 255, 0), 2)
                    cv2.circle(vis_frame, (x, y), 2, (0, 0, 255), 3)
                
                # Display frame info
                cv2.putText(
                    vis_frame, 
                    f"Frame: {frame_idx}, Droplets: {len(droplets)}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    2
                )
                
                # Create composite view with original, background, subtracted, and result
                background_color = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
                subtracted_color = cv2.cvtColor(subtracted, cv2.COLOR_GRAY2BGR)
                
                # Apply colormap to subtracted image for better visualization
                subtracted_color = cv2.applyColorMap(subtracted, cv2.COLORMAP_JET)
                
                top_row = np.hstack([frame, background_color])
                bottom_row = np.hstack([subtracted_color, vis_frame])
                composite = np.vstack([top_row, bottom_row])
                
                # Add labels to the composite view
                h, w = frame.shape[:2]
                cv2.putText(composite, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(composite, "Background Model", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(composite, "Subtracted", (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(composite, "Detected Droplets", (w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Resize for display if necessary
                scale = 1.0
                if composite.shape[1] > 1600:
                    scale = 1600 / composite.shape[1]
                    composite = cv2.resize(composite, (0, 0), fx=scale, fy=scale)
                
                # Show frame
                cv2.imshow('Droplet Detection with Background Subtraction', composite)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        frame_idx += 1
        
        # Show progress
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames ({frame_idx/frame_count*100:.1f}%)")
    
    # Release video and close windows
    cap.release()
    if display:
        cv2.destroyAllWindows()
    
    # Create DataFrame and save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)
        print(f"Saved data to {output_csv}")
        
        # Generate summary statistics
        print("\nSummary Statistics:")
        print(f"Total frames processed: {frame_idx}")
        print(f"Total droplets detected: {len(df)}")
        print(f"Average droplets per frame: {len(df) / (frame_idx/sample_rate):.2f}")
        print(f"Average droplet radius: {df['radius'].mean():.2f} pixels")
        
        return df
    else:
        print("No droplets detected")
        return None

def analyze_results(csv_path, output_dir=None):
    """
    Analyze droplet detection results.
    
    Args:
        csv_path: Path to the CSV file with detection results
        output_dir: Directory to save analysis plots
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load data
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print("No data to analyze")
        return
    
    # Count droplets per frame
    droplets_per_frame = df.groupby('frame').size()
    
    # Plot droplet count over time
    plt.figure(figsize=(12, 6))
    plt.plot(droplets_per_frame.index, droplets_per_frame.values)
    plt.title('Number of Droplets Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Droplets')
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'droplets_over_time.png'))
    plt.show()
    
    # Histogram of droplet sizes
    plt.figure(figsize=(12, 6))
    plt.hist(df['radius'], bins=30)
    plt.title('Distribution of Droplet Sizes')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Frequency')
    plt.grid(True)

def enhance_droplet_detection(frame, background):
    """
    Enhanced droplet detection specifically for radial patterns with bright center.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Subtract background
    subtracted = cv2.absdiff(background, blurred)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(subtracted)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2    # Constant subtracted from mean
    )
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    return cleaned, subtracted, enhanced

def create_center_mask(frame, border_percentage=15):
    """
    Create a mask that only includes the center region of the petri dish,
    excluding the outer ring.
    
    Args:
        frame: Input frame
        border_percentage: Percentage of dish radius to exclude from edge
        
    Returns:
        Binary mask with center region as white (255) and rest as black (0)
    """
    height, width = frame.shape[:2]
    
    # Find the center of the image (approximating center of the dish)
    center_x, center_y = width // 2, height // 2
    
    # Estimate the dish radius - assuming dish covers most of the image
    estimated_radius = min(width, height) // 2
    
    # Create a blank mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate inner radius that excludes the border
    inner_radius = int(estimated_radius * (1 - border_percentage / 100))
    
    # Draw filled white circle for the center region
    cv2.circle(mask, (center_x, center_y), inner_radius, 255, -1)
    
    return mask

def detect_center_droplets(frame, background, min_radius=5, max_radius=30, 
                           border_percentage=15, dp=1.5, param1=50, param2=30):
    """
    Detect droplets only in the center region of the petri dish.
    
    Args:
        frame: Current frame
        background: Background model
        min_radius: Minimum radius of droplets to detect
        max_radius: Maximum radius of droplets to detect
        border_percentage: Percentage of dish radius to exclude from edge
        dp: Inverse ratio of the accumulator resolution
        param1: Higher threshold for Canny edge detector
        param2: Accumulator threshold for the circle centers
        
    Returns:
        List of detected droplets as (x, y, radius) tuples and processed image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Subtract background
    subtracted = cv2.absdiff(background, blurred)
    
    # Create center mask
    center_mask = create_center_mask(frame, border_percentage)
    
    # Apply mask to exclude the dish border
    masked = cv2.bitwise_and(subtracted, subtracted, mask=center_mask)
    
    # Apply CLAHE for enhanced contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked)
    
    # Apply Hough Circle Transform to find complete circles
    circles = cv2.HoughCircles(
        enhanced,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_radius*2,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    # Create color visualization of the processing steps
    # Convert the enhanced image to color for visualization
    vis_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # If circles are found, draw them on the visualization image
    droplets = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Get circle parameters
            x, y, r = i[0], i[1], i[2]
            droplets.append((x, y, r))
            
            # Draw the outer circle
            cv2.circle(vis_enhanced, (x, y), r, (0, 255, 0), 2)
            # Draw the center
            cv2.circle(vis_enhanced, (x, y), 2, (0, 0, 255), 3)
    
    return droplets, enhanced, vis_enhanced

def detect_central_structure(frame, background, center_radius_pct=25):
    """
    Detect the central structure pattern: white dot circled by black circle circled by white circle.
    
    Args:
        frame: Current frame
        background: Background model
        center_radius_pct: Percentage of the image radius to consider as the center
        
    Returns:
        Processed image showing the central structure and coordinates
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Subtract background
    subtracted = cv2.absdiff(background, blurred)
    
    # Get image dimensions
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Create a tight center mask to focus only on the very center
    center_mask = np.zeros((height, width), dtype=np.uint8)
    center_radius = int(min(width, height) * center_radius_pct / 100)
    cv2.circle(center_mask, (center_x, center_y), center_radius, 255, -1)
    
    # Apply mask to get only the center area
    center_area = cv2.bitwise_and(subtracted, subtracted, mask=center_mask)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(center_area)
    
    # Create a visualization image
    vis_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # Apply adaptive thresholding to highlight structures
    thresh = cv2.adaptiveThreshold(
        enhanced, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    # Apply distance transform to identify potential center points
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Find contours of potential structures
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the center point (if any contours exist)
    center_point = None
    if contours:
        # Sort contours by distance to the center of the image
        sorted_contours = sorted(contours, key=lambda c: 
            abs(cv2.moments(c)['m10']/max(cv2.moments(c)['m00'], 1e-5) - center_x) + 
            abs(cv2.moments(c)['m01']/max(cv2.moments(c)['m00'], 1e-5) - center_y))
        
        # Get the contour closest to the center
        central_contour = sorted_contours[0]
        
        # Calculate center of this contour
        M = cv2.moments(central_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            center_point = (cx, cy)
    
    # Apply multi-level thresholding to identify different layers
    # (white dot, black circle, white circle)
    _, white_inner = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
    _, black_middle = cv2.threshold(enhanced, 100, 255, cv2.THRESH_BINARY_INV)
    _, white_outer = cv2.threshold(enhanced, 50, 255, cv2.THRESH_BINARY)
    
    # Identify the structure with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    white_inner = cv2.morphologyEx(white_inner, cv2.MORPH_OPEN, kernel)
    black_middle = cv2.morphologyEx(black_middle, cv2.MORPH_OPEN, kernel)
    white_outer = cv2.morphologyEx(white_outer, cv2.MORPH_OPEN, kernel)
    
    # Create a color-coded visualization of the different layers
    vis_layers = np.zeros((height, width, 3), dtype=np.uint8)
    vis_layers[white_inner > 0] = [0, 255, 255]  # Yellow for inner white dot
    vis_layers[black_middle > 0] = [255, 0, 0]    # Blue for middle black circle
    vis_layers[white_outer > 0] = [0, 255, 0]     # Green for outer white circle
    
    # Apply center mask to the visualization
    vis_layers = cv2.bitwise_and(vis_layers, vis_layers, mask=center_mask)
    
    # Create combined visualization
    combined = cv2.addWeighted(frame, 0.6, vis_layers, 0.4, 0)
    
    # Draw the center point if found
    if center_point:
        cv2.circle(combined, center_point, 5, (0, 0, 255), -1)
        
        # Draw concentric circles to highlight the structure pattern
        cv2.circle(combined, center_point, 10, (0, 255, 255), 2)  # Inner
        cv2.circle(combined, center_point, 20, (255, 0, 0), 2)    # Middle
        cv2.circle(combined, center_point, 30, (0, 255, 0), 2)    # Outer
    
    return combined, enhanced, vis_layers, center_point

def detect_center_droplets_direct(frame, background, center_radius=150):
    """
    Direct approach to detect droplets in the center region using simpler thresholding 
    and blob detection, designed to catch the bright spots visible in the enhanced image.
    
    Args:
        frame: Current frame
        background: Background model
        center_radius: Radius of the circular region to analyze in the center
        
    Returns:
        Processed images and list of detected droplet coordinates
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Subtract background
    subtracted = cv2.absdiff(background, blurred)
    
    # Get image dimensions
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Create a circular mask for the center region
    center_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(center_mask, (center_x, center_y), center_radius, 255, -1)
    
    # Apply mask to get only the center area
    center_area = cv2.bitwise_and(subtracted, subtracted, mask=center_mask)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))
    enhanced = clahe.apply(center_area)
    
    # Create a colormap version for visualization
    enhanced_color = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
    
    # Use simple thresholding with a lower threshold to catch more potential droplets
    _, binary = cv2.threshold(enhanced, 50, 255, cv2.THRESH_BINARY)
    
    # Create a visualization to show what was detected
    binary_vis = np.zeros((height, width, 3), dtype=np.uint8)
    binary_vis[:,:,0] = binary  # Show binary in blue channel
    
    # Find connected components (blobs) in the binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create a layer visualization with blue background
    layer_vis = np.zeros((height, width, 3), dtype=np.uint8)
    layer_vis[center_mask > 0] = [50, 0, 0]  # Dark blue background
    
    # Process detected blobs
    droplets = []
    min_area = 5  # Smaller minimum area to detect small droplets
    max_area = 2000  # Larger maximum area to catch larger structures
    
    # Skip label 0 which is the background
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filter by area
        if min_area <= area <= max_area:
            # Get centroid and calculate radius
            cx, cy = int(centroids[i, 0]), int(centroids[i, 1])
            
            # Calculate radius from area
            radius = int(np.sqrt(area / np.pi))
            
            # Calculate weighted brightness (average pixel value in the blob)
            blob_mask = (labels == i).astype(np.uint8) * 255
            mean_value = cv2.mean(enhanced, blob_mask)[0]
            
            # Only include if it's bright enough
            if mean_value > 80:  # Adjust this threshold as needed
                droplets.append((cx, cy, radius, area, mean_value))
                
                # Color code by brightness
                if mean_value > 150:
                    color = (0, 255, 255)  # Yellow for very bright spots
                else:
                    color = (0, 255, 0)  # Green for moderately bright spots
                
                # Draw the blob in the layer visualization
                cv2.circle(layer_vis, (cx, cy), radius, color, 2)
                cv2.circle(layer_vis, (cx, cy), 2, (0, 0, 255), -1)  # Red center dot
                
                # Also draw the blob in the binary visualization
                cv2.circle(binary_vis, (cx, cy), radius, (0, 255, 0), 1)
    
    # Create a result frame with detected droplets
    result_frame = frame.copy()
    
    # Draw the center region
    cv2.circle(result_frame, (center_x, center_y), center_radius, (0, 0, 255), 2)
    
    # Draw detected droplets on the result frame
    for cx, cy, radius, area, brightness in droplets:
        # Different colors based on brightness
        if brightness > 150:
            color = (0, 255, 255)  # Yellow for brighter droplets
        else:
            color = (0, 255, 0)    # Green for less bright droplets
            
        cv2.circle(result_frame, (cx, cy), radius, color, 2)
        cv2.circle(result_frame, (cx, cy), 2, (0, 0, 255), -1)
        
        # Add text with droplet information
        text = f"{brightness:.0f}"
        cv2.putText(result_frame, text, (cx + 10, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result_frame, enhanced_color, layer_vis, droplets, binary_vis

def detect_droplet_centers(binary_image, original_frame, center_radius=150, min_area=10, max_area=500, 
                         min_radius=5, merge_distance=15):
    """
    Enhanced droplet center detection with size filtering and clustering
    1. Focuses on finding center blue dots of droplets
    2. Filters out circles that are too small
    3. Merges nearby circles into a single representative circle
    
    Args:
        binary_image: The binary detection image (blue channel with droplet signals)
        original_frame: Original video frame for visualization
        center_radius: Radius of center region to analyze
        min_area: Minimum area to consider as a droplet
        max_area: Maximum area to consider as a droplet
        min_radius: Minimum radius to consider (to filter tiny circles)
        merge_distance: Threshold distance to merge nearby circles
        
    Returns:
        Result frame with detected droplet centers marked and list of droplets
    """
    # Extract the blue channel where droplets should be visible
    blue_channel = binary_image[:,:,0].copy()
    
    # Get image dimensions
    height, width = binary_image.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Create a circular mask for the center region
    center_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(center_mask, (center_x, center_y), center_radius, 255, -1)
    
    # Apply mask to focus only on the center region
    masked_binary = cv2.bitwise_and(blue_channel, blue_channel, mask=center_mask)
    
    # Create visualization images
    result_frame = original_frame.copy()
    cv2.circle(result_frame, (center_x, center_y), center_radius, (0, 0, 255), 2)
    
    pattern_vis = np.zeros((height, width, 3), dtype=np.uint8)
    pattern_vis[center_mask > 0] = [30, 0, 0]  # Dark blue background
    
    # Use different thresholds to identify:
    # 1. Bright center dots (high threshold)
    # 2. Surrounding structure (medium threshold)
    # 3. Overall pattern (low threshold)
    
    # High threshold for bright center dots
    _, bright_centers = cv2.threshold(masked_binary, 150, 255, cv2.THRESH_BINARY)
    
    # Medium threshold for intermediate structures
    _, mid_structures = cv2.threshold(masked_binary, 80, 255, cv2.THRESH_BINARY)
    
    # Low threshold for overall droplet patterns
    _, overall_pattern = cv2.threshold(masked_binary, 30, 255, cv2.THRESH_BINARY)
    
    # Apply slight morphological operations to clean up noise
    kernel = np.ones((2, 2), np.uint8)
    bright_centers = cv2.morphologyEx(bright_centers, cv2.MORPH_OPEN, kernel)
    mid_structures = cv2.morphologyEx(mid_structures, cv2.MORPH_OPEN, kernel)
    overall_pattern = cv2.morphologyEx(overall_pattern, cv2.MORPH_CLOSE, kernel)
    
    # Initial droplet candidates (before merging)
    initial_droplets = []
    
    # STEP 1: Try to find droplets with clear center dots first
    # Find connected components in bright centers
    num_centers, center_labels, center_stats, center_centroids = cv2.connectedComponentsWithStats(
        bright_centers, connectivity=8)
    
    # Process each bright center
    for i in range(1, num_centers):
        area = center_stats[i, cv2.CC_STAT_AREA]
        
        # Filter by area
        if min_area <= area <= max_area:
            cx, cy = int(center_centroids[i][0]), int(center_centroids[i][1])
            
            # Check if inside our region of interest
            if center_mask[cy, cx] > 0:
                # Calculate radius from area
                radius = int(np.sqrt(area / np.pi))
                
                # Filter out circles that are too small
                if radius >= min_radius:
                    initial_droplets.append((cx, cy, radius, area, 1.0))  # 1.0 for circularity
                    
                    # Draw center dot in pattern visualization
                    cv2.circle(pattern_vis, (cx, cy), 3, (255, 255, 0), -1)  # Yellow for bright centers
    
    # If no bright centers found, try finding structures in the overall pattern
    if not initial_droplets:
        # Use distance transform to find centers of structures
        dist_transform = cv2.distanceTransform(overall_pattern, cv2.DIST_L2, 5)
        
        # Find local maxima in the distance transform
        kernel_size = 5
        max_dist = cv2.dilate(dist_transform, np.ones((kernel_size, kernel_size), np.uint8))
        peaks = (dist_transform == max_dist) & (dist_transform > 1.5)
        peaks = peaks.astype(np.uint8) * 255
        
        # Find connected components in the peaks
        num_peaks, peak_labels, peak_stats, peak_centroids = cv2.connectedComponentsWithStats(
            peaks, connectivity=8)
        
        # Process each peak
        for i in range(1, num_peaks):
            cx, cy = int(peak_centroids[i][0]), int(peak_centroids[i][1])
            
            # Check if inside our region of interest
            if center_mask[cy, cx] > 0:
                # Estimate radius from distance transform
                radius = int(dist_transform[cy, cx])
                
                # Calculate approximate area
                area = np.pi * radius * radius
                
                # Filter by area and radius
                if min_area <= area <= max_area and radius >= min_radius:
                    initial_droplets.append((cx, cy, radius, area, 0.9))  # 0.9 for estimated circularity
    
    # STEP 2: Merge nearby droplets (clustering)
    merged_droplets = []
    
    # If we have droplets to merge
    if initial_droplets:
        # Calculate distances between all pairs of droplets
        droplet_positions = np.array([(d[0], d[1]) for d in initial_droplets])
        
        # Track which droplets have been merged
        processed = [False] * len(initial_droplets)
        
        for i, (x1, y1, r1, a1, c1) in enumerate(initial_droplets):
            if processed[i]:
                continue
                
            # Mark as processed
            processed[i] = True
            
            # Start a new cluster with this droplet
            cluster = [(x1, y1, r1, a1, c1)]
            
            # Find all other droplets close to this one
            for j, (x2, y2, r2, a2, c2) in enumerate(initial_droplets):
                if i != j and not processed[j]:
                    # Calculate distance between centers
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    
                    # If close enough, add to cluster and mark as processed
                    if distance < merge_distance:
                        cluster.append((x2, y2, r2, a2, c2))
                        processed[j] = True
            
            # Calculate merged droplet properties
            if len(cluster) == 1:
                # No merging needed, just add the original
                merged_droplets.append(cluster[0])
            else:
                # Calculate weighted average position and properties
                total_area = sum(a for _, _, _, a, _ in cluster)
                
                # Weight by area (larger droplets have more influence)
                weighted_x = sum(x * a for x, _, _, a, _ in cluster) / total_area
                weighted_y = sum(y * a for _, y, _, a, _ in cluster) / total_area
                
                # Take the maximum radius to ensure we encompass all merged droplets
                max_radius = max(r for _, _, r, _, _ in cluster)
                
                # Alternative: calculate a radius based on combined area
                combined_radius = int(np.sqrt(total_area / np.pi))
                
                # Use the larger of the two radius calculations
                merged_radius = max(max_radius, combined_radius)
                
                # Add to merged droplets
                merged_droplets.append((int(weighted_x), int(weighted_y), merged_radius, total_area, 0.95))
                
                # Draw the merging in the pattern visualization
                for x, y, r, a, c in cluster:
                    cv2.circle(pattern_vis, (x, y), r, (0, 128, 255), 1)  # Orange for original circles
                
                # Draw a connecting line between merged droplets
                for k in range(len(cluster)-1):
                    x1, y1 = cluster[k][0], cluster[k][1]
                    x2, y2 = cluster[k+1][0], cluster[k+1][1]
                    cv2.line(pattern_vis, (x1, y1), (x2, y2), (0, 128, 255), 1)
    
    # STEP 3: Draw final merged droplets
    for cx, cy, radius, area, circularity in merged_droplets:
        # Draw in pattern visualization
        cv2.circle(pattern_vis, (cx, cy), radius, (0, 255, 0), 1)  # Green circle
        cv2.circle(pattern_vis, (cx, cy), 3, (0, 0, 255), -1)  # Red center
        
        # Draw on result frame
        cv2.circle(result_frame, (cx, cy), radius, (0, 255, 255), 2)  # Yellow circle
        cv2.circle(result_frame, (cx, cy), 3, (0, 0, 255), -1)  # Red center
        
        # Add text with area info
        cv2.putText(result_frame, f"{area:.0f}", (cx + 5, cy - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return result_frame, merged_droplets, pattern_vis

def detect_ring_droplets_improved(binary_image, original_frame, center_radius=150, 
                                previous_droplets=None, frame_idx=0):
    """
    Enhanced droplet detection with better handling of incomplete circles and C-shapes.
    No parameter sliders, focused on stable detection of incomplete rings.
    
    Args:
        binary_image: The binary detection image (blue channel with droplet signals)
        original_frame: Original video frame for visualization
        center_radius: Radius of center region to analyze
        previous_droplets: Droplets detected in the previous frame for tracking
        frame_idx: Current frame index for tracking
        
    Returns:
        Result frame with detected rings marked, list of droplet rings, and visualization
    """
    # Fixed parameters optimized for the specific droplet detection
    min_ring_radius = 3
    max_ring_radius = 10  # Reduced maximum radius as requested
    min_area = 10
    max_area = 300
    circularity_threshold = 0.5  # Lower to catch C-shapes better
    c_shape_circularity_min = 0.3  # Even lower threshold specific for C-shapes
    edge_threshold = 30
    merge_distance = 12
    
    # Extract the blue channel where droplets should be visible
    blue_channel = binary_image[:,:,0].copy()
    
    # Get image dimensions
    height, width = binary_image.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Create a circular mask for the center region
    center_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(center_mask, (center_x, center_y), center_radius, 255, -1)
    
    # Apply mask to focus only on the center region
    masked_binary = cv2.bitwise_and(blue_channel, blue_channel, mask=center_mask)
    
    # Create visualization images
    result_frame = original_frame.copy()
    cv2.circle(result_frame, (center_x, center_y), center_radius, (0, 0, 255), 2)
    
    pattern_vis = np.zeros((height, width, 3), dtype=np.uint8)
    pattern_vis[center_mask > 0] = [30, 0, 0]  # Dark blue background
    
    # First invert the binary image to make rings more visible
    inverted = cv2.bitwise_not(masked_binary)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    
    # Store detected ring droplets
    ring_droplets = []
    
    # SPECIFIC C-SHAPE DETECTION FOR DOWNWARD FACING C's
    # ===================================================
    
    # Apply morphological operations to enhance C-shapes
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours in the processed image
    c_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process contours to find C-shaped structures
    for contour in c_contours:
        # Calculate area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Filter by area
        if min_area <= area <= max_area:
            # Calculate circularity (C-shapes have lower circularity than full circles)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Filter for C-shaped structures
            if c_shape_circularity_min <= circularity <= 0.7:
                # Find minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Filter by radius
                if min_ring_radius <= radius <= max_ring_radius:
                    # Calculate moments to find true centroid
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = center
                    
                    # Calculate shape orientation to detect downward facing C-shapes
                    # For this, use PCA to find the principal axes
                    data_points = np.array(contour).reshape(-1, 2)
                    if len(data_points) >= 5:  # Need sufficient points for PCA
                        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_points.astype(np.float32), np.array([]))
                        
                        # Calculate angle of the principal axis
                        angle = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]) * 180 / np.pi
                        
                        # Find top point and bottom point along the contour
                        top_point = tuple(contour[contour[:, :, 1].argmin()][0])
                        bottom_point = tuple(contour[contour[:, :, 1].argmax()][0])
                        
                        # Check if opening is likely facing downward (bottom has opening)
                        # by examining the distance from the bottom point to the centroid
                        bottom_dist = np.sqrt((bottom_point[0] - cx)**2 + (bottom_point[1] - cy)**2)
                        top_dist = np.sqrt((top_point[0] - cx)**2 + (top_point[1] - cy)**2)
                        
                        # If bottom distance is greater than top distance, it suggests an opening at the bottom
                        is_downward_c = bottom_dist > 0.8 * radius and top_dist < 0.7 * radius
                        
                        # Add to ring_droplets if it's a downward C or just a general C shape
                        ring_droplets.append((cx, cy, radius, area, circularity, 'c_shape', 1.0))
                        
                        # Draw visualization with specific color for downward C-shapes
                        if is_downward_c:
                            cv2.circle(pattern_vis, (cx, cy), radius, (255, 0, 255), 1)  # Magenta for downward C
                            cv2.circle(pattern_vis, (cx, cy), 3, (255, 255, 0), -1)  # Yellow center
                            # Draw the actual contour
                            cv2.drawContours(pattern_vis, [contour], 0, (255, 0, 255), 1)
                        else:
                            cv2.circle(pattern_vis, (cx, cy), radius, (0, 255, 255), 1)  # Cyan for other C-shapes
                            cv2.circle(pattern_vis, (cx, cy), 3, (0, 255, 0), -1)  # Green center
                            # Draw the actual contour
                            cv2.drawContours(pattern_vis, [contour], 0, (0, 255, 255), 1)
    
    # APPROACH 1: Traditional Circle Detection
    # ========================================
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, edge_threshold, edge_threshold * 3)
    
    # Apply dilation to connect nearby edges
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours in the edge image
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process contours to find circular structures
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if min_area <= area <= max_area:
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate circularity (1.0 is a perfect circle)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Filter based on circularity for rings
            if circularity > circularity_threshold:
                # Find minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Filter by radius
                if min_ring_radius <= radius <= max_ring_radius:
                    # Check if this might be a duplicate of an existing ring
                    is_duplicate = False
                    for existing_x, existing_y, _, _, _, _, _ in ring_droplets:
                        dist = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                        if dist < radius:  # If centers are closer than radius
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        ring_droplets.append((center[0], center[1], radius, area, circularity, 'contour', 1.0))
                        
                        # Draw in pattern visualization
                        cv2.circle(pattern_vis, center, radius, (0, 255, 0), 1)  # Green circle
                        cv2.circle(pattern_vis, center, 3, (0, 0, 255), -1)  # Red center
    
    # STABILIZATION: Use previous detections to help with stability
    # ============================================================
    
    # If we have previous droplets, try to match them with current ones
    # This helps reduce flickering when a droplet temporarily disappears
    if previous_droplets:
        for prev_x, prev_y, prev_radius, prev_area, prev_circularity, prev_method, prev_confidence in previous_droplets:
            # Check if this previous droplet is close to any current droplet
            has_match = False
            for i, (curr_x, curr_y, _, _, _, _, _) in enumerate(ring_droplets):
                dist = np.sqrt((prev_x - curr_x)**2 + (prev_y - curr_y)**2)
                if dist < max(prev_radius, 10):  # If centers are close enough
                    has_match = True
                    break
                    
            # If no match found and it's from a recent frame, add it with reduced confidence
            if not has_match and prev_confidence > 0.5:  # Only keep relatively confident detections
                # Add with reduced confidence for smooth transition
                confidence = prev_confidence * 0.7  # Decay confidence over time
                if confidence > 0.3:  # Only keep if still reasonably confident
                    ring_droplets.append((prev_x, prev_y, prev_radius, prev_area, 
                                       prev_circularity, f"tracked_{prev_method}", confidence))
                    
                    # Draw tracked points with distinct color
                    cv2.circle(pattern_vis, (prev_x, prev_y), prev_radius, (255, 200, 0), 1)  # Orange for tracked
                    cv2.circle(pattern_vis, (prev_x, prev_y), 3, (0, 100, 255), -1)  # Dark orange center
    
    # Draw all detected ring droplets on the result frame
    for x, y, radius, area, circularity, method, confidence in ring_droplets:
        # Color based on detection method
        if method == 'contour':
            color = (0, 255, 0)  # Green for contour method
        elif method == 'c_shape':
            color = (255, 0, 255)  # Magenta for C-shape method
        elif method.startswith('tracked'):
            # Fade color based on confidence
            intensity = int(255 * confidence)
            color = (0, intensity, 255-intensity)  # Blend from yellow to red
        else:
            color = (0, 0, 255)  # Red default
        
        # Draw with transparency based on confidence
        cv2.circle(result_frame, (x, y), radius, color, 2)
        cv2.circle(result_frame, (x, y), 3, (0, 0, 255), -1)  # Red center
        
        # Add text with info
        cv2.putText(result_frame, f"{method[0]}", (x + 5, y - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add info to pattern visualization
    cv2.putText(pattern_vis, "Green: Complete Circles", (10, 20), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(pattern_vis, "Magenta: C-Shapes", (10, 40), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    cv2.putText(pattern_vis, "Orange: Tracked", (10, 60), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    
    return result_frame, ring_droplets, pattern_vis

def main():
    """
    Main function to demonstrate improved droplet detection focusing on C-shapes and incomplete rings.
    """
    # Get video path - use default if not specified
    video_path = "t09.mov"  # Default path
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"improved_ring_detection_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing video: {video_path}")
    
    # Create background model
    print("Creating background model...")
    background = create_background_model(video_path)
    
    if background is None:
        print("Failed to create background model")
        return
    
    # Save background model
    cv2.imwrite(os.path.join(output_dir, 'background_model.png'), background)
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {frame_count} frames")
    
    # Store droplet data over time
    all_droplets = []
    previous_droplets = None
    
    # Sample rate (process every Nth frame)
    sample_rate = 5
    
    # Create a video writer for output visualization
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(
        os.path.join(output_dir, 'improved_ring_detection.avi'),
        fourcc, fps/sample_rate, (width*2, height*2)
    )
    
    # Process video
    frame_idx = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("End of video reached")
            break
        
        # Process every Nth frame
        if frame_idx % sample_rate == 0:
            # Calculate time in seconds
            time_sec = frame_idx / fps
            
            # Display progress
            print(f"Processing frame {frame_idx}/{frame_count} ({100*frame_idx/frame_count:.1f}%)")
            
            # Get grayscale frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Subtract background
            subtracted = cv2.absdiff(background, blurred)
            
            # Define center region parameters
            center_x, center_y = width // 2, height // 2
            center_radius = 150  # Center region radius
            
            center_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(center_mask, (center_x, center_y), center_radius, 255, -1)
            
            # Apply mask to get only the center area
            center_area = cv2.bitwise_and(subtracted, subtracted, mask=center_mask)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))
            enhanced = clahe.apply(center_area)
            
            # Create a colormap version for visualization
            enhanced_color = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
            
            # Use simple thresholding to create binary image
            _, binary = cv2.threshold(enhanced, 50, 255, cv2.THRESH_BINARY)
            
            # Create binary visualization (blue channel)
            binary_vis = np.zeros((height, width, 3), dtype=np.uint8)
            binary_vis[:,:,0] = binary  # Show binary in blue channel
            
            # Detect droplets with improved incomplete circle detection
            result_frame, droplets, pattern_vis = detect_ring_droplets_improved(
                binary_vis, frame, center_radius=center_radius, 
                previous_droplets=previous_droplets, 
                frame_idx=frame_idx
            )
            # Update previous droplets for next frame
            previous_droplets = droplets
            
            # Store droplet data if any found
            for idx, (cx, cy, radius, area, circularity, method, confidence) in enumerate(droplets):
                all_droplets.append({
                    'frame': frame_idx,
                    'time_sec': time_sec,
                    'droplet_id': idx,
                    'x': cx,
                    'y': cy,
                    'radius': radius,
                    'area': area,
                    'circularity': circularity,
                    'method': method
                })
            
            # Create a composite view for visualization
            top_row = np.hstack([frame, enhanced_color])
            bottom_row = np.hstack([binary_vis, result_frame])
            composite = np.vstack([top_row, bottom_row])
            
            # Add labels and info to the composite view
            h, w = frame.shape[:2]
            cv2.putText(composite, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(composite, "Enhanced Center", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(composite, "Binary Detection", (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(composite, f"Droplets: {len(droplets)}", 
                      (w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(composite, f"Frame: {frame_idx}", (10, h+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Write to output video
            output_video.write(composite)
            
            # Show the composite view
            cv2.imshow('Improved Ring Detection', composite)
            
            # Show parameter visualization separately
            cv2.imshow('Pattern Visualization', pattern_vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
            # Save sample frames
            if frame_idx < 5 * sample_rate or frame_idx % (100 * sample_rate) == 0:
                sample_path = os.path.join(output_dir, f"frame_{frame_idx}.png")
                cv2.imwrite(sample_path, composite)
                
                # Also save the pattern visualization
                pattern_path = os.path.join(output_dir, f"pattern_{frame_idx}.png")
                cv2.imwrite(pattern_path, pattern_vis)
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
    
    # Save droplet data to CSV
    if all_droplets:
        df = pd.DataFrame(all_droplets)
        df.to_csv(os.path.join(output_dir, 'improved_droplet_data.csv'), index=False)
        print(f"Saved improved droplet data to improved_droplet_data.csv")
        
        # Analyze droplet data
        analyze_improved_droplets(df, output_dir)
    else:
        print("No droplets detected")

def analyze_improved_droplets(df, output_dir):
    """
    Analyze the improved droplet detection data.
    
    Args:
        df: DataFrame containing droplet data
        output_dir: Directory to save analysis plots
    """
    if len(df) == 0:
        print("No data to analyze")
        return
    
    # Count droplets per frame
    droplets_per_frame = df.groupby('frame').size()
    
    # Plot droplet count over time
    plt.figure(figsize=(12, 6))
    plt.plot(droplets_per_frame.index, droplets_per_frame.values)
    plt.title('Number of Detected Droplets Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Droplets')
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'droplets_over_time.png'))
    plt.show()
    
    # Plot methods distribution
    plt.figure(figsize=(10, 6))
    method_counts = df['method'].value_counts()
    plt.bar(method_counts.index, method_counts.values)
    plt.title('Detection Methods Used')
    plt.xlabel('Method')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'detection_methods.png'))
    plt.show()
    
    # Scatter plot of droplet positions
    plt.figure(figsize=(10, 10))
    for method, group in df.groupby('method'):
        plt.scatter(group['x'], group['y'], s=group['radius'], alpha=0.5, label=method)
    plt.legend()
    plt.title('Spatial Distribution of Droplets by Detection Method')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'droplet_positions_by_method.png'))
    plt.show()
    
    # Histogram of droplet sizes
    plt.figure(figsize=(12, 6))
    plt.hist(df['radius'], bins=30)
    plt.axvline(df['radius'].min(), color='r', linestyle='--', label=f'Min: {df["radius"].min():.1f}')
    plt.axvline(df['radius'].max(), color='g', linestyle='--', label=f'Max: {df["radius"].max():.1f}')
    plt.legend()
    plt.title('Distribution of Droplet Sizes')
    plt.xlabel('Radius (pixels)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'droplet_size_distribution.png'))
    plt.show()

if __name__ == "__main__":
    main()

