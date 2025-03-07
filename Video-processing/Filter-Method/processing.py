import sys
import subprocess

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import os
from multiprocessing import Pool, cpu_count

print("v1.3")

def detect_droplets(frame, min_radius=3, max_radius=30, param1=50, param2=30, center_focus=True, center_focus_pct=85):
    """
    Detect droplets in a frame using circle detection.
    
    Args:
        frame: Input image frame
        min_radius: Minimum radius of droplets to detect
        max_radius: Maximum radius of droplets to detect
        param1: First parameter of Hough Circle method (edge detection sensitivity)
        param2: Second parameter of Hough Circle method (circle detection threshold)
        center_focus: Whether to focus on center region of the dish
        center_focus_pct: Percentage of the dish radius to include (1-100)
        
    Returns:
        List of detected droplets as (x, y, radius) tuples
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Create a mask for the center region if requested
    if center_focus:
        # Calculate the center of the image
        center_x, center_y = width // 2, height // 2
        
        # Create a mask for the central region
        # Use the center_focus_pct to determine how much of the dish to include
        dish_radius = min(width, height) // 2
        inner_radius = int(dish_radius * center_focus_pct / 100)
        
        # Create black mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Draw a filled white circle for the center region
        cv2.circle(mask, (center_x, center_y), inner_radius, 255, -1)
        
        # Apply the mask
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        enhanced,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=10,  # Decreased to allow closer droplets
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    # If no circles are detected, return empty list
    if circles is None:
        return []
    
    # Convert to integer coordinates
    circles = np.round(circles[0, :]).astype(int)
    
    # Return as list of (x, y, radius) tuples
    return [(x, y, r) for x, y, r in circles]

def detect_droplets_watershed(frame):
    """Detect droplets using watershed algorithm"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Noise removal with morphological operations
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    
    # Apply watershed
    markers = cv2.watershed(frame, markers)
    
    # Extract droplet information
    droplets = []
    for label in range(2, markers.max() + 1):
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[markers == label] = 255
        
        # Find contour of the droplet
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            droplets.append((int(x), int(y), int(radius)))
    
    return droplets

def process_frame(args):
    """Process a single frame (for parallel processing)"""
    frame_idx, frame, fps = args
    time_sec = frame_idx / fps
    
    # Detect droplets
    droplets = detect_droplets(frame)
    
    # Prepare data
    frame_data = []
    for idx, (x, y, radius) in enumerate(droplets):
        frame_data.append({
            'frame': frame_idx,
            'time_sec': time_sec,
            'droplet_id': idx,
            'x': x,
            'y': y,
            'radius': radius,
            'area': np.pi * radius * radius
        })
    
    return frame_data

def process_video_parallel(video_path, output_csv, sample_rate=1):
    """Process video with parallel processing"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Collect frames to process
    frames_to_process = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % sample_rate == 0:
            frames_to_process.append((frame_idx, frame, fps))
        
        frame_idx += 1
    
    cap.release()
    
    # Process frames in parallel
    with Pool(processes=max(1, cpu_count()-1)) as pool:
        results = pool.map(process_frame, frames_to_process)
    
    # Flatten results
    all_data = [item for sublist in results for item in sublist]
    
    # Save to CSV
    pd.DataFrame(all_data).to_csv(output_csv, index=False)

def process_video(video_path, output_csv, display=True, sample_rate=1):
    """
    Process video to detect droplets and save data to CSV.
    
    Args:
        video_path: Path to the input video file
        output_csv: Path to save the output CSV file
        display: Whether to display processing results
        sample_rate: Process every Nth frame (for speed)
    """
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
    
    # Process frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process only every sample_rate frame
        if frame_idx % sample_rate == 0:
            time_sec = frame_idx / fps
            
            # Detect droplets
            droplets = detect_droplets(frame)
            
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
                
                # Resize for display if necessary
                scale = 1.0
                if width > 1200:
                    scale = 1200 / width
                    vis_frame = cv2.resize(vis_frame, (0, 0), fx=scale, fy=scale)
                
                # Show frame
                cv2.imshow('Droplet Detection', vis_frame)
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
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'droplet_size_distribution.png'))
    plt.show()
    
    # 2D distribution of droplet positions
    plt.figure(figsize=(10, 10))
    plt.scatter(df['x'], df['y'], s=df['radius'], alpha=0.5)
    plt.title('Spatial Distribution of Droplets')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'droplet_spatial_distribution.png'))
    plt.show()

def create_parameter_window():
    def on_change(val):
        pass  # Parameters will be read directly from trackbars

    cv2.namedWindow('Parameters')
    cv2.createTrackbar('Min Radius', 'Parameters', min_radius, 100, on_change)
    cv2.createTrackbar('Max Radius', 'Parameters', max_radius, 200, on_change)
    cv2.createTrackbar('Param1', 'Parameters', param1, 300, on_change)
    cv2.createTrackbar('Param2', 'Parameters', param2, 100, on_change)

def get_parameters():
    min_radius = cv2.getTrackbarPos('Min Radius', 'Parameters')
    max_radius = cv2.getTrackbarPos('Max Radius', 'Parameters')
    param1 = cv2.getTrackbarPos('Param1', 'Parameters')
    param2 = cv2.getTrackbarPos('Param2', 'Parameters')
    return min_radius, max_radius, param1, param2

def track_droplets(video_path, output_csv):
    """Track individual droplets across frames"""
    cap = cv2.VideoCapture(video_path)
    
    # Initialize tracker
    tracker = cv2.legacy.TrackerCSRT_create()
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        return
    
    # Detect droplets in first frame
    droplets = detect_droplets(frame)
    
    # Initialize trackers for each droplet
    trackers = []
    for x, y, r in droplets:
        tracker = cv2.legacy.TrackerCSRT_create()
        bbox = (x-r, y-r, 2*r, 2*r)  # Convert to x,y,w,h format
        success = tracker.init(frame, bbox)
        if success:
            trackers.append({
                'tracker': tracker,
                'id': len(trackers),
                'trajectory': [(0, x, y, r)]  # (frame, x, y, radius)
            })
    
    # Process remaining frames
    frame_idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update trackers
        for droplet in trackers:
            success, bbox = droplet['tracker'].update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                radius = (w + h) // 4
                droplet['trajectory'].append((frame_idx, x + w//2, y + h//2, radius))
        
        frame_idx += 1
    
    # Save trajectory data
    all_data = []
    for droplet in trackers:
        for frame, x, y, r in droplet['trajectory']:
            all_data.append({
                'frame': frame,
                'droplet_id': droplet['id'],
                'x': x,
                'y': y,
                'radius': r,
                'area': np.pi * r * r
            })
    
    # Save to CSV
    pd.DataFrame(all_data).to_csv(output_csv, index=False)

def tune_detection_parameters(video_path):
    """
    Interactive tool for tuning detection parameters.
    
    Args:
        video_path: Path to the input video file
    
    Returns:
        Tuple of (min_radius, max_radius, param1, param2, center_focus_pct)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return (5, 30, 50, 30, 85)
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create main window
    cv2.namedWindow('Droplet Detection Tuner', cv2.WINDOW_NORMAL)
    
    # Create parameter window
    cv2.namedWindow('Parameters')
    
    # Default parameter values with minimums to avoid invalid values
    min_radius = 3
    max_radius = 20
    param1 = 50
    param2 = 30
    center_focus_pct = 85  # Default: exclude outer 15% of dish radius
    frame_idx = 0
    
    # Create trackbars with appropriate ranges
    cv2.createTrackbar('Min Radius', 'Parameters', min_radius, 20, lambda x: None)
    cv2.createTrackbar('Max Radius', 'Parameters', max_radius, 50, lambda x: None)
    cv2.createTrackbar('Param1', 'Parameters', param1, 300, lambda x: None)
    cv2.createTrackbar('Param2', 'Parameters', param2, 100, lambda x: None)
    cv2.createTrackbar('Center Focus %', 'Parameters', center_focus_pct, 100, lambda x: None)
    cv2.createTrackbar('Frame', 'Parameters', frame_idx, total_frames-1, lambda x: None)
    
    # Sample frames for quick tuning
    sample_frames = []
    sample_indices = [int(total_frames * i / 10) for i in range(10)]  # 10 frames evenly distributed
    
    # Load sample frames
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            sample_frames.append((idx, frame))
    
    # Additional status info
    saved_params = []
    current_sample = 0
    
    # Instructions text
    instructions = [
        "INSTRUCTIONS:",
        "- Adjust parameters using sliders",
        "- 'Center Focus %' controls how much of dish edge to exclude",
        "- Press 'n' for next sample frame",
        "- Press 'p' for previous sample frame",
        "- Press 's' to save current parameters",
        "- Press 'a' to apply saved parameters (average)",
        "- Press 'q' to quit and process video"
    ]
    
    while True:
        # Get current frame index from trackbar
        new_frame_idx = cv2.getTrackbarPos('Frame', 'Parameters')
        
        # If manually selecting a frame that's not in samples
        if new_frame_idx != frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_idx = new_frame_idx
            else:
                continue
        else:
            # Use the current sample frame
            frame_idx, frame = sample_frames[current_sample]
            cv2.setTrackbarPos('Frame', 'Parameters', frame_idx)
        
        # Get current parameters
        min_radius = max(1, cv2.getTrackbarPos('Min Radius', 'Parameters'))  # Ensure minimum 1
        max_radius = max(min_radius+1, cv2.getTrackbarPos('Max Radius', 'Parameters'))  # Ensure greater than min
        param1 = max(10, cv2.getTrackbarPos('Param1', 'Parameters'))  # Ensure minimum 10
        param2 = max(1, cv2.getTrackbarPos('Param2', 'Parameters'))  # Ensure minimum 1
        center_focus_pct = max(1, cv2.getTrackbarPos('Center Focus %', 'Parameters'))  # Ensure minimum 1
        
        # Make a copy for visualization
        vis_frame = frame.copy()
        
        # Detect droplets with current parameters
        droplets = detect_droplets(frame, min_radius, max_radius, param1, param2, center_focus=True, center_focus_pct=center_focus_pct)
        
        # Draw the center focus region
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        dish_radius = min(width, height) // 2
        inner_radius = int(dish_radius * center_focus_pct / 100)
        cv2.circle(vis_frame, (center_x, center_y), inner_radius, (0, 255, 255), 2)
        
        # Draw detected droplets
        for x, y, radius in droplets:
            cv2.circle(vis_frame, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(vis_frame, (x, y), 2, (0, 0, 255), 3)
        
        # Display parameter info
        cv2.putText(
            vis_frame, 
            f"Frame: {frame_idx}/{total_frames} | Droplets: {len(droplets)}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        cv2.putText(
            vis_frame, 
            f"Min R: {min_radius}, Max R: {max_radius}, P1: {param1}, P2: {param2}, Focus: {center_focus_pct}%", 
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 0, 255), 
            2
        )
        
        # Display saved parameters
        if saved_params:
            saved_text = f"Saved sets: {len(saved_params)}"
            cv2.putText(
                vis_frame, 
                saved_text, 
                (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 0, 0), 
                2
            )
        
        # Display instructions
        for i, text in enumerate(instructions):
            cv2.putText(
                vis_frame, 
                text, 
                (10, vis_frame.shape[0] - 30 * (len(instructions) - i)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                1
            )
        
        # Resize for display if necessary
        height, width = vis_frame.shape[:2]
        scale = 1.0
        if width > 1200:
            scale = 1200 / width
            display_frame = cv2.resize(vis_frame, (0, 0), fx=scale, fy=scale)
        else:
            display_frame = vis_frame
        
        # Show frame
        cv2.imshow('Droplet Detection Tuner', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Quit
            break
        elif key == ord('n'):
            # Next sample
            current_sample = (current_sample + 1) % len(sample_frames)
        elif key == ord('p'):
            # Previous sample
            current_sample = (current_sample - 1) % len(sample_frames)
        elif key == ord('s'):
            # Save current parameters
            saved_params.append((min_radius, max_radius, param1, param2, center_focus_pct))
            print(f"Saved parameters #{len(saved_params)}: min_radius={min_radius}, max_radius={max_radius}, param1={param1}, param2={param2}, center_focus={center_focus_pct}%")
        elif key == ord('a'):
            # Apply (average) saved parameters
            if saved_params:
                avg_min_radius = int(sum(p[0] for p in saved_params) / len(saved_params))
                avg_max_radius = int(sum(p[1] for p in saved_params) / len(saved_params))
                avg_param1 = int(sum(p[2] for p in saved_params) / len(saved_params))
                avg_param2 = int(sum(p[3] for p in saved_params) / len(saved_params))
                avg_center_focus = int(sum(p[4] for p in saved_params) / len(saved_params))
                
                cv2.setTrackbarPos('Min Radius', 'Parameters', avg_min_radius)
                cv2.setTrackbarPos('Max Radius', 'Parameters', avg_max_radius)
                cv2.setTrackbarPos('Param1', 'Parameters', avg_param1)
                cv2.setTrackbarPos('Param2', 'Parameters', avg_param2)
                cv2.setTrackbarPos('Center Focus %', 'Parameters', avg_center_focus)
                
                print(f"Applied average parameters: min_radius={avg_min_radius}, max_radius={avg_max_radius}, param1={avg_param1}, param2={avg_param2}, center_focus={avg_center_focus}%")
                
                # Use these as final parameters
                min_radius, max_radius, param1, param2, center_focus_pct = avg_min_radius, avg_max_radius, avg_param1, avg_param2, avg_center_focus
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Return parameters (either the last used or averaged if saved)
    if saved_params and len(saved_params) > 0:
        # Automatically use average of saved parameters if any exist
        avg_min_radius = int(sum(p[0] for p in saved_params) / len(saved_params))
        avg_max_radius = int(sum(p[1] for p in saved_params) / len(saved_params))
        avg_param1 = int(sum(p[2] for p in saved_params) / len(saved_params))
        avg_param2 = int(sum(p[3] for p in saved_params) / len(saved_params))
        avg_center_focus = int(sum(p[4] for p in saved_params) / len(saved_params))
        return (avg_min_radius, avg_max_radius, avg_param1, avg_param2, avg_center_focus)
    
    return (min_radius, max_radius, param1, param2, center_focus_pct)

def main():
    # Get current timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use default file path without asking
    video_path = "t09.mov"
    output_csv = f"droplet_data_{timestamp}.csv"
    output_dir = f"droplet_analysis_{timestamp}"
    
    print(f"Processing video: {video_path}")
    
    # Automatically go to tuning mode
    min_radius, max_radius, param1, param2, center_focus_pct = tune_detection_parameters(video_path)
    print(f"Using parameters: min_radius={min_radius}, max_radius={max_radius}, param1={param1}, param2={param2}, center_focus={center_focus_pct}%")
    
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
    
    # Sample rate (process every Nth frame)
    sample_rate = 5
    
    # Process frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process only every sample_rate frame
        if frame_idx % sample_rate == 0:
            time_sec = frame_idx / fps
            
            # Detect droplets using center focus
            droplets = detect_droplets(
                frame, 
                min_radius=min_radius, 
                max_radius=max_radius, 
                param1=param1, 
                param2=param2,
                center_focus=True,
                center_focus_pct=center_focus_pct
            )
            
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
            vis_frame = frame.copy()
            
            # Draw the center focus region
            center_x, center_y = width // 2, height // 2
            dish_radius = min(width, height) // 2
            inner_radius = int(dish_radius * center_focus_pct / 100)
            cv2.circle(vis_frame, (center_x, center_y), inner_radius, (0, 255, 255), 2)
            
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
            
            # Resize for display if necessary
            scale = 1.0
            if width > 1200:
                scale = 1200 / width
                vis_frame = cv2.resize(vis_frame, (0, 0), fx=scale, fy=scale)
            
            # Show frame
            cv2.imshow('Droplet Detection', vis_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        frame_idx += 1
        
        # Show progress
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames ({frame_idx/frame_count*100:.1f}%)")
    
    # Release video and close windows
    cap.release()
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
        
        # Analyze results
        print(f"\nAnalyzing results from {output_csv}")
        analyze_results(output_csv, output_dir)
    else:
        print("No droplets detected")

if __name__ == "__main__":
    main()
