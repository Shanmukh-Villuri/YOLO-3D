#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np
import torch
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Set MPS fallback for operations not supported on Apple Silicon
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import our modules
from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView
from load_camera_params import load_camera_params, apply_camera_params_to_estimator


start = time.perf_counter()


def main():
    """Main function."""
    # Configuration variables (modify these as needed)
    # ===============================================
    
    # Input/Output
    source = "/home/gazebo/src/YOLO-3D/input/kiit_ 30FPS.mp4"  # Path to input video file or webcam index (0 for default camera)
    output_path = "/home/gazebo/src/YOLO-3D/output/kitt_30FPS_1_llo.mp4"  # Path to output video file
    if os.environ.get('YOLO3D_OUTPUT'):
        output_path = os.environ['YOLO3D_OUTPUT']
    
    # Model settings
    yolo_model_size = "small"  # YOLOv11 model size: "nano", "small", "medium", "large", "extra"
    depth_model_size = "small"  # Depth Anything v2 model size: "small", "base", "large"
    yolo_model_size = os.environ.get('YOLO3D_YOLO_SIZE', yolo_model_size)
    depth_model_size = os.environ.get('YOLO3D_DEPTH_SIZE', depth_model_size)
    depth_every = int(os.environ.get('YOLO3D_DEPTH_EVERY', '3'))  # run depth every Nth frame, reuse map in between
    depth_metric = True  # True = real distance in meters
    depth_scene = "outdoor"  # "indoor" or "outdoor" — match your scene
    
    # Device settings
    device = 'cuda'
    use_half = os.environ.get('YOLO3D_HALF', '0') == '1'  # FP16: slower here (tracker overhead), keep off
    
    # Detection settings
    conf_threshold = 0.25  # Confidence threshold for object detection
    iou_threshold = 0.45  # IoU threshold for NMS
    classes = None  # Filter by class, e.g., [0, 1, 2] for specific classes, None for all classes
    
    # Feature toggles
    enable_tracking = True  # Enable object tracking
    if os.environ.get('YOLO3D_TRACK') == '0':
        enable_tracking = False
    enable_bev = True  # Enable Bird's Eye View visualization
    enable_pseudo_3d = True  # Enable pseudo-3D visualization
    
    # Camera parameters - simplified approach
    camera_params_file = None  # Path to camera parameters file (None to use default parameters)
    # ===============================================
    
    print(f"Using device: {device}")
    
    # Initialize models
    print("Initializing models...")
    try:
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device=device,
            half=use_half
        )
    except Exception as e:
        print(f"Error initializing object detector: {e}")
        print("Falling back to CPU for object detection")
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device='cpu',
            half=False
        )
    
    try:
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device=device,
            metric=depth_metric,
            scene=depth_scene,
            half=use_half
        )
    except Exception as e:
        print(f"Error initializing depth estimator: {e}")
        print("Falling back to CPU for depth estimation")
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device='cpu',
            metric=depth_metric,
            scene=depth_scene,
            half=False
        )    
    # Initialize 3D bounding box estimator with default parameters
    # Simplified approach - focus on 2D detection with depth information
    bbox3d_estimator = BBox3DEstimator()
    
    # Initialize Bird's Eye View if enabled
    if enable_bev:
        # Use a scale that works well for the 1-5 meter range
        bev = BirdEyeView(scale=60, size=(300, 300))  # Increased scale to spread objects out
    
    # Open video source
    try:
        if isinstance(source, str) and source.isdigit():
            source = int(source)  # Convert string number to integer for webcam
    except ValueError:
        pass  # Keep as string (for video file)
    
    print(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:  # Sometimes happens with webcams
        fps = 30
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"

    # Timing accumulators for speed benchmarking (whole-setup frame rate)
    max_frames = int(os.environ.get('YOLO3D_MAX_FRAMES', '0'))  # 0 = no limit
    headless = os.environ.get('YOLO3D_HEADLESS', '') == '1'  # skip cv2.imshow
    stage_times = {'detect': 0.0, 'depth': 0.0, 'post': 0.0, 'io': 0.0, 'frame_wall': 0.0, 'parallel': 0.0}
    n_timed = 0
    total_start = time.perf_counter()

    # Thread pool so detection and depth run in parallel (frame cost ~= max, not sum)
    executor = ThreadPoolExecutor(max_workers=2)
    cached_depth_map = None
    cached_depth_colored = None

    def _detect_job(img):
        t = time.perf_counter()
        try:
            ann, dets = detector.detect(img, track=enable_tracking)
        except Exception as e:
            print(f"Error during object detection: {e}")
            ann, dets = img, []
            cv2.putText(ann, "Detection Error", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return ann, dets, time.perf_counter() - t

    def _depth_job(img):
        t = time.perf_counter()
        try:
            m = depth_estimator.estimate_depth(img)
            c = depth_estimator.colorize_depth(m)
        except Exception as e:
            print(f"Error during depth estimation: {e}")
            m = np.zeros((height, width), dtype=np.float32)
            c = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(c, "Depth Error", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return m, c, time.perf_counter() - t

# --- add these 3 lines --- 
    import csv 
    csv_file = open('distances_log.csv', 'w', newline='') 
    csv_writer = csv.writer(csv_file) 
    csv_writer.writerow(['frame_idx', 'object_id', 'class_name', 'x1', 'y1', 'x2', 'y2', 'predicted_distance_m']) 

    
    print("Starting processing...")
    
    # Main loop
    while True:
        if max_frames and frame_count >= max_frames:
            print(f"Reached YOLO3D_MAX_FRAMES={max_frames}, stopping.")
            break
        # Check for key press at the beginning of each loop
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
            print("Exiting program...")
            break
            
        try:
            t_frame = time.perf_counter()
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Make copies for different visualizations
            original_frame = frame.copy()
            detection_frame = frame.copy()
            depth_frame = frame.copy()
            result_frame = frame.copy()
            
            # Steps 1+2 in parallel: detection runs every frame,
            # depth runs every Nth frame (cached map reused in between)
            t0 = time.perf_counter()
            fut_det = executor.submit(_detect_job, detection_frame)
            depth_due = (frame_count % depth_every == 0) or (cached_depth_map is None)
            fut_depth = executor.submit(_depth_job, original_frame) if depth_due else None

            detection_frame, detections, t_detect = fut_det.result()
            if fut_depth is not None:
                depth_map, depth_colored, t_depth = fut_depth.result()
                cached_depth_map, cached_depth_colored = depth_map, depth_colored
            else:
                depth_map, depth_colored, t_depth = cached_depth_map, cached_depth_colored, 0.0
            stage_times['parallel'] += time.perf_counter() - t0

            t0 = time.perf_counter()
            
            # Step 3: 3D Bounding Box Estimation
            boxes_3d = []
            active_ids = []
            
            for detection in detections:
                try:
                    bbox, score, class_id, obj_id = detection
                    
                    # Get class name
                    class_name = detector.get_class_names()[class_id]
                    
                    # Get depth in the region of the bounding box
                    # Try different methods for depth estimation
                    if class_name.lower() in ['person', 'cat', 'dog']:
                        # For people and animals, use the center point depth
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        depth_value = depth_estimator.get_depth_at_point(depth_map, center_x, center_y)
                        depth_method = 'center'
                    else:
                        # For other objects, use the median depth in the region
                        depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                        depth_method = 'median'
                    
                    # Create a simplified 3D box representation
                    box_3d = {
                        'bbox_2d': bbox,
                        'depth_value': depth_value,
                        'depth_method': depth_method,
                        'class_name': class_name,
                        'object_id': obj_id,
                        'score': score
                    }
                    
                    boxes_3d.append(box_3d) 

# --- add this --- 
                    csv_writer.writerow([ 
                        frame_count, 
                        box_3d.get('object_id', -1), 
                        box_3d.get('class_name', ''), 
                        *box_3d['bbox_2d'], 
                        box_3d.get('depth_value', -1) 
                        ]) 

                    
                    # Keep track of active IDs for tracker cleanup
                    if obj_id is not None:
                        active_ids.append(obj_id)
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
            
            # Clean up trackers for objects that are no longer detected
            bbox3d_estimator.cleanup_trackers(active_ids)
            
            # Step 4: Visualization
            # Draw boxes on the result frame
            for box_3d in boxes_3d:
                try:
                    # Determine color based on class
                    class_name = box_3d['class_name'].lower()
                    if 'car' in class_name or 'vehicle' in class_name:
                        color = (0, 0, 255)  # Red
                    elif 'person' in class_name:
                        color = (0, 255, 0)  # Green
                    elif 'bicycle' in class_name or 'motorcycle' in class_name:
                        color = (255, 0, 0)  # Blue
                    elif 'potted plant' in class_name or 'plant' in class_name:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (255, 255, 255)  # White
                    
                    # Draw box with depth information
                    result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
                except Exception as e:
                    print(f"Error drawing box: {e}")
                    continue
            
            # Draw Bird's Eye View if enabled
            if enable_bev:
                try:
                    # Reset BEV and draw objects
                    bev.reset()
                    for box_3d in boxes_3d:
                        bev.draw_box(box_3d)
                    bev_image = bev.get_image()
                    
                    # Resize BEV image to fit in the corner of the result frame
                    bev_height = height // 4  # Reduced from height/3 to height/4 for better fit
                    bev_width = bev_height
                    
                    # Ensure dimensions are valid
                    if bev_height > 0 and bev_width > 0:
                        # Resize BEV image
                        bev_resized = cv2.resize(bev_image, (bev_width, bev_height))
                        
                        # Create a region of interest in the result frame
                        roi = result_frame[height - bev_height:height, 0:bev_width]
                        
                        # Simple overlay - just copy the BEV image to the ROI
                        result_frame[height - bev_height:height, 0:bev_width] = bev_resized
                        
                        # Add a border around the BEV visualization
                        cv2.rectangle(result_frame, 
                                     (0, height - bev_height), 
                                     (bev_width, height), 
                                     (255, 255, 255), 1)
                        
                        # Add a title to the BEV visualization
                        cv2.putText(result_frame, "Bird's Eye View", 
                                   (10, height - bev_height + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                except Exception as e:
                    print(f"Error drawing BEV: {e}")
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 10 == 0:  # Update FPS every 10 frames
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps_value = frame_count / elapsed_time
                fps_display = f"FPS: {fps_value:.1f}"
            
            # Add FPS and device info to the result frame
            cv2.putText(result_frame, f"{fps_display} | Device: {device}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            
            # Add depth map to the corner of the result frame
            try:
                depth_height = height // 4
                depth_width = depth_height * width // height
                depth_resized = cv2.resize(depth_colored, (depth_width, depth_height))
                result_frame[0:depth_height, 0:depth_width] = depth_resized
            except Exception as e:
                print(f"Error adding depth map to result: {e}")
            
            t_post = time.perf_counter() - t0

            t0 = time.perf_counter()
            # Write frame to output video
            out.write(result_frame)
            
            # Display frames
            if not headless:
                cv2.imshow("3D Object Detection", result_frame)
                cv2.imshow("Depth Map", depth_colored)
                cv2.imshow("Object Detection", detection_frame)
            t_io = time.perf_counter() - t0

            stage_times['detect'] += t_detect
            stage_times['depth'] += t_depth
            stage_times['post'] += t_post
            stage_times['io'] += t_io
            stage_times['frame_wall'] += time.perf_counter() - t_frame
            n_timed += 1
            
            # Check for key press again at the end of the loop
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
                print("Exiting program...")
                break
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Also check for key press during exception handling
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
                print("Exiting program...")
                break
            continue
    
    # Clean up
    print("Cleaning up resources...")
    total_time = time.perf_counter() - total_start
    if n_timed > 0:
        wall_ms = stage_times['frame_wall'] / n_timed * 1000
        avg = {k: stage_times[k] / n_timed * 1000 for k in ('detect', 'depth', 'post', 'io')}
        print(f"[SPEED] frames={n_timed} total={total_time:.1f}s "
              f"wall_fps={1000 / wall_ms:.2f} avg_frame={wall_ms:.1f}ms "
              f"(compute detect={avg['detect']:.1f}ms depth={avg['depth']:.1f}ms "
              f"post={avg['post']:.1f}ms io={avg['io']:.1f}ms)")
    executor.shutdown()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"Processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        # Clean up OpenCV windows
        cv2.destroyAllWindows() 

end = time.perf_counter()

print(f"Runtime: {end - start:.2f} seconds")
