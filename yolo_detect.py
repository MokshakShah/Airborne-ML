import os
import sys
import argparse
import glob
# import time
import time

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# Define and parse user input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train3/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')
parser.add_argument('--imgsz', help='Inference image size used by YOLO (higher can improve far/small object recall).',
                    type=int, default=1280)
parser.add_argument('--focus-zoom', help='Enable second-pass ROI zoom and re-detection for small/far objects.',
                    action='store_true')
parser.add_argument('--far-object-ratio', help='Relative bbox area threshold to consider object far/small (example: "0.02").',
                    type=float, default=0.02)
parser.add_argument('--zoom-factor', help='ROI expansion factor around far object before second-pass detection (example: "2.0").',
                    type=float, default=2.0)
parser.add_argument('--focus-imgsz', help='Inference size for second-pass focus ROI. Defaults to --imgsz.',
                    type=int, default=None)
parser.add_argument('--preserve-quality', help='Avoid resizing inference frames to preserve detection quality.',
                    action='store_true')
parser.add_argument('--far-label-guard', help='Avoid hard class labels for tiny/far objects unless confidence is very high.',
                    action='store_true')
parser.add_argument('--far-classify-conf', help='Min confidence to keep fine class label for tiny/far objects.',
                    type=float, default=0.82)
parser.add_argument('--class-stability-frames', help='Frames used for majority-vote class stabilization.',
                    type=int, default=8)

args = parser.parse_args()


# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record
imgsz = args.imgsz
focus_zoom = args.focus_zoom
far_object_ratio = args.far_object_ratio
zoom_factor = args.zoom_factor
focus_imgsz = args.focus_imgsz
preserve_quality = args.preserve_quality
far_label_guard = args.far_label_guard
far_classify_conf = args.far_classify_conf
class_stability_frames = args.class_stability_frames

# Normalize threshold to float
min_thresh = float(min_thresh)

# Basic argument validation
if far_object_ratio <= 0 or far_object_ratio >= 1:
    print('ERROR: --far-object-ratio must be between 0 and 1 (exclusive).')
    sys.exit(0)

if zoom_factor < 1.0:
    print('ERROR: --zoom-factor must be >= 1.0.')
    sys.exit(0)

if imgsz < 320:
    print('ERROR: --imgsz must be >= 320.')
    sys.exit(0)

if focus_imgsz is None:
    focus_imgsz = imgsz
elif focus_imgsz < 320:
    print('ERROR: --focus-imgsz must be >= 320.')
    sys.exit(0)

if far_classify_conf <= 0 or far_classify_conf > 1:
    print('ERROR: --far-classify-conf must be in (0, 1].')
    sys.exit(0)

if class_stability_frames < 1:
    print('ERROR: --class-stability-frames must be >= 1.')
    sys.exit(0)

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':

    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
display_enabled = True
class_history = deque(maxlen=class_stability_frames)


def clamp(val, low, high):
    return max(low, min(high, val))


def compute_zoom_roi(xmin, ymin, xmax, ymax, frame_w, frame_h, zoom_scale):
    """Return expanded ROI around a detection while staying inside frame bounds."""
    box_w = xmax - xmin
    box_h = ymax - ymin
    cx = xmin + (box_w // 2)
    cy = ymin + (box_h // 2)

    roi_w = int(box_w * zoom_scale)
    roi_h = int(box_h * zoom_scale)
    # Keep enough contextual pixels around far objects for stable second-pass inference.
    roi_w = clamp(roi_w, 96, frame_w)
    roi_h = clamp(roi_h, 96, frame_h)

    x1 = int(clamp(cx - (roi_w // 2), 0, frame_w - 1))
    y1 = int(clamp(cy - (roi_h // 2), 0, frame_h - 1))
    x2 = int(clamp(x1 + roi_w, 1, frame_w))
    y2 = int(clamp(y1 + roi_h, 1, frame_h))

    # Re-adjust start if clamped end changed size
    x1 = int(clamp(x2 - roi_w, 0, frame_w - 1))
    y1 = int(clamp(y2 - roi_h, 0, frame_h - 1))
    return x1, y1, x2, y2


def stable_label(current_label):
    """Return majority-vote label over recent frames to reduce flicker/confusion."""
    class_history.append(current_label)
    if len(class_history) == 0:
        return current_label
    counts = {}
    for label in class_history:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)

# Begin inference loop
while True:

    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder': # If source is image or image folder, load the image using its filename
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video': # If source is a video, load next frame from video file
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    
    elif source_type == 'usb': # If source is a USB camera, grab frame from camera
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    elif source_type == 'picamera': # If source is a Picamera, grab frames using picamera interface
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if (frame is None):
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Keep source quality for inference when requested.
    infer_frame = frame
    if resize == True and not preserve_quality:
        infer_frame = cv2.resize(frame,(resW,resH))
    elif resize == True and preserve_quality:
        # Preserve-quality mode keeps inference on native frame.
        # Display/record frame can still be resized later.
        infer_frame = frame

    # Run first-pass inference on frame
    results = model(infer_frame, imgsz=imgsz, verbose=False)

    # Draw on this frame and only resize for display at the end if needed
    output_frame = infer_frame.copy()

    # Extract results
    detections = results[0].boxes

    # Initialize variable for basic object counting example
    object_count = 0

    frame_h, frame_w = infer_frame.shape[:2]
    frame_area = float(frame_h * frame_w)

    # Track far/small detections to run second-pass zoom inference
    focus_candidates = []

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):

        # Get bounding box coordinates
        # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > min_thresh:

            color = bbox_colors[classidx % 10]
            cv2.rectangle(output_frame, (xmin,ymin), (xmax,ymax), color, 2)

            # Estimate relative size to identify tiny/far targets.
            bbox_area = float(max(1, (xmax - xmin) * (ymax - ymin)))
            area_ratio = bbox_area / frame_area

            label_name = classname

            # Guard fine-grained class decisions on tiny/far objects.
            if far_label_guard and area_ratio <= far_object_ratio and conf < far_classify_conf:
                label_name = 'airborne-object-far'

            # Stabilize label output across nearby frames.
            label_name = stable_label(label_name)

            label = f'{label_name}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(output_frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
            cv2.putText(output_frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

            # Basic example: count the number of objects in the image
            object_count = object_count + 1

            # Mark small/far objects for second-pass zoom inference
            if focus_zoom and area_ratio <= far_object_ratio:
                focus_candidates.append((xmin, ymin, xmax, ymax))

    # Run second-pass detection in zoomed ROI for far/small candidates
    for (xmin, ymin, xmax, ymax) in focus_candidates:
        rx1, ry1, rx2, ry2 = compute_zoom_roi(xmin, ymin, xmax, ymax, frame_w, frame_h, zoom_factor)
        roi = infer_frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            continue

        # Run second-pass detection directly on native ROI pixels (no synthetic upscaling artifacts).
        zoom_results = model(roi, imgsz=focus_imgsz, verbose=False)
        zoom_detections = zoom_results[0].boxes

        # Visualize focus region
        cv2.rectangle(output_frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)
        cv2.putText(output_frame, 'Focus ROI', (rx1, max(15, ry1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        scale_x = (rx2 - rx1) / float(max(1, roi.shape[1]))
        scale_y = (ry2 - ry1) / float(max(1, roi.shape[0]))

        for j in range(len(zoom_detections)):
            z_conf = zoom_detections[j].conf.item()
            if z_conf <= min_thresh:
                continue

            z_xyxy = zoom_detections[j].xyxy.cpu().numpy().squeeze().astype(int)
            zx1, zy1, zx2, zy2 = z_xyxy
            z_classidx = int(zoom_detections[j].cls.item())
            z_classname = labels[z_classidx]
            z_color = bbox_colors[z_classidx % 10]

            # Map zoomed ROI detections back to full-frame coordinates
            gx1 = int(clamp(rx1 + zx1 * scale_x, 0, frame_w - 1))
            gy1 = int(clamp(ry1 + zy1 * scale_y, 0, frame_h - 1))
            gx2 = int(clamp(rx1 + zx2 * scale_x, 0, frame_w - 1))
            gy2 = int(clamp(ry1 + zy2 * scale_y, 0, frame_h - 1))

            cv2.rectangle(output_frame, (gx1, gy1), (gx2, gy2), z_color, 2)
            z_label = f'{z_classname}: {int(z_conf*100)}% (zoom)'
            z_size, z_base = cv2.getTextSize(z_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            z_y = max(gy1, z_size[1] + 10)
            cv2.rectangle(output_frame, (gx1, z_y-z_size[1]-10), (gx1+z_size[0], z_y+z_base-10), z_color, cv2.FILLED)
            cv2.putText(output_frame, z_label, (gx1, z_y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            object_count = object_count + 1

    # Optionally resize only for viewing/recording
    frame = output_frame
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # Calculate and draw framerate (if using video, USB, or Picamera source)
    if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw framerate
    
    # Display detection results
    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw total number of detected objects
    if display_enabled:
        try:
            cv2.imshow('YOLO detection results',frame) # Display image
        except cv2.error:
            # Continue inference/recording when OpenCV is built without GUI support.
            display_enabled = False
            print('WARNING: OpenCV GUI is not available on this system. Continuing without display window.')
    if record: recorder.write(frame)

    # If inferencing on individual images, wait for user keypress before moving to next image. Otherwise, wait 5ms before moving to next frame.
    key = -1
    if display_enabled:
        if source_type == 'image' or source_type == 'folder':
            key = cv2.waitKey()
        elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
            key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)
    
    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)


# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()
