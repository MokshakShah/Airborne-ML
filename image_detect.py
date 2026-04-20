import os
import sys
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="YOLO Image Detection - Detect airborne threats in images"
    )
    parser.add_argument('--image', help='Path to image file (e.g., "test.jpg", "photo.png")',
                       required=True)
    parser.add_argument('--model', help='Path to YOLO model file (e.g., "my_model.pt")',
                       required=True)
    parser.add_argument('--thresh', help='Minimum confidence threshold (0-1, default: 0.5)',
                       type=float, default=0.5)
    parser.add_argument('--imgsz', help='Inference image size (e.g., 640, 1280)',
                       type=int, default=1280)
    parser.add_argument('--resolution', help='Display resolution (e.g., "1280x720")',
                       default=None)
    parser.add_argument('--save-output', help='Save annotated image to file',
                       action='store_true')
    parser.add_argument('--output-name', help='Output filename (default: image_detection.jpg)',
                       default='image_detection.jpg')
    parser.add_argument('--focus-zoom', help='Enable multi-scale detection for small objects',
                       action='store_true')
    parser.add_argument('--far-object-ratio', help='Threshold for far/small objects (default: 0.02)',
                       type=float, default=0.02)
    parser.add_argument('--zoom-factor', help='ROI zoom expansion factor (default: 2.0)',
                       type=float, default=2.0)

    return parser.parse_args()


def clamp(val, low, high):
    """Clamp value between low and high"""
    return max(low, min(high, val))


def compute_zoom_roi(xmin, ymin, xmax, ymax, frame_w, frame_h, zoom_scale):
    """Compute expanded ROI around a detection for zoom detection"""
    box_w = xmax - xmin
    box_h = ymax - ymin
    cx = xmin + (box_w // 2)
    cy = ymin + (box_h // 2)

    roi_w = int(box_w * zoom_scale)
    roi_h = int(box_h * zoom_scale)
    roi_w = clamp(roi_w, 96, frame_w)
    roi_h = clamp(roi_h, 96, frame_h)

    x1 = int(clamp(cx - (roi_w // 2), 0, frame_w - 1))
    y1 = int(clamp(cy - (roi_h // 2), 0, frame_h - 1))
    x2 = int(clamp(x1 + roi_w, 1, frame_w))
    y2 = int(clamp(y1 + roi_h, 1, frame_h))

    x1 = int(clamp(x2 - roi_w, 0, frame_w - 1))
    y1 = int(clamp(y2 - roi_h, 0, frame_h - 1))
    return x1, y1, x2, y2


def main():
    """Main function for image detection"""
    args = parse_args()

    # Validate image path
    if not os.path.exists(args.image):
        print(f'ERROR: Image path does not exist: {args.image}')
        sys.exit(1)

    # Validate model path
    if not os.path.exists(args.model):
        print(f'ERROR: Model path does not exist: {args.model}')
        sys.exit(1)

    # Validate threshold
    if args.thresh < 0 or args.thresh > 1:
        print('ERROR: Threshold must be between 0 and 1')
        sys.exit(1)

    # Load model
    print(f"Loading model from: {args.model}")
    model = YOLO(args.model, task='detect')
    labels = model.names

    # Load image
    print(f"Loading image from: {args.image}")
    frame = cv2.imread(args.image)
    
    if frame is None:
        print(f'ERROR: Failed to load image: {args.image}')
        sys.exit(1)

    frame_h, frame_w = frame.shape[:2]
    print(f"Image size: {frame_w}×{frame_h}")

    # Parse display resolution if specified
    resize = False
    resW, resH = None, None
    if args.resolution:
        resize = True
        resW, resH = int(args.resolution.split('x')[0]), int(args.resolution.split('x')[1])
        print(f"Display resolution: {resW}×{resH}")

    print(f"\nRunning inference with confidence threshold: {args.thresh * 100:.0f}%")
    print(f"Inference image size: {args.imgsz}×{args.imgsz}")

    # Measure inference time
    t_start = time.perf_counter()

    # Run inference
    results = model(frame, imgsz=args.imgsz, verbose=False)

    # Copy frame for output
    output_frame = frame.copy()

    # Extract results
    detections = results[0].boxes

    # Object counting
    object_count = 0
    frame_area = float(frame_h * frame_w)

    # Track far/small detections for zoom
    focus_candidates = []

    print(f"\nDetections found:")

    # Process each detection
    for i in range(len(detections)):

        # Get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        # Get class and confidence
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        # Draw if confidence threshold met
        if conf > args.thresh:
            # Green color for boxes
            color = (0, 255, 0)
            cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), color, 2)

            # Create label with confidence percentage
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_ymin = max(ymin, labelSize[1] + 10)

            # Draw label background
            cv2.rectangle(output_frame, (xmin, label_ymin-labelSize[1]-10), 
                         (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)

            # Draw label text
            cv2.putText(output_frame, label, (xmin, label_ymin-7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            object_count += 1

            # Estimate size ratio for far objects
            bbox_area = float(max(1, (xmax - xmin) * (ymax - ymin)))
            area_ratio = bbox_area / frame_area

            print(f"  [{object_count}] {classname}: {int(conf*100)}% | "
                  f"Box: ({xmin},{ymin})-({xmax},{ymax}) | "
                  f"Size ratio: {area_ratio:.4f}")

            # Mark for zoom detection if small
            if args.focus_zoom and area_ratio <= args.far_object_ratio:
                focus_candidates.append((xmin, ymin, xmax, ymax))
                print(f"      → Marked for multi-scale re-detection (small object)")

    # Run zoom detection on far/small objects
    if args.focus_zoom and focus_candidates:
        print(f"\nRunning multi-scale detection on {len(focus_candidates)} small object(s)...")

        for (xmin, ymin, xmax, ymax) in focus_candidates:
            rx1, ry1, rx2, ry2 = compute_zoom_roi(xmin, ymin, xmax, ymax, frame_w, frame_h, args.zoom_factor)
            roi = frame[ry1:ry2, rx1:rx2]

            if roi.size == 0:
                continue

            # Re-detect on zoomed ROI
            zoom_results = model(roi, imgsz=args.imgsz, verbose=False)
            zoom_detections = zoom_results[0].boxes

            # Draw focus ROI boundary
            cv2.rectangle(output_frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)
            cv2.putText(output_frame, 'Focus ROI', (rx1, max(15, ry1 - 5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            scale_x = (rx2 - rx1) / float(max(1, roi.shape[1]))
            scale_y = (ry2 - ry1) / float(max(1, roi.shape[0]))

            # Process zoom detections
            for j in range(len(zoom_detections)):
                z_conf = zoom_detections[j].conf.item()
                if z_conf <= args.thresh:
                    continue

                z_xyxy = zoom_detections[j].xyxy.cpu().numpy().squeeze().astype(int)
                zx1, zy1, zx2, zy2 = z_xyxy
                z_classidx = int(zoom_detections[j].cls.item())
                z_classname = labels[z_classidx]
                z_color = (0, 0, 255)  # Red for zoomed detections

                # Map back to full frame
                gx1 = int(clamp(rx1 + zx1 * scale_x, 0, frame_w - 1))
                gy1 = int(clamp(ry1 + zy1 * scale_y, 0, frame_h - 1))
                gx2 = int(clamp(rx1 + zx2 * scale_x, 0, frame_w - 1))
                gy2 = int(clamp(ry1 + zy2 * scale_y, 0, frame_h - 1))

                cv2.rectangle(output_frame, (gx1, gy1), (gx2, gy2), z_color, 2)
                z_label = f'{z_classname}: {int(z_conf*100)}% (zoom)'
                z_size, z_base = cv2.getTextSize(z_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                z_y = max(gy1, z_size[1] + 10)
                cv2.rectangle(output_frame, (gx1, z_y-z_size[1]-10), 
                             (gx1+z_size[0], z_y+z_base-10), z_color, cv2.FILLED)
                cv2.putText(output_frame, z_label, (gx1, z_y-7), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                object_count += 1
                print(f"  [ZOOM] {z_classname}: {int(z_conf*100)}% (re-detected from ROI)")

    # Measure inference time
    t_stop = time.perf_counter()
    inference_time = (t_stop - t_start) * 1000  # Convert to ms

    # Add summary text to image
    cv2.putText(output_frame, f'Objects Detected: {object_count}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(output_frame, f'Inference Time: {inference_time:.1f}ms', (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Total Objects Detected: {object_count}")
    print(f"  Inference Time: {inference_time:.1f}ms")
    print(f"  Confidence Threshold: {args.thresh * 100:.0f}%")
    if args.focus_zoom:
        print(f"  Multi-scale Detection: ENABLED")
    print(f"{'='*60}\n")

    # Resize for display if needed
    display_frame = output_frame
    if resize:
        display_frame = cv2.resize(output_frame, (resW, resH))
        print(f"Displaying at {resW}×{resH}")
    else:
        print(f"Displaying at {frame_w}×{frame_h}")

    # Display result
    print("\nPress any key to close the image window...")
    cv2.imshow('Image Detection Result', display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save output if requested
    if args.save_output:
        output_path = args.output_name
        cv2.imwrite(output_path, output_frame)
        print(f"\n✓ Annotated image saved to: {output_path}")
    else:
        print(f"\nNote: To save the annotated image, use --save-output flag")

    print("Detection completed!\n")


if __name__ == "__main__":
    main()
