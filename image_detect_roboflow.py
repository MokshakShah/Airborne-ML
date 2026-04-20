"""
Roboflow-based Image Detection and Classification
Detects objects in static images using Roboflow Workflow API with green bounding box visualization
"""

import argparse
import base64
import cv2
import numpy as np
import requests
from pathlib import Path
import time


class RoboflowImageClient:
    """Roboflow Workflow API client for image detection"""
    
    def __init__(self, api_key, api_url="https://serverless.roboflow.com", 
                 workspace="mokshaks-workspace", workflow="general-segmentation-api-2", max_retries=3):
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.workspace = workspace
        self.workflow = workflow
        self.max_retries = max_retries
        self.session = requests.Session()
    
    @staticmethod
    def encode_image_base64(image_path):
        """Encode image to base64"""
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def detect(self, image_path, classes="bird, drone, airplane"):
        """
        Send image to Roboflow Workflow API for inference
        
        Args:
            image_path: Path to image file
            classes: Classes to detect (comma-separated)
        
        Returns:
            dict: Predictions or empty dict if failed
        """
        try:
            # Encode image to base64
            encoded = self.encode_image_base64(image_path)
            if not encoded:
                return {}
            
            # Prepare payload for Workflow API
            payload = {
                "api_key": self.api_key,
                "use_cache": True,
                "inputs": {
                    "image": {"type": "base64", "value": encoded},
                    "classes": classes,
                },
            }
            
            # Try multiple endpoint formats
            url_candidates = [
                f"{self.api_url}/{self.workspace}/workflows/{self.workflow}",
                f"{self.api_url}/infer/workflows/{self.workspace}/{self.workflow}",
            ]
            
            last_error = None
            for attempt, url in enumerate(url_candidates):
                try:
                    print(f"  Trying endpoint {attempt + 1}/{len(url_candidates)}: {url.split('/')[-1]}")
                    response = self.session.post(
                        url,
                        json=payload,
                        timeout=120,
                        verify=True
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        print(f"  Status {response.status_code}: {response.text[:100]}")
                        last_error = f"Status {response.status_code}"
                        continue
                
                except requests.exceptions.Timeout:
                    print(f"  Timeout")
                    last_error = "Timeout"
                    continue
                
                except requests.exceptions.ConnectionError as e:
                    print(f"  Connection error: {str(e)[:50]}")
                    last_error = "Connection error"
                    continue
                
                except Exception as e:
                    print(f"  Error: {str(e)[:50]}")
                    last_error = str(e)
                    continue
            
            if last_error:
                print(f"❌ All endpoints failed: {last_error}")
            return {}
        
        except FileNotFoundError:
            print(f"Error: Image file not found - {image_path}")
            return {}
        except Exception as e:
            print(f"Error during detection: {e}")
            return {}


def extract_predictions(api_response):
    """Extract predictions from Roboflow Workflow API response"""
    try:
        predictions = []
        
        # Workflow API returns nested structure
        def walk(node):
            if isinstance(node, dict):
                # Check if this node is a detection (has bounding box and class)
                has_bbox = all(k in node for k in ("x", "y", "width", "height"))
                has_label = ("class" in node) or ("class_name" in node)
                
                if has_bbox and has_label:
                    predictions.append({
                        'class': node.get('class', node.get('class_name', 'Unknown')),
                        'confidence': node.get('confidence', 0.0),
                        'x': node.get('x', 0),
                        'y': node.get('y', 0),
                        'width': node.get('width', 0),
                        'height': node.get('height', 0)
                    })
                
                # Recurse into all values
                for value in node.values():
                    walk(value)
            
            elif isinstance(node, list):
                # Recurse into list items
                for item in node:
                    walk(item)
        
        # Walk the entire response
        walk(api_response)
        
        return predictions
    except Exception as e:
        print(f"Error extracting predictions: {e}")
        return []


def draw_predictions(image, predictions, confidence_threshold=0.0):
    """
    Draw green bounding boxes with labels on image
    
    Args:
        image: OpenCV image
        predictions: List of prediction dictionaries
        confidence_threshold: Only draw if confidence > threshold
    
    Returns:
        Annotated image, detection count
    """
    annotated = image.copy()
    detection_count = 0
    
    for pred in predictions:
        # Skip low confidence predictions
        conf = pred.get('confidence', 0)
        if conf and conf < confidence_threshold:
            continue
        
        detection_count += 1
        
        # Extract coordinates
        x = int(pred['x'])
        y = int(pred['y'])
        w = int(pred['width'] / 2)
        h = int(pred['height'] / 2)
        
        # Calculate bounding box corners
        x1 = max(0, x - w)
        y1 = max(0, y - h)
        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)
        
        # Draw green bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with confidence
        if conf:
            label = f"{pred['class']}: {conf:.1%}"
        else:
            label = f"{pred['class']}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        # Get text size for background
        (text_width, text_height) = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Draw white background for text
        cv2.rectangle(
            annotated,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 5, y1),
            (255, 255, 255),
            -1
        )
        
        # Draw text
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - 5),
            font,
            font_scale,
            (0, 0, 0),
            thickness
        )
    
    return annotated, detection_count


def display_image_with_stats(image, detection_count, inference_time_ms):
    """Display image with statistics overlay"""
    display = image.copy()
    
    # Add inference time and count at top
    stats_text = f"Detections: {detection_count} | Inference: {inference_time_ms:.1f}ms"
    cv2.putText(
        display,
        stats_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    return display


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Roboflow Workflow-based image detection with green bounding boxes'
    )
    
    parser.add_argument(
        '--image',
        required=True,
        help='Path to input image (jpg, png, etc.)'
    )
    
    parser.add_argument(
        '--api-key',
        required=True,
        help='Roboflow API key'
    )
    
    parser.add_argument(
        '--api-url',
        default='https://serverless.roboflow.com',
        help='Roboflow API URL (default: https://serverless.roboflow.com)'
    )
    
    parser.add_argument(
        '--workspace',
        default='mokshaks-workspace',
        help='Roboflow workspace name (default: mokshaks-workspace)'
    )
    
    parser.add_argument(
        '--workflow',
        default='general-segmentation-api-2',
        help='Workflow ID (default: general-segmentation-api-2)'
    )
    
    parser.add_argument(
        '--classes',
        default='bird, drone, airplane',
        help='Classes to detect (comma-separated, default: bird, drone, airplane)'
    )
    
    parser.add_argument(
        '--save-output',
        action='store_true',
        help='Save annotated image to file'
    )
    
    parser.add_argument(
        '--output-name',
        default='roboflow_detection.jpg',
        help='Output filename for annotated image, default: roboflow_detection.jpg'
    )
    
    parser.add_argument(
        '--resolution',
        default=None,
        help='Image resolution (e.g., 1280x720), default: original'
    )
    
    return parser.parse_args()


def main():
    """Main detection pipeline"""
    args = parse_args()
    
    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image not found - {args.image}")
        return
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Error: Could not load image - {args.image}")
        return
    
    print(f"📸 Image loaded: {image_path.name} ({image.shape[1]}x{image.shape[0]})")
    
    # Resize if requested
    if args.resolution:
        try:
            w, h = map(int, args.resolution.split('x'))
            image = cv2.resize(image, (w, h))
            print(f"📐 Resized to: {w}x{h}")
        except ValueError:
            print(f"⚠️ Invalid resolution format, using original")
    
    # Initialize Roboflow Workflow client
    print(f"🔄 Sending to Roboflow Workflow API...")
    print(f"   Workspace: {args.workspace}")
    print(f"   Workflow: {args.workflow}")
    print(f"   Classes: {args.classes}")
    
    client = RoboflowImageClient(
        api_key=args.api_key,
        api_url=args.api_url,
        workspace=args.workspace,
        workflow=args.workflow
    )
    
    # Inference
    start_time = time.time()
    api_response = client.detect(str(image_path), classes=args.classes)
    inference_time_ms = (time.time() - start_time) * 1000
    
    if not api_response:
        print("❌ Detection failed - check API key, workspace, workflow, and connection")
        return
    
    # Extract predictions
    predictions = extract_predictions(api_response)
    
    if not predictions:
        print("⚠️ No detections found")
        annotated = image.copy()
        detection_count = 0
    else:
        print(f"✅ {len(predictions)} detections received")
        annotated, detection_count = draw_predictions(image, predictions, confidence_threshold=0.0)
    
    # Display with statistics
    display = display_image_with_stats(annotated, detection_count, inference_time_ms)
    
    # Show detection details
    print(f"\n📊 Detection Results:")
    print(f"   • Objects detected: {detection_count}")
    print(f"   • Inference time: {inference_time_ms:.1f}ms")
    print(f"   • Classes: {args.classes}")
    
    if predictions:
        print(f"\n   Detections:")
        for i, pred in enumerate(predictions[:15], 1):
            conf = pred.get('confidence', 0)
            if conf:
                print(f"   {i}. {pred['class']}: {conf:.1%}")
            else:
                print(f"   {i}. {pred['class']}")
        if len(predictions) > 15:
            print(f"   ... and {len(predictions) - 15} more")
    
    # Save output if requested
    if args.save_output:
        # Create images folder if it doesn't exist
        images_folder = Path('images')
        images_folder.mkdir(exist_ok=True)
        
        # Save to images folder
        output_path = images_folder / args.output_name
        cv2.imwrite(str(output_path), annotated)
        print(f"\n💾 Saved to: images/{args.output_name}")
    
    # Display image
    print(f"\n👀 Displaying image (press any key to close)...")
    cv2.imshow('Roboflow Image Detection', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("✨ Done!")


if __name__ == '__main__':
    main()
