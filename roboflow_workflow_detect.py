import argparse
import base64
import os
import tempfile
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from inference_sdk import InferenceHTTPClient
except Exception:
    InferenceHTTPClient = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time Roboflow workflow inference with live video display."
    )
    parser.add_argument("--source", required=True, help="Video file path or 'usb0' for webcam.")
    parser.add_argument(
        "--api-key",
        default=os.getenv("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key. Prefer env var ROBOFLOW_API_KEY.",
    )
    parser.add_argument(
        "--api-url",
        default="https://serverless.roboflow.com",
        help="Roboflow inference API URL.",
    )
    parser.add_argument("--workspace", default="mokshaks-workspace", help="Workspace name.")
    parser.add_argument("--workflow", default="general-segmentation-api-2", help="Workflow ID.")
    parser.add_argument(
        "--classes",
        default="bird, drone, airplane",
        help="Class filter for workflow parameter.",
    )
    parser.add_argument("--confidence", type=float, default=0.25, help="Display confidence threshold.")
    parser.add_argument("--stride", type=int, default=1, help="Run inference every Nth frame for videos.")
    parser.add_argument(
        "--resolution",
        default=None,
        help="Display resolution in WxH format (e.g., '1280x720')",
    )
    parser.add_argument(
        "--transport",
        choices=["auto", "sdk", "rest"],
        default="auto",
        help="Inference transport. Use 'rest' for Python 3.13 compatibility.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification (only if SSL handshake fails on your network).",
    )
    return parser.parse_args()


class RoboflowWorkflowClient:
    def __init__(self, api_url, api_key, workspace, workflow, transport="auto", insecure=False):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.workspace = workspace
        self.workflow = workflow
        self.transport = transport
        self.insecure = insecure

        use_sdk = (
            transport in ("auto", "sdk")
            and InferenceHTTPClient is not None
        )
        self._sdk_client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key) if use_sdk else None
        self._session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        if transport == "sdk" and self._sdk_client is None:
            raise RuntimeError("inference-sdk not available in this Python environment.")

    @staticmethod
    def _encode_image_base64(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def run_workflow(self, image_path, classes):
        if self._sdk_client is not None:
            return self._sdk_client.run_workflow(
                workspace_name=self.workspace,
                workflow_id=self.workflow,
                images={"image": image_path},
                parameters={"classes": classes},
                use_cache=True,
            )

        encoded = self._encode_image_base64(image_path)
        url_candidates = [
            f"{self.api_url}/{self.workspace}/workflows/{self.workflow}",
            f"{self.api_url}/infer/workflows/{self.workspace}/{self.workflow}",
        ]
        payload = {
            "api_key": self.api_key,
            "use_cache": True,
            "inputs": {
                "image": {"type": "base64", "value": encoded},
                "classes": classes,
            },
        }
        last_error = None
        for url in url_candidates:
            try:
                response = self._session.post(
                    url,
                    json=payload,
                    timeout=120,
                    verify=not self.insecure,
                )
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Workflow request failed across endpoints: {last_error}")


def extract_predictions(result):
    predictions = []

    def walk(node):
        if isinstance(node, dict):
            has_bbox = all(k in node for k in ("x", "y", "width", "height"))
            has_label = ("class" in node) or ("class_name" in node)
            if has_bbox and has_label:
                predictions.append(
                    {
                        "x": node.get("x"),
                        "y": node.get("y"),
                        "width": node.get("width"),
                        "height": node.get("height"),
                        "class": node.get("class", node.get("class_name", "object")),
                        "confidence": node.get("confidence"),
                    }
                )
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(result)
    return predictions


def draw_predictions(frame, predictions, min_conf):
    """Draw green bounding boxes and labels on frame like YOLO demo."""
    object_count = 0
    for pred in predictions:
        conf_raw = pred.get("confidence")
        conf = float(conf_raw) if conf_raw is not None else None
        if conf is not None and conf < min_conf:
            continue

        x = int(pred.get("x", 0))
        y = int(pred.get("y", 0))
        w = int(pred.get("width", 0))
        h = int(pred.get("height", 0))
        cls = str(pred.get("class", "object")).strip()

        x1 = max(0, x - w // 2)
        y1 = max(0, y - h // 2)
        x2 = min(frame.shape[1] - 1, x + w // 2)
        y2 = min(frame.shape[0] - 1, y + h // 2)

        # Green box and label (like YOLO demo)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Format label with confidence percentage
        label = f"{cls}: {int(conf*100)}%" if conf is not None else cls
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(y1, labelSize[1] + 10)
        cv2.rectangle(frame, (x1, label_ymin-labelSize[1]-10), (x1+labelSize[0], label_ymin+baseLine-10), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, label, (x1, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        object_count += 1
    
    return object_count


def main():
    args = parse_args()
    if not args.api_key:
        raise RuntimeError("Missing API key. Pass --api-key or set ROBOFLOW_API_KEY.")
    if args.stride < 1:
        raise RuntimeError("--stride must be >= 1.")

    src_path = Path(args.source)
    
    # Check if source is USB camera or video file
    if "usb" in str(args.source):
        usb_idx = int(args.source[3:])
        cap = cv2.VideoCapture(usb_idx)
    elif os.path.isfile(src_path):
        cap = cv2.VideoCapture(str(src_path))
    else:
        raise RuntimeError(f"Source path does not exist or invalid: {src_path}")
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    # Parse resolution if specified
    resize = False
    resW, resH = None, None
    if args.resolution:
        resize = True
        resW, resH = int(args.resolution.split('x')[0]), int(args.resolution.split('x')[1])

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if resize:
        display_width, display_height = resW, resH
    else:
        display_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        display_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize client
    client = RoboflowWorkflowClient(
        api_url=args.api_url,
        api_key=args.api_key,
        workspace=args.workspace,
        workflow=args.workflow,
        transport=args.transport,
        insecure=args.insecure,
    )

    frame_idx = 0
    processed = 0
    display_enabled = True
    frame_rate_buffer = deque(maxlen=200)
    avg_frame_rate = 0

    print(f"\nStarting real-time inference on: {args.source}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Stride: {args.stride} (infer every {args.stride} frame(s))")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Pause")
    print("  'p' - Save frame capture")
    print("\n")

    try:
        while True:
            t_start = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                print("\nReached end of video. Exiting.")
                break

            run_now = (frame_idx % args.stride == 0)
            predictions = []
            object_count = 0

            if run_now:
                # Prepare frame for inference
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    temp_path = tmp.name
                try:
                    cv2.imwrite(temp_path, frame)
                    result = client.run_workflow(image_path=temp_path, classes=args.classes)
                    predictions = extract_predictions(result)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                processed += 1

            # Draw predictions on frame
            output_frame = frame.copy()
            object_count = draw_predictions(output_frame, predictions, args.confidence)

            # Resize for display if needed
            if resize:
                output_frame = cv2.resize(output_frame, (resW, resH))

            # Add FPS and object count info
            cv2.putText(output_frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(output_frame, f'Number of objects: {object_count}', (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(output_frame, f'Frame: {frame_idx} | Processed: {processed}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display
            if display_enabled:
                try:
                    cv2.imshow('Roboflow Real-time Detection', output_frame)
                except cv2.error:
                    display_enabled = False
                    print("WARNING: OpenCV GUI not available. Continuing without display window.")

            # Handle keyboard input
            key = cv2.waitKey(5) if display_enabled else -1
            if key == ord('q') or key == ord('Q'):
                print("\nQuit requested.")
                break
            elif key == ord('s') or key == ord('S'):
                print("Paused. Press any key to continue...")
                cv2.waitKey()
            elif key == ord('p') or key == ord('P'):
                cv2.imwrite('roboflow_capture.png', output_frame)
                print(f"Frame captured: roboflow_capture.png")

            # Calculate FPS
            frame_idx += 1
            t_stop = time.perf_counter()
            frame_rate_calc = float(1 / (t_stop - t_start))
            frame_rate_buffer.append(frame_rate_calc)
            avg_frame_rate = np.mean(frame_rate_buffer)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nAverage FPS: {avg_frame_rate:.2f}")
        print(f"Total frames processed: {processed}")


if __name__ == "__main__":
    main()
