import cv2
import numpy as np
import time
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torchvision.models import mobilenet_v2
import os

models = [
    {'name': 'yolov8n.pt', 'type': 'yolo'},
    {'name': 'yolov8s.pt', 'type': 'yolo'},
    {'name': 'yolov8m.pt', 'type': 'yolo'},
    {'name': 'yolov8x.pt', 'type': 'yolo'},
    {'name': 'faster_rcnn', 'type': 'faster_rcnn'},
    {'name': 'fasterrcnn_mobilenet', 'type': 'faster_rcnn'}
]


def get_fasterrcnn_mobilenet_model():
    # Load a pre-trained MobileNetV2 model
    backbone = mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    # Generate anchor sizes and aspect ratios
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    # Set up the ROI pooling
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2
    )

    # Create the Faster R-CNN model
    model = FasterRCNN(
        backbone,
        num_classes=2,  # 1 class (person) + background
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )

    return model

def load_model(model_name, model_type):
    if model_type == 'yolo':
        return YOLO(model_name)
    elif model_type == 'faster_rcnn':
        if model_name == 'fasterrcnn_mobilenet':
            model = get_fasterrcnn_mobilenet_model()
            model.eval()
            return model
        else:
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()
            return model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def calculate_iou(box1, box2):
    # Calculate intersection over union
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def calculate_map(detections, num_classes=1, iou_threshold=0.5, confidence_threshold=0.5):
    if not detections:
        return 0  # Return 0 if there are no detections

    average_precisions = []

    for class_id in range(num_classes):
        class_detections = [d for d in detections if d['class'] == class_id and d['confidence'] >= confidence_threshold]
        if not class_detections:
            continue

        class_detections.sort(key=lambda x: x['confidence'], reverse=True)

        num_gt = len(class_detections)  # Assuming each detection corresponds to a ground truth
        true_positives = np.zeros(len(class_detections))
        false_positives = np.zeros(len(class_detections))

        detected_gt = set()

        for i, detection in enumerate(class_detections):
            best_iou = 0
            best_gt_idx = -1

            for j, gt in enumerate(class_detections):
                if j in detected_gt:
                    continue

                iou = calculate_iou(detection['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold:
                if best_gt_idx not in detected_gt:
                    true_positives[i] = 1
                    detected_gt.add(best_gt_idx)
                else:
                    false_positives[i] = 1
            else:
                false_positives[i] = 1

        cumulative_tp = np.cumsum(true_positives)
        cumulative_fp = np.cumsum(false_positives)

        recalls = cumulative_tp / num_gt
        precisions = cumulative_tp / (cumulative_tp + cumulative_fp)

        # Compute average precision
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap = ap + p / 11
        average_precisions.append(ap)

    if not average_precisions:
        return 0  # Return 0 if no class had any detections

    return np.mean(average_precisions)


def detect_people(frame, model, model_type, conf_threshold=0.5, iou_threshold=0.5):
    start_time = time.time()

    if model_type == 'yolo':
        results = model(frame, conf=conf_threshold, iou=iou_threshold)
    elif model_type == 'faster_rcnn':
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = F.to_tensor(img)
        results = model([img])[0]

    inference_time = time.time() - start_time

    person_count = 0
    detections = []

    if model_type == 'yolo':
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.cls == 0:  # Class 0 is person in COCO dataset
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                                2)
                    detections.append({
                        'class': 0,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
    elif model_type == 'faster_rcnn':
        for i in range(len(results['boxes'])):
            box = results['boxes'][i].detach().numpy()
            score = results['scores'][i].detach().numpy()
            label = results['labels'][i].detach().numpy()
            if label == 1 and score >= conf_threshold:  # Label 1 is person in COCO dataset
                person_count += 1
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                detections.append({
                    'class': 0,
                    'confidence': score,
                    'bbox': [x1, y1, x2, y2]
                })

    return frame, person_count, inference_time, detections


def process_webcam():
    cap = cv2.VideoCapture(0)

    for model_name in models:
        print(f"\nMenggunakan model: {model_name}")
        model = YOLO(model_name)

        frame_count = 0
        total_inference_time = 0
        total_person_count = 0
        all_detections = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_result, person_count, inference_time, detections = detect_people(frame, model)

            total_inference_time += inference_time
            total_person_count += person_count
            all_detections.extend(detections)

            cv2.putText(frame_result, f"Model: {model_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame_result, f"Inference Time: {inference_time:.4f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

            cv2.imshow('Webcam Detection', frame_result)

            if cv2.waitKey(1) & 0xFF == ord('q') or frame_count >= 100:  # Process 100 frames
                break

        avg_inference_time = total_inference_time / frame_count
        avg_person_count = total_person_count / frame_count
        estimated_map = calculate_map(all_detections)

        print(f"Rata-rata jumlah orang terdeteksi per frame: {avg_person_count:.2f}")
        print(f"Rata-rata inference time: {avg_inference_time:.4f} detik")
        print(f"Estimated mAP: {estimated_map:.4f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def add_text_below_image(image, text):
    """Adds text below the image."""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    result_image = cv2.copyMakeBorder(image, 0, h + 20, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    cv2.putText(result_image, text, (10, image.shape[0] + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return result_image


def process_image(file_path):
    img = cv2.imread(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    upload_folder = os.path.dirname(file_path)

    static_upload_folder = os.path.join('static', 'uploads')
    if not os.path.exists(static_upload_folder):
        os.makedirs(static_upload_folder)

    results = {}

    for model_info in models:
        print(f"\nUsing model: {model_info['name']}")
        model = load_model(model_info['name'], model_info['type'])

        start_time = time.time()
        img_result, person_count, inference_time, detections = detect_people(img.copy(), model, model_info['type'])
        latency_time = time.time() - start_time

        estimated_map = calculate_map(detections)

        print(f"Number of people detected: {person_count}")
        print(f"Inference time: {inference_time:.4f} seconds")
        print(f"Latency time: {latency_time:.4f} seconds")
        print(f"Estimated mAP: {estimated_map:.4f}")

        text = f"People: {person_count}, Inference time: {inference_time:.4f}s"
        result_image_with_text = add_text_below_image(img_result, text)

        result_image_path = os.path.join(static_upload_folder, f"{base_name}_{model_info['name']}_result.jpg")
        cv2.imwrite(result_image_path, result_image_with_text)

        results[model_info['name']] = {
            'person_count': person_count,
            'inference_time': inference_time,
            'latency_time': latency_time,
            'estimated_map': estimated_map,
            'image_path': os.path.relpath(result_image_path, 'static').replace("\\", "/")
        }

    original_image_path = os.path.join(static_upload_folder, f"{base_name}_original.jpg")
    cv2.imwrite(original_image_path, img)

    print(f"\nOriginal image saved at {original_image_path}")

    for model_name, result in results.items():
        print(f"\nResults for {model_name}:")
        print(f"Number of people detected: {result['person_count']}")
        print(f"Inference time: {result['inference_time']:.4f} seconds")
        print(f"Latency time: {result['latency_time']:.4f} seconds")
        print(f"Estimated mAP: {result['estimated_map']:.4f}")
        print(f"Result image saved at {result['image_path']}")

    return os.path.relpath(original_image_path, 'static').replace("\\", "/"), results


def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    static_upload_folder = os.path.join('static', 'uploads')

    if not os.path.exists(static_upload_folder):
        os.makedirs(static_upload_folder)

    results = {}  # Initialize results dictionary

    for model_info in models:
        print(f"\nUsing model: {model_info['name']}")
        try:
            model = load_model(model_info['name'], model_info['type'])
            print(f"Model {model_info['name']} loaded successfully.")
        except Exception as e:
            print(f"Failed to load model {model_info['name']}: {e}")
            continue

        frame_count = 0
        total_inference_time = 0
        total_person_count = 0
        all_detections = []

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Ensure correct codec for .mp4
        result_video_path = os.path.join(static_upload_folder, f"{base_name}_{model_info['name']}_result.mp4")
        out = cv2.VideoWriter(result_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            try:
                frame_result, person_count, inference_time, detections = detect_people(frame, model, model_info['type'])
                print(f"Frame {frame_count} processed with model {model_info['name']}.")
            except Exception as e:
                print(f"Failed to process frame {frame_count} with model {model_info['name']}: {e}")
                continue

            total_inference_time += inference_time
            total_person_count += person_count
            all_detections.extend(detections)

            cv2.putText(frame_result, f"Model: {model_info['name']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            cv2.putText(frame_result, f"Person (amount): {person_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            cv2.putText(frame_result, f"Frame: {frame_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            out.write(frame_result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_count > 0:
            avg_inference_time = total_inference_time / frame_count
            avg_person_count = total_person_count / frame_count
            estimated_map = calculate_map(all_detections)

            print(f"Average number of people detected per frame: {avg_person_count:.2f}")
            print(f"Average inference time: {avg_inference_time:.4f} seconds")
            print(f"Estimated mAP: {estimated_map:.4f}")

            # Save video information
            results[model_info['name']] = {
                'avg_person_count': avg_person_count,
                'avg_inference_time': avg_inference_time,
                'estimated_map': estimated_map,
                'video_path': os.path.relpath(result_video_path, 'static').replace("\\", "/")
            }
        else:
            print(f"No frames were processed for model: {model_info['name']}")

        # Release the VideoCapture and VideoWriter objects for this model
        out.release()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset the video capture to the first frame

    cap.release()
    cv2.destroyAllWindows()

    return results


def main():
    while True:
        print("\nPilih mode:")
        print("1. Deteksi menggunakan webcam")
        print("2. Deteksi pada gambar")
        print("3. Deteksi pada video")
        print("4. Keluar")

        choice = input("Masukkan pilihan (1/2/3/4): ")

        if choice == '1':
            process_webcam()
        elif choice == '2':
            process_image()
        elif choice == '3':
            process_video()
        elif choice == '4':
            break
        else:
            print("Pilihan tidak valid. Silakan coba lagi.")


if __name__ == "__main__":
    main()
