from collections import Counter
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

SUPPORTED_CLASSES = ["apple", "kiwi", "orange", "pear", "strawberry", "tomato"]

CLASS_COLORS = {
    "apple": (239, 68, 68),
    "kiwi": (34, 197, 94),
    "orange": (249, 115, 22),
    "pear": (132, 204, 22),
    "strawberry": (244, 63, 94),
    "tomato": (248, 113, 113),
}


def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    img = np.array(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def predict_and_annotate(model: YOLO, image: Image.Image, conf_threshold: float = 0.25):
    img_cv = pil_to_cv2(image)

    results = model.predict(
        source=img_cv,
        conf=conf_threshold,
        verbose=False
    )

    result = results[0]
    annotated = img_cv.copy()
    detections: List[Dict] = []

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)

        names = result.names

        for box, conf, cls_id in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            label = names[int(cls_id)]

            if label not in SUPPORTED_CLASSES:
                continue

            detections.append({
                "label": label,
                "confidence": float(conf)
            })

            color = CLASS_COLORS.get(label, (168, 85, 247))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            text = f"{label} {conf*100:.1f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)

            top_y = max(0, y1 - th - 12)
            cv2.rectangle(annotated, (x1, top_y), (x1 + tw + 12, y1), color, -1)
            cv2.putText(
                annotated,
                text,
                (x1 + 6, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    return cv2_to_pil(annotated), detections


def build_summary_text(detections: List[Dict]) -> str:
    if not detections:
        return (
            "I could not find any of the supported fruit classes in this image. "
            "This platform detects only: apple, kiwi, orange, pear, strawberry, and tomato."
        )

    counts = Counter([d["label"] for d in detections])
    total = sum(counts.values())

    parts = []
    for label, count in counts.items():
        if count == 1:
            parts.append(f"1 {label}")
        else:
            parts.append(f"{count} {label}s")

    if len(parts) == 1:
        joined = parts[0]
    else:
        joined = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    avg_conf = sum(d["confidence"] for d in detections) / len(detections)

    return f"We found {total} fruit(s) in this image: {joined}. Average confidence: {avg_conf*100:.1f}%."


def format_detection_table(detections: List[Dict]):
    if not detections:
        return []

    grouped = {}
    for d in detections:
        grouped.setdefault(d["label"], []).append(d["confidence"])

    rows = []
    for label, confs in grouped.items():
        avg_conf = (sum(confs) / len(confs)) * 100
        rows.append({
            "Fruit": label,
            "Confidence (%)": round(avg_conf, 2),
            "Count": len(confs)
        })

    rows.sort(key=lambda x: x["Fruit"])
    return rows
