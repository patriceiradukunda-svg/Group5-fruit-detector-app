from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

SUPPORTED_CLASSES = ["apple", "kiwi", "orange", "pear", "strawberry", "tomato"]

# Confidence below this triggers a "low confidence / may be wrong" warning
LOW_CONFIDENCE_WARNING_THRESHOLD = 0.45

CLASS_COLORS = {
    "apple":      (239, 68,  68),
    "kiwi":       (34,  197, 94),
    "orange":     (249, 115, 22),
    "pear":       (132, 204, 22),
    "strawberry": (244, 63,  94),
    "tomato":     (248, 113, 113),
}


def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def predict_and_annotate(
    model: YOLO,
    image: Image.Image,
    conf_threshold: float = 0.25
) -> Tuple[Image.Image, List[Dict], bool]:
    img_rgb = image.convert("RGB")
    img_np  = np.array(img_rgb)

    results = model.predict(source=img_np, conf=conf_threshold, verbose=False)

    result    = results[0]
    annotated = img_rgb.copy()
    draw      = ImageDraw.Draw(annotated)
    detections: List[Dict] = []
    low_conf_warning = False

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss  = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names

        for box, conf, cls_id in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            label = names[int(cls_id)]

            if label not in SUPPORTED_CLASSES:
                continue

            if float(conf) < LOW_CONFIDENCE_WARNING_THRESHOLD:
                low_conf_warning = True

            detections.append({"label": label, "confidence": float(conf)})

            color = CLASS_COLORS.get(label, (168, 85, 247))

            for t in range(3):
                draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)

            text = f"{label} {conf * 100:.1f}%"
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            pad = 4
            top_y = max(0, y1 - th - pad * 2)
            draw.rectangle([x1, top_y, x1 + tw + pad * 2, y1], fill=color)
            draw.text((x1 + pad, top_y + pad // 2), text, fill=(255, 255, 255), font=font)

    return annotated, detections, low_conf_warning


def build_summary_text(detections: List[Dict]) -> str:
    if not detections:
        return (
            "No supported fruits were detected in this image. "
            "This platform only detects: apple, kiwi, orange, pear, strawberry, and tomato. "
            "If you uploaded a mango, banana, grape, or any other fruit, "
            "the model cannot recognise it — it may give a wrong label or no label at all."
        )
    counts = Counter([d["label"] for d in detections])
    total  = sum(counts.values())
    parts  = [f"{c} {l}{'s' if c > 1 else ''}" for l, c in counts.items()]
    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f", and {parts[-1]}"
    avg_conf = sum(d["confidence"] for d in detections) / len(detections)
    return f"We found {total} fruit(s) in this image: {joined}. Average confidence: {avg_conf * 100:.1f}%."


def build_low_conf_warning() -> str:
    return (
        "⚠️ **Low confidence detected.** One or more detections scored below 45%. "
        "This often happens when the image contains a fruit the model was **not trained on** "
        "(e.g. mango, banana, grape, watermelon). "
        "The model tried its best guess from the 6 supported classes — but the result may be **incorrect**. "
        "Only apple, kiwi, orange, pear, strawberry, and tomato are reliably supported."
    )


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
            "Fruit":          label,
            "Confidence (%)": round(avg_conf, 2),
            "Count":          len(confs),
            "Reliable?":      "✅ Yes" if avg_conf >= 45 else "⚠️ Low — may be wrong"
        })
    rows.sort(key=lambda x: x["Fruit"])
    return rows
