import os
import cv2
from PIL import Image
from typing import List
from rapidfuzz import process, fuzz


def normalize(text):
    return text.lower().replace("a ", "").strip()

def normalize_label(label):
    return label.lower().replace("a ", "").strip()

def is_similar(label, product_names, threshold=60):
    label_norm = normalize_label(label)
    for name in product_names:
        name_norm = normalize_label(name)
        if fuzz.partial_ratio(label_norm, name_norm) >= threshold:
            return True
    return False

def match_to_product_name(detected_label, product_names, threshold=60):
    norm_label = normalize(detected_label)
    norm_names = [normalize(p) for p in product_names]
    match = process.extractOne(norm_label, norm_names, scorer=fuzz.partial_ratio)
    if match and match[1] >= threshold:
        return product_names[norm_names.index(match[0])]
    return None

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0])) * max(1, (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0])) * max(1, (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def detect_individual(image_path, product_names, detect_with_huggingface, threshold=0.3):
    final_items = []
    for name in product_names:
        results = detect_with_huggingface(image_path, [name])
        if results:
            top_result = sorted(results, key=lambda x: x["confidence"], reverse=True)[0]
            if top_result["confidence"] >= threshold:
                top_result["item_name"] = name
                final_items.append(top_result)
    return final_items

def detect_grouped(image_path, product_names, detect_with_huggingface):
    return detect_with_huggingface(image_path, product_names)

def detect_grouped_with_matching(image_path, product_names, detect_with_huggingface, iou_threshold=0.5):
    raw_items = detect_with_huggingface(image_path, product_names)
    matched = []

    for item in raw_items:
        corrected = match_to_product_name(item["item_name"], product_names)
        if corrected:
            item["item_name"] = corrected
            matched.append(item)

    unique = []
    for i, current in enumerate(matched):
        keep = True
        for prev in unique:
            if (normalize(current["item_name"]) == normalize(prev["item_name"])
                and compute_iou(current["bounding_box"], prev["bounding_box"]) > iou_threshold):
                if current["confidence"] > prev["confidence"]:
                    prev.update(current)
                keep = False
                break
        if keep:
            unique.append(current)
    print(f"Grouped-matching before dedup: {len(matched)}, after dedup: {len(unique)}")
    return unique

def print_detection_summary(title: str, prompts: List[str], detections: List[dict]):
    print(f"\n=== {title.upper()} DETECTION ===")
    print("Product names (normalized prompts):", prompts)
    print("Detection results:")
    for item in detections:
        label = item.get("item_name", "Unknown")
        conf = item.get("confidence", 0.0)
        box = item.get("bounding_box", [])
        print(f"- {label} | confidence: {conf:.3f} | box: {box}")

def draw_and_save(image_path, products, output_path, title=""):
    image = cv2.imread(image_path)

    for product in products:
        label = product.get("item_name", "Unknown")
        confidence = product.get("confidence", 0.0)
        bbox = product.get("bounding_box", [])

        if bbox and len(bbox) == 4:
            try:
                x_min, y_min, x_max, y_max = map(int, bbox)
                color = (
                    (0, 255, 0) if confidence >= 0.40
                    else (0, 165, 255) if confidence >= 0.30
                    else (0, 0, 255)
                )
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 3)
                label_text = f"{label} ({confidence:.2f})"
                cv2.putText(image, label_text, (x_min, max(y_min - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                coord_text = f"{x_min},{y_min},{x_max},{y_max}"
                cv2.putText(image, coord_text, (x_min, y_max + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            except Exception as e:
                print(f"Error drawing box for {label}: {e}")

    if title:
        if "blurred_people" in title.lower():
            title_color = (211, 0, 255)  # Purple
        elif "blurred_background" in title.lower():
            title_color = (139, 0, 139)  # Dark purple-blue
        else:
            title_color = (255, 255, 0)  # Yellow (original)

        cv2.putText(image, title, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, title_color, 2)

    cv2.imwrite(output_path, image)
    print(f"Saved output with boxes at: {output_path}")




def save_bounding_boxes(img_path, items, save_dir, prefix=""):
    image = Image.open(img_path).convert("RGB")

    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []

    for idx, item in enumerate(items):
        box = item.get("bounding_box")
        name = item.get("item_name", f"item_{idx}")
        x1, y1, x2, y2 = map(int, box)

        cropped = image.crop((x1, y1, x2, y2))

        filename = f"{prefix}_{idx}_{normalize(name)}.jpg"
        path = os.path.join(save_dir, filename)
        cropped.save(path)
        saved_paths.append(path)
    
    return saved_paths