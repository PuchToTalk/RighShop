from PIL import Image as PILImage
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from difflib import SequenceMatcher

def detect_label_with_huggingface(image, text_prompts: list, min_size=32):
    """
    Returns (best_label, best_score) using GroundingDINO for the given image and prompts.
    Skips tiny images (like 1x1 pixels) to avoid processor errors.
    """
    # Sanitize to RGB PIL.Image
    image = sanitize_to_rgb(image)

    # Skip very small images (tracking pixels, favicons, etc.)
    if image.width < min_size or image.height < min_size:
        print(f"[DEBUG] Skipping image too small ({image.width}x{image.height})")
        return "", 0.0

    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    text_prompt = ". ".join(f"a {name}" for name in text_prompts if name) + "."

    inputs = processor(
        images=image,
        text=text_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        threshold=0.35,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )[0]

    # If no detections
    if len(results["scores"]) == 0:
        return "", 0.0

    # Find best label
    scores = results["scores"].cpu().numpy()
    labels = results["text_labels"]

    best_idx = scores.argmax()
    return labels[best_idx], float(scores[best_idx])


def sanitize_to_rgb(image):
    """
    Ensure the input image is converted to a valid RGB PIL.Image.
    Handles:
      - Numpy arrays in (H, W), (H, W, 1), (H, W, 3), or (C, H, W) format
      - Converts grayscale to RGB by duplicating channels
      - Validates that the final output has 3 channels
    """
    if isinstance(image, np.ndarray):
        # Remove singleton dimensions (e.g., (1, 1, 3))
        image = np.squeeze(image)

        # If image is (C, H, W) format, move channels to the end
        if image.ndim == 3 and image.shape[0] <= 4 and image.shape[-1] not in (1, 3):
            image = np.moveaxis(image, 0, -1)

        # If grayscale with shape (H, W, 1), replicate to 3 channels
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        # If grayscale 2D (H, W), stack to make (H, W, 3)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Final check: must have 3 channels
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"sanitize_to_rgb: Unexpected shape {image.shape}")

        # Convert to PIL Image
        image = PILImage.fromarray(image.astype("uint8"))

    elif not isinstance(image, PILImage.Image):
        raise ValueError(f"sanitize_to_rgb: Unsupported image type {type(image)}")

    # Ensure RGB mode
    return image.convert("RGB")



def verify_image_with_dino(image, product_name, threshold=0.5, verbose=True):
    """
    Verify if the given image matches the product_name using Grounding DINO.
    Returns True if a match is found with confidence >= threshold, else False.
    """
    if verbose:
        print(f"[DINO] Verifying: '{product_name}'")

    # Ensure image is valid RGB
    image = sanitize_to_rgb(image)

    detections = detect_with_huggingface(image, [product_name])
    if not detections:
        if verbose:
            print(f"[DINO] No detections for '{product_name}'")
        return False

    # Find highest-confidence detection
    best = max(detections, key=lambda x: x["confidence"])
    score = best["confidence"]
    label = best["item_name"]

    if verbose:
        print(f"[DINO] Detection: '{label}' (score={score:.3f}, threshold={threshold})")

    # Match decision
    match = score >= threshold
    if verbose:
        print(f"[DINO] Result: {'MATCH' if match else 'NO MATCH'}")

    return match

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


