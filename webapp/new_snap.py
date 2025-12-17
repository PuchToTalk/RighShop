import os
import base64
import io
import requests
import tempfile
from PIL import Image as PILImage
import subprocess
from openai import OpenAI
###from dotenv import load_dotenv
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from wikipedia import add_wiki_links
from shop_search import format_search_query, get_product_urls, get_product_images, get_valid_image, is_link_relevant_with_llm, quick_keyword_filter, overlap_score, verify_image_with_llm, fetch_main_image_from_page
import re
import uuid
from dotenv import load_dotenv
import json
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from pydantic import BaseModel
from typing import List
from google import genai
from google.genai.types import GenerateContentConfig
from video_data import get_video_url, get_video_metadata, get_transcript_window, extract_video_id, metadata_exists, scrape_video_metadata, upload_video_metadata
# from pip_embeddings import get_embeddings, save_embeddings_bulk
from upload_data import upload_image_from_str
from sign import generate_signed_url
import torch
from detection_utils import normalize, detect_grouped_with_matching
from dino_utils import detect_label_with_huggingface, sanitize_to_rgb, verify_image_with_dino, similarity
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Set API key and model name
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

openai_client = OpenAI()
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
# gemini_pro_client = OpenAI(
#     api_key=GEMINI_API_KEY,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

is_processing = False # track image processing

with open("prompts/system_prompt.txt", "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = file.read()

usage = []
latency = []
responses = []

def preprocess_image(in_bytes, out_bytes):
    with PILImage.open(in_bytes) as img:
        img = img.convert("RGB")
        img.save(out_bytes, format="JPEG", quality=95, optimize=True)

def encode_image(image_path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def save_as_jpg(image, jpg_image_path):
    # save image locally
    with PILImage.open(image) as img:
        img = img.convert("RGB")
        img.save(jpg_image_path, "JPEG", quality=95, optimize=True)

def base64_to_pil_image(base64_image: str) -> PILImage.Image:
    image_data = base64.b64decode(base64_image)
    return PILImage.open(io.BytesIO(image_data)).convert("RGB")

def convert_to_dict(json_str: str) -> dict:
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print("JSONDecodeError:", e)
        return

# extract and summarize key information from description and transcript
def clean_metadata(description: str, transcript: str = "", time_position: int = 0, duration: int = 0) -> dict:
    if transcript:
        window_s = min(300000, duration / 10)
        print(window_s)
        transcript_window = get_transcript_window(transcript, time_position, window_s) if duration else ""
        if transcript_window:
            with open("prompts/metadata_prompt.txt", "r", encoding="utf-8") as file:
                METADATA_PROMPT = file.read()

            PROMPT = METADATA_PROMPT.format(description=description, transcript=transcript_window)

            class Metadata(BaseModel):
                people: list[str]
                items_and_brands: list[str]
                locations: list[str]
                transcript_summary: str
        else:
            print("No transcript")
            with open("prompts/metadata_prompt_no_transcript.txt", "r", encoding="utf-8") as file:
                METADATA_PROMPT = file.read()
                
            PROMPT = METADATA_PROMPT.format(description=description)

            class Metadata(BaseModel):
                people: list[str]
                items_and_brands: list[str]
                locations: list[str]
    else: 
        with open("prompts/metadata_prompt_no_transcript.txt", "r", encoding="utf-8") as file:
            METADATA_PROMPT = file.read()

        PROMPT = METADATA_PROMPT.format(description=description)

        class Metadata(BaseModel):
            people: list[str]
            items_and_brands: list[str]
            locations: list[str]

    try:
        model = "gemini-2.0-flash"
        start_time = time.perf_counter()
        response = gemini_client.models.generate_content(
            model = model,
            contents=PROMPT,
            config={
                "response_mime_type": "application/json",
                "response_schema": Metadata
            }
        )
        final_response = response.parsed.model_dump()
        end_time = time.perf_counter()
        response_time = end_time - start_time

        # usage info
        prompt_tokens = response.usage_metadata.prompt_token_count
        completion_tokens = response.usage_metadata.candidates_token_count
        total_tokens = response.usage_metadata.total_token_count

    except Exception as e:
        print(f"clean_metadata error: {e}")
        content = [
            {"type": "text", "text": PROMPT}
        ]
        # Content for OpenAI API call
        model = "gpt-4.1-nano-2025-04-14"

        # Call OpenAI
        start_time = time.perf_counter()
        response = openai_client.beta.chat.completions.parse(
                model=model,
                temperature=1.0,
                # top_p=0.75,
                messages=[
                    {"role": "user", "content": content}
                ],
                response_format=Metadata,
            )
        end_time = time.perf_counter()
        response_time = end_time - start_time # calculate snap latency 
        final_response = response.choices[0].message.parsed.model_dump()

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

    model_name = f"{model}-metadata"

    # store response info
    global responses
    responses.append({
        "model": model_name,
        "response": final_response
    })

    # store usage info
    global usage
    usage.append({
        "model": model_name,
        "tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    })

    # store latency info
    global latency
    latency.append({
        "model": model_name,
        "latency": response_time
    })
    return final_response

def process_image_people(base64_image, video_metadata={}, has_people=True) -> dict:
    if not has_people:
        print("No people")
        return {"people": []}

    with open("prompts/people_prompt.txt", "r", encoding="utf-8") as file:
        USER_PROMPT_PEOPLE = file.read()
    with open("prompts/people_prompt_format.txt", "r", encoding="utf-8") as file:
        USER_PROMPT_FORMAT = file.read()

    PROMPT = f"{USER_PROMPT_PEOPLE.format(video_metadata=video_metadata)}\n{USER_PROMPT_FORMAT}"

    # Content for API call
    content = [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}},
    ]

    # call OpenAI API
    start_time = time.perf_counter()
    response = openai_client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        temperature=1.0,
        # top_p=0.75,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        # max_tokens=500
    )
    end_time = time.perf_counter()
    response_time = end_time - start_time # calculate snap latency 
    final_response = response.choices[0].message.content.encode("utf-8").decode()

    # store response info
    global responses
    responses.append({
        "model": response.model,
        "response": final_response
    })

    # store usage info
    global usage
    usage.append({
        "model": response.model,
        "tokens": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    })

    # store latency info
    global latency
    latency.append({
        "model": response.model,
        "latency": response_time
    })

    if ("i do" in final_response.lower() or
        "i can" in final_response.lower() or
        "identif" in final_response.lower()  or
        "detected" in final_response.lower() or
        "unknown" in final_response.lower()):
        return {"people": []}

    else:
        try:
            final_response = final_response.replace("`", "").replace("json", "")
            famous_people = json.loads(final_response)
            famous_people = {"people": [
                                person for person in famous_people.get("people", [])
                                if isinstance(person, dict) and person.get("name", "")
                                ]
                            }
            return famous_people
        except json.JSONDecodeError as e:
            print(final_response)
            print(f"process_image_people JSONDecodeError: {e}")
            return {"people": []}

def process_image_items(base64_image, famous_people={}, video_metadata={}, scene_attr=None) -> dict:
    with open("is_place.txt", "r", encoding="utf-8") as file:
        places = file.read()
    
    # check if location detection is needed
    is_place = scene_attr is None or (
        isinstance(scene_attr, str) and scene_attr in places
    )
    if not is_place:
        print("No place")
        with open("prompts/items_prompt_no_location.txt", "r", encoding="utf-8") as file:
            USER_PROMPT_ITEMS = file.read()
        with open("prompts/items_prompt_no_location_format.txt", "r", encoding="utf-8") as file:
            USER_PROMPT_FORMAT = file.read()
    else:
        print("Place detected")
        with open("prompts/items_prompt.txt", "r", encoding="utf-8") as file:
            USER_PROMPT_ITEMS = file.read()
        with open("prompts/items_prompt_format.txt", "r", encoding="utf-8") as file:
            USER_PROMPT_FORMAT = file.read()

    if not famous_people or famous_people == {"people": []}:
        people = []
    else: 
        people = [
            {
                "name": p.get("name", ""),
                "description": p.get("description", "")
            }
            for p in famous_people.get("people", [])
        ]
    print(people)
    PROMPT = f"{USER_PROMPT_ITEMS.format(famous_people=people, video_metadata=video_metadata)}\n{USER_PROMPT_FORMAT}" 
    
    # Content for API call
    content = [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}},
    ]

    # call OpenAI API 
    start_time = time.perf_counter()
    response = openai_client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        temperature=1.0,
        # top_p=0.75,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        # max_tokens=1000
    )
    end_time = time.perf_counter()
    response_time = end_time - start_time # calculate snap latency 
    final_response = response.choices[0].message.content.encode("utf-8").decode()

    model_name = f"{response.model}-items"

    # store response info
    global responses
    responses.append({
        "model": model_name,
        "response": final_response
    })

    # store usage info
    global usage
    usage.append({
        "model": model_name,
        "tokens": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    })

    # store latency info
    global latency
    latency.append({
        "model": model_name,
        "latency": response_time
    })

    try:
        cleaned_response = final_response.replace("`", "").replace("json", "")
        cleaned_final_response = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print(final_response)
        print("process_image_items JSONDecodeError: ", e)
        return {"products": [], "background": {}}

    if not is_place:
        cleaned_final_response["background"] = {
            "location": "",
            "description": ""
            }
        print("added background")

    # Response
    return cleaned_final_response

def cleanup_response(response):
    unwanted_phrases_items = ["i do", "i can", "identif", "detected", "unknown", "n/a", "none", "possibly", "likely", "no visib", "not visib"]
    unwanted_phrases_background = ["i do", "i can", "identif", "detected", "unknown", "n/a", "none", "no visib", "not visib"]

    # clean background fields
    background = response.get("background", {})
    location = background.get("location", "")
    description = background.get("description", "")
    if any(phrase in location.lower() for phrase in unwanted_phrases_background):
        background["location"] = ""
        background["description"] = ""
    if any(phrase in description.lower() for phrase in unwanted_phrases_background):
        background["description"] = ""
    if not background.get("location", "").strip():
        background["description"] = ""
    response["background"] = background

    # clean and filter products
    products = response.get("products", [])
    cleaned_products = []
    for product in products:
        item_name = product.get("item_name", "")
        brand = product.get("brand", "")
        if any(phrase in item_name.lower() for phrase in unwanted_phrases_items):
            product["item_name"] = ""
        if any(phrase in brand.lower() for phrase in unwanted_phrases_items):
            product["brand"] = ""
        # Only keep if at least one field is not blank
        if product.get("item_name", "").strip():
            cleaned_products.append(product)
    response["products"] = cleaned_products

    return response

def translate_items(products: list, gl_code: str) -> list:
    with open("prompts/translate_prompt.txt", "r", encoding="utf-8") as file:
        TRANSLATE_PROMPT = file.read()

    PROMPT = f"{TRANSLATE_PROMPT.format(products=products, gl_code=gl_code)}"

    # define output structure
    class Product(BaseModel):
        item_name: str
        brand: str

    class ProductList(BaseModel):
        products: List[Product]


    start_time = time.perf_counter()
    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who translates product names."},
            {"role": "user", "content": PROMPT}
        ],
        response_format=ProductList
    )
    end_time = time.perf_counter()
    response_time = end_time - start_time

    final_response = response.choices[0].message.parsed.model_dump()
    translated_products = [p for p in final_response.get("products", [])]

    model_name = f"{response.model}-translate"

    # store response info
    global responses
    responses.append({
        "model": model_name,
        "response": translated_products
    })

    # store usage info
    global usage
    usage.append({
        "model": model_name,
        "tokens": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    })

    # store latency info
    global latency
    latency.append({
        "model": model_name,
        "latency": response_time
    })

    return translated_products

def choose_best_link_with_dino(product_name, image_search_results, max_candidates=3, score_threshold=0.5):
    """
    For a given product and list of candidate links, fetch each image and verify with DINO.
    Returns the best matching link (or None if no valid match).
    """

    # Limit to top candidates
    candidates = image_search_results[:max_candidates]
    #print(f"[DEBUG] Top {len(candidates)} candidate links for '{product_name}':")
    #for i, l in enumerate(candidates, start=1):
    #    print(f"    {i}. {l}")

    final_link = None
    best_score = 0.0
    best_label = None

    for link in candidates:
        #print(f"[DEBUG] Processing candidate link: {link}")
        img = fetch_main_image_from_page(link)
        if img is None:
            #print(f"[DEBUG] Skipping {link} (no image found)")
            continue

        # Skip tiny images (common for tracking pixels)
        if img.width < 50 or img.height < 50:
            #print(f"[DEBUG] Skipping {link} (image too small: {img.width}x{img.height})")
            continue

        # Sanitize image
        img = sanitize_to_rgb(img)

        # Run DINO detection
        try:
            label, score = detect_label_with_huggingface(img, [product_name])
        except Exception as e:
            #print(f"[WARN] DINO failed for {link}: {e}")
            continue

        if not label:
            #print(f"[DEBUG] No label detected for {link}")
            continue

        print(f"[DEBUG] Detection result for {link}: label={label}, score={score:.3f}")

        # Update best match
        if score >= score_threshold and score > best_score:
            best_score = score
            best_label = label
            final_link = link

    # After all candidates
    if final_link:
        print(f"[DEBUG] Dino matched: label={best_label}, score={best_score}")
        return final_link
    else:
        print(f"[DEBUG] No valid DINO match for '{product_name}'")
        return None
    
# adds product links and info to a list of product dictionaries
def add_product_links(products: list, country: str, max_workers: int = 8) -> list:
    COUNTRY_CONFIG = {
        "tw": {"currency": "TWD", "shop": "momoshop.com.tw"},
        "kr": {"currency": "KRW", "shop": "coupang.com"},
        "default": {"currency": "USD", "shop": "amazon.com"},
    }

    # clean product names, exclude photo field
    cleaned_products = [
    {
        "item_name": format_search_query(product.get("item_name", "")),
        "brand": product.get("brand", "")
    }
    for product in products
    ]

    # only translate if taiwan and korea for now
    translated_products = translate_items(cleaned_products, country) if country in ("tw", "kr") else cleaned_products
    print("Translated products.")
    translated_products = [
        {
            **trans,
            "photo": orig.get("photo", "")
        }
        for orig, trans in zip(products, translated_products)
    ]

    updated_products = []
    def add_product_link(original_product, translated_product):
        brand = original_product.get("brand", "")

        # use translated name for searching
        translated_name = translated_product.get("item_name", "")
        translated_brand = translated_product.get("brand", "")

        full_product_name = f"{translated_brand} {translated_name}".strip() if translated_brand not in translated_name else f"{translated_name}".strip()
        product_crop = original_product.get("photo", "")

        # configuration for country
        config = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG["default"])
        currency = config["currency"]
        shop_website = config["shop"]

        # Initialize future variables
        future_shop = None
        future_regular = None
        future_image_search = None
        future_image = None
        with ThreadPoolExecutor(max_workers=3) as link_executor:
            if full_product_name:
                future_shop = link_executor.submit(
                    get_product_urls, query=full_product_name, search_type="text", shop_website=shop_website, country=country, num_results=10
                )
                future_regular = link_executor.submit(
                    get_product_urls, query=full_product_name, search_type="text", shop_website="", country=country, num_results=10
                )
                if product_crop:
                    future_image_search = link_executor.submit(
                        get_product_urls, image_url=product_crop, search_type="image",  shop_website="", country=country, num_results=10
                    )
                if brand:
                    future_image = link_executor.submit(
                        get_product_images, full_product_name, country=country, num_results=10
                    )

            shop_search_results = future_shop.result() if future_shop else []
            regular_search_results = future_regular.result() if future_regular else []
            image_search_results = future_image_search.result() if future_image_search else []
            product_image_results = future_image.result() if future_image else []

            original_results = list(image_search_results)
            # quick keyword filtering 
            product_name = original_product.get("item_name", "")
            image_search_results = [
                l for l in image_search_results
                if quick_keyword_filter(l)
            ]

            # sort by overlap score (most relevant first)
            image_search_results.sort(
                key=lambda l: overlap_score(l, product_name),
                reverse=True
            )
            candidates = image_search_results[:3]  # limit to 3 first links
            print(f"[INFO] Checking product: {product_name} ({len(candidates)} candidates)")
            for i, link in enumerate(candidates, 1):
                print(f"    {i}. {link}")

            final_link = choose_best_link_with_dino(product_name, image_search_results, max_candidates=3, score_threshold=0.5)
            # if no image-based match, fallback to empty
            image_search_results = [final_link] if final_link else []
            filtered_out = [l for l in original_results if l not in image_search_results]

            print(f"[DEBUG] Final kept link: {image_search_results}")

        # extract shopping links
        links_result = []
        shop_link = ""
        regular_link = ""
        image_search_link = ""

        # get shop-specific shopping link
        shop_search_results = [link for link in shop_search_results if shop_website in link]

        if shop_search_results:
            shop_link = shop_search_results[0]
            links_result.append({
                "currency": currency,
                "link": shop_link,
                "logo": "",
                "price": ""
            })

        # get regular shopping link
        if regular_search_results:
            regular_link = next(
                (link for link in regular_search_results if link != shop_link),
                ""
            )
            if regular_link:
                links_result.append({
                    "currency": currency,
                    "link": regular_link,
                    "logo": "",
                    "price": ""
                })
        
        # get image search shopping link
        if image_search_results:
            image_search_link = next(
                (link for link in image_search_results if link != shop_link and link != regular_link),
                ""
            )
            if image_search_link:
                links_result.append({
                    "currency": currency,
                    "link": image_search_link,
                    "logo": "",
                    "price": ""
                })

        # get product image
        product_image = (
            # product_image_results[0]
            get_valid_image(product_image_results)
            if product_image_results 
            else product_crop
        )

        updated_product = {
            **original_product,
            "shop_links": links_result,
            "photo": product_image,
            "currency": currency,
        }
        return updated_product

    # parallelize product link search
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(add_product_link, original_product, translated_product)
            for original_product, translated_product in zip(products, translated_products)
        ]
        updated_products = [future.result() for future in futures]

    return updated_products

def detect_with_huggingface(image, text_prompts: list):
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

    items_detected = []
    for box, label, score in zip(results["boxes"], results["text_labels"], results["scores"]):
        items_detected.append({
            "item_name": label,
            "confidence": float(score),
            "bounding_box": list(map(int, box.tolist()))
        })

    return items_detected

def get_dino_crops_for_products(base64_image, products, detect_with_huggingface):
    orig_image = base64_to_pil_image(base64_image)
    product_names = [p["item_name"] for p in products]
    detected_items = detect_grouped_with_matching(orig_image, product_names, detect_with_huggingface)
    crops = []
    for prod in products:
        match = next((item for item in detected_items if normalize(item["item_name"]) == normalize(prod["item_name"])), None)
        if match and "bounding_box" in match and match["bounding_box"]:
            x1, y1, x2, y2 = map(int, match["bounding_box"])
            crop = orig_image.crop((x1, y1, x2, y2))
            crops.append(crop)
        else:
            crops.append(None)
    return crops

def save_response_to_file(response, mode, output_path):
    with open(output_path, mode, encoding="utf-8") as file:
        if isinstance(response, dict):
            file.write(json.dumps(response, indent=2))
        else:
            file.write(str(response))

def capture_screen():
    print("Taking snap...")
    subprocess.run(["/usr/bin/adb", "shell", "screencatch", "-m"])
    subprocess.run(["/usr/bin/adb", "pull", "/data/temp/1.bmp", "/home/innovo/righshop-demo-dev/webapp/static/"])

def enhance_image(image: PILImage.Image) -> PILImage.Image:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load RRDBNet model used by RealESRGAN
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path='weights/RealESRGAN_x4plus.pth', # weights path
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,  # True if using GPU with support fp16
        device=device
    )

    img_np = np.array(image.convert("RGB"))  # PIL to NumPy
    output, _ = upsampler.enhance(img_np)
    return PILImage.fromarray(output)

def display_before_after(original, enhanced, i, title=None, save_path="static/output"):
    os.makedirs(save_path, exist_ok=True)
    safe_name = (
        "".join(c if c.isalnum() else "_" for c in title.lower())[:30]
        if title else f"product_{i}"
    )

    filename = f"compare_{i}_{safe_name}.jpg"
    save_path = os.path.join(save_path, filename)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original)
    axs[0].set_title("Before Enhancement")
    axs[0].axis("off")

    axs[1].imshow(enhanced)
    axs[1].set_title("After Enhancement")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved comparison to {save_path}")
    
def process_image_task(base64_image, output_path=None, video_metadata={}, country="us", has_people=True, scene_attr=None) -> dict:
    import os
    from PIL import Image

    global usage, latency, responses
    usage, latency, responses = [], [], []

    print("Processing image...")

    # metadata extraction
    if video_metadata:
        video_id = video_metadata.get("video_id", "")
        title = video_metadata.get("title", "")
        time_position = video_metadata.get("position", 0)
        duration = video_metadata.get("duration", 0)
        description, transcript = get_video_metadata(video_id) if video_id else ("", "")
        people_metadata, items_loc_metadata = "", ""

        if description:
            cleaned = clean_metadata(description, transcript, time_position, duration)
            people_metadata = f"Video Title: {title}\nNames mentioned: {cleaned.get('people', [])}"
            items_loc_metadata = f"Video Title: {title}\nItems and brands mentioned: {cleaned.get('items_and_brands', [])}\nLocations mentioned: {cleaned.get('locations', [])}"
            if transcript:
                summary = f"\nTranscript Summary: {cleaned.get('transcript_summary', '')}"
                people_metadata += summary
                items_loc_metadata += summary
        elif title:
            people_metadata = items_loc_metadata = f"Video Title: {title}"
    else:
        people_metadata = items_loc_metadata = ""

    # people detection
    print(people_metadata)
    famous_people = process_image_people(base64_image, people_metadata, has_people=has_people)
    print(famous_people)
    save_response_to_file(famous_people, "w", output_path)
    save_response_to_file(famous_people, "w", "static/test.txt")

    # items + location detection
    print(items_loc_metadata)
    items_and_loc = process_image_items(base64_image, famous_people, items_loc_metadata, scene_attr=scene_attr)
    save_response_to_file(items_and_loc, "a", "static/test.txt")

    # combine & clean response
    combined_response = {**famous_people, **items_and_loc}
    final_response = cleanup_response(combined_response)
    save_response_to_file(final_response, "w", output_path)

    # add Wikipedia links to people
    start = time.perf_counter()
    final_response["people"] = add_wiki_links(final_response["people"], country_code=country)
    print("wiki", time.perf_counter() - start)

    # crop products using DINO 
    start_dino = time.perf_counter()
    products = [{**p, "product_id": str(uuid.uuid4())} for p in final_response["products"]]
    dino_crops = get_dino_crops_for_products(base64_image, products, detect_with_huggingface)
    output_products = [None] * len(products)

    def process_product_dual(i, product, crop_image):
        if crop_image is None:
            return i, {**product, "photo": "", "photo_original": ""}

        name_short = normalize(product["item_name"])[:30]
        original_path = f"static/output/crop_{i}_{name_short}_original.jpg"
        enhanced_path = f"static/output/crop_{i}_{name_short}_enhanced.jpg"

        # save original locally and upload
        original_io = io.BytesIO()
        crop_image.save(original_io, format='JPEG', quality=95)
        original_bytes = original_io.getvalue()
        crop_image.save(original_path)
        url_original = upload_image_from_str(original_bytes, f"{product['product_id']}_original", collection_name="alpha-dogfood")
        signed_original = generate_signed_url(os.path.basename(url_original)) if url_original else ""
        
        # keep a copy of the original image before enhancement
        original_crop = crop_image.copy()

        # enhance image
        try:
            start_enhance = time.perf_counter()
            crop_image = enhance_image(crop_image)
            end_enhance = time.perf_counter()
            print(f"[Real-ESRGAN] enhancement time for '{product['item_name']}': {end_enhance - start_enhance:.3f} seconds")

            latency.append({
                "model": "Real-ESRGAN",
                "item": product["item_name"],
                "latency": end_enhance - start_enhance
            })
        except Exception as e:
            print(f"Failed to enhance crop for {product['item_name']}: {e}")

        # save enhanced locally and upload
        enhanced_io = io.BytesIO()
        crop_image.save(enhanced_io, format='JPEG', quality=95)
        enhanced_bytes = enhanced_io.getvalue()
        crop_image.save(enhanced_path)
        url_enhanced = upload_image_from_str(enhanced_bytes, product["product_id"], collection_name="alpha-dogfood")
        signed_enhanced = generate_signed_url(os.path.basename(url_enhanced)) if url_enhanced else ""

        # save side-by-side comparison
        display_before_after(original=original_crop, enhanced=Image.open(enhanced_path), i=i, title=product["item_name"])

        return i, {
            **product,
            "photo": signed_enhanced,
            "photo_original": signed_original
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_product_dual, i, p, c) for i, (p, c) in enumerate(zip(products, dino_crops))]
        for future in futures:
            i, prod_with_imgs = future.result()
            output_products[i] = prod_with_imgs

    print("dino: ", time.perf_counter() - start_dino)

    # product links
    products_with_original_links = add_product_links(
        [{**p, "photo": p.get("photo_original", "")} for p in output_products],
        country=country
    )
    products_with_links = add_product_links(output_products, country=country)

    final_products = []
    for enhanced, original in zip(products_with_links, products_with_original_links):
        final_products.append({
            **enhanced,
            "original_links": original.get("shop_links", [])
        })
    final_response["products"] = final_products

    # log latency for search
    search_start = time.perf_counter()
    search_end = time.perf_counter()
    latency.append({
        "model": "google_search_api",
        "latency": search_end - search_start
    })

    save_response_to_file(final_response, "w", output_path)
    print("Processing complete.")

    return {
        "text": final_response,
        "llm_responses": responses,
        "usage": usage,
        "latency": latency
    }


if __name__ == "__main__":
    image_path = "/Users/paulc/righshop-demo-dev/webapp/static/Images/item_10.png"
    #image_path = "/Users/paulc/righshop-demo-dev/webapp/static/Images/item_12_blurred_background.png"
    output_path = "static/output.txt"

    country = "us"
    # country = "sg"
    # country = "tw"
    #country = "kr"

#    video_title = "All 37 Songs of the Eurovision Song Contest 2025 🎵 | #Eurovision2025"
#    channel_name = "Eurovision Song Contest"
#    time_position = 395
#    duration = 1236

    #video_title = "Tout Ce Que Tu Peux Porter, Je Le Paie"
    #channel_name = "MrBeast"
    #time_position = 595
    #duration = 908

    #video_title = "Unboxing ANOTHER $100,000 Sneaker Mystery Box..."
    #channel_name = "Harrison Nevel"
    #time_position = 364
    #duration = 1282

    #video_title = "WARRIORS at LAKERS | FULL GAME HIGHLIGHTS | April 3, 2025"
    #channel_name = "NBA"
    #time_position = 638
    #duration = 732

    #video_title = "#4 NUGGETS at #5 CLIPPERS | FULL GAME 6 HIGHLIGHTS | May 1, 2025"
    #channel_name = "NBA"
    #time_position = 458
    #duration = 684
    
    #video_title = "AJ AUXERRE - PARIS FC (2 - 0) - Résumé - (AJA - PFC) / 2023-2024"
    #channel_name = "Ligue 2 BKT"
    #time_position = 124
    #duration = 240

    #video_title = "BLACKPINK ARRIVAL AT ICN AIRPORT FLYING TO US FOR VMAS"
    #channel_name = "Fayer Flutter"
    #time_position = 26
    #duration = 48

    #video_title = "Zach LaVine wins 2015 & 2016 NBA Slam Dunk Contests | NBA All-Star Highlights"
    #channel_name = "ESPN"
    #time_position = 207
    #duration = 232

    #video_title = "Stray Kids's Felix Gets Ready for the Louis Vuitton Show in Barcelona | Last Looks | Vogue"
    #channel_name = "Vogue"
    #time_position = 45
    #duration = 322

    #video_title = "Highlights Sinner vs Alcaraz Final | Roland-Garros 2025"
    #channel_name = "Roland-Garros"
    #time_position = 702
    #duration = 725

    #video_title = "Le titre surprise de Niska et DJ Snake pour la victoire du PSG"
    #channel_name = "M6 Info"
    #time_position = 49
    #duration = 166

    video_title = "State-Of-The-Art Prompting For AI Agents"
    channel_name = "Y Combinator"
    time_position = 864
    duration = 1885

    #video_title = "Whose Outfit Is The Most Expensive? | Assumptions vs Actual"
    #channel_name = "Jubilee"
    #time_position = 503
    #duration = 519

    #video_title = "older sibling swipes 20 guys for sister | versus 1"
    #channel_name = "nectar"
    #time_position = 248
    #duration = 695

    #video_title = "personal trainer builds dream swole mate | build-a-boo"
    #channel_name = "nectar"
    #time_position = 172
    #duration = 1310

    #video_title = "MHD - AFRO TRAP Part.3 (Champions League)"
    #channel_name = "Mhd Officiel"
    #time_position = 66
    #duration = 153

    #video_title = "Investigating the 21/yo AI CEO kicked out of Columbia"
    #channel_name = "Daniel Mints"
    #time_position = 578
    #duration = 1219

    #video_title = "I Secretly Hid In MrBeast's YouTube Videos"
    #channel_name = "Airrack"
    #time_position = 494
    #duration = 1964


    video_url = get_video_url(video_title, channel_name, website="youtube.com")     
    video_id = extract_video_id(video_url) if video_url else ""
    is_metadata = metadata_exists(video_id)
    if video_id and not is_metadata:
        description, transcript = scrape_video_metadata(video_url)
        upload_video_metadata(video_id, video_title, channel_name, duration, description, transcript)

    video_metadata = {
                    "video_id": video_id,
                    "title": video_title,
                    "position": time_position,
                    "duration": duration
                }
    base64_image = encode_image(image_path)
    snap_id = str(uuid.uuid4())
    user_id = "jchao"

    response = process_image_task(base64_image, output_path, video_metadata, country)

    #print(json.dumps(response["text"], indent=2))
    #print(json.dumps(response["latency"], indent=2))
    products = response["text"].get("products", [])

    for product in products:
        print(f"\nProduct: {product.get('item_name', '')}")
        print(f"Photo: {product.get('photo', '')}")
        
        print("Shop Links (Original):")
        for link in product.get("original_links", []):
            print(f"- {link.get('link', '')}")     
        
        print("Shop Links (Enhanced):")
        for link in product.get("shop_links", []):
            print(f"- {link.get('link', '')}")

        