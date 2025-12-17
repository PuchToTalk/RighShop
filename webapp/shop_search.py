import http.client
import requests
import json
import time
import io
import os
import base64
from rapidfuzz import fuzz
import unicodedata
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import random
import requests
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, Any
from urllib.parse import urljoin, urlparse
import re
import difflib

try:
    import cloudscraper
    _HAS_CLOUDSCRAPER = True
    print("Cloudscraper library found. Will use for Cloudflare-protected sites.")
except ImportError:
    _HAS_CLOUDSCRAPER = False
    print("Warning: cloudscraper library not found. Install with 'pip install cloudscraper' for better results.")

load_dotenv()

# Configuration for OpenAI API
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY in your .env file.")

client = OpenAI(api_key=api_key)

headers = {
    'X-API-KEY': '108d0d6e1320908677453002481b0ab25f4c10a7',
    'Content-Type': 'application/json'
    }

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
]

BROWSER_HEADERS = { 
    "User-Agent": random.choice(USER_AGENTS),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-User": "?1",
    "Sec-Fetch-Dest": "document",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
}

GENERAL_OOS_PHRASES = [
    # English
    "out of stock", "sold out", "unavailable", "discontinued",
    "not available", "backordered", "no longer available", "SOLD OUT",
    # Korean
    "품절",
    "판매가 중지",
    "판매중지",
    "더 이상 판매",
    "죄송합니다. 판매가 중지된 상품",
    "죄송합니다. 요청하신 상품을 찾을 수 없습니다",
    "상품이 존재하지 않습니다",
    "잘못된 상품번호",
    "현재 판매중인 상품이 아닙니다.",
]

NON_SHOPPING_DOMAINS = [
    'reddit.com', 'quora.com', 'pinterest.com', 'twitter.com',
    'facebook.com', 'bbc.com', 'cnn.com', 'wikipedia.org',
    'forbes.com', 'nytimes.com', 'washingtonpost.com', 'buzzfeed.com',
    'youtube.com', 'linkedin.com', 'instagram.com', 'tiktok.com',
    'blog.naver', 'post.naver', 'github.com', 'news.', '/news',
    '.wiki', '.pdf', '/article', '/photos', '/feed', '.txt', '/view', '.json',
    'cisco.com', 'tumgik.com', 'x.com', 'hubermanlab.com', 'rumble.com',
    '.ai', 'basketball.eurobasket.com', 'spotify.com', 'brainly.com',
    'poemhunter.com', 'patch.com', 'mwcconnection.com', 'windycitytimes.com',
    'fiver.com', 'danharris.com', 'crunchyroll.com', 'eventeny.com', 'deviantart.com'
]

NON_SHOPPING_DOMAINS += [
    '.pdf', '.ppt', '.xls', '.zip',
    'arca.live', 'personality-database.com',
    'kipo.go.kr', '/download', '/attachment',
]

STOPWORDS = [
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'for', 'to', 
    'of', 'with', 'is', 'are', 'by', 'this', 'that', 'these', 'those', 
    'be', 'was', 'were', 'has', 'have', 'had', 'been', 'do', 'does', 
    'did', 'so', 'if', 'not', 'then', 'as', 'out', 'about', 'up', 
    'down', 'over', 'under', 'again', 'here', 'there', 'when', 'where', 
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
    'other', 'some', 'such', 'only', 'own', 'same', 'than', 'too', 
    'very', 'can', 'will', 'just'
]

URL_KEYWORDS_TO_EXCLUDE = ['blog', 'article', 'post', 'news', 'ai.', 'podcast', 'interview', 'coach', 'forum', 'discussion', 'wiki', 'wikipedia', 'video', '.ai', 'Muslim', 'Ramadan', 'Islamic']

ALLOWED_FORMATS = {"JPEG", "PNG", "GIF", "WEBP", "BMP", "WBMP", "HEIF", "AVIF"}

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp",
    "image/vnd.wap.wbmp", "image/heif", "image/avif"
    # "image/x-adobe-dng", "image/x-icon",
}

def get_text_from_url(url, session):
    try:
        response = session.get(url, headers=BROWSER_HEADERS, timeout=15, allow_redirects=True)
        
        if response.status_code == 403 and _HAS_CLOUDSCRAPER:
            print(f"Standard request failed with 403. Retrying with cloudscraper for {url}...")
            scraper = cloudscraper.create_scraper()
            response = scraper.get(url, timeout=15)

        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)

        # Limit text size to avoid sending huge documents to the API (e.g., first 15000 chars).
        return clean_text[:15000]

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing URL {url}: {e}")
        return None
    
def extract_price_from_text(text_content):
    if not text_content:
        return "No text content was provided to analyze."

    prompt = f"""
    From the product page text provided below, please extract only the current price.
    - Look for the most prominent price for the item.
    - Ignore any prices that are struck through, "list prices," or "MSRP."
    - Return only the numerical value (e.g., 35000 or 299.99), with any currency symbols (like $, £, ¥, 원), commas, text, or explanations.

    --- TEXT ---
    {text_content}
    --- END TEXT ---

    Price:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system", "content": "You are an expert data extractor specializing in finding product prices on e-commerce websites. You return only the final numerical price."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=20
        )
        extracted_price = response.choices[0].message.content.strip()
        return extracted_price

    except Exception as e:
        return f"API call failed: {e}"

def extract_availability_from_text(text_content: str) -> str:
    if not text_content:
        return "UNKNOWN"

    prompt = f"""
    You are an e‑commerce availability detector.
    From the product‑page text below, answer with **only** one word:

    IN_STOCK       → the main item can be purchased now
    OUT_OF_STOCK   → the main item is not purchasable (sold out, unavailable) 
    UNKNOWN        → you cannot decide

    Do not explain yourself; output just the keyword.

    --- TEXT START ---
    {text_content}
    --- TEXT END ---
    Answer:
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"[LLM] Availability extraction failed: {e}")
        return "UNKNOWN"

def get_product_name(url: str, sess: requests.Session, timeout: int = 15) -> str:
    try:
        resp = sess.get(url, headers=BROWSER_HEADERS, timeout=timeout, allow_redirects=True)
        if resp.status_code == 403 and _HAS_CLOUDSCRAPER:
            scraper = cloudscraper.create_scraper()
            resp = scraper.get(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "html.parser")

        # 1. OpenGraph
        tag = soup.find("meta", property="og:title")
        if tag and tag.get("content"):
            return tag["content"].strip()[:120]

        # 2. meta name="title"
        tag = soup.find("meta", attrs={"name": "title"})
        if tag and tag.get("content"):
            return tag["content"].strip()[:120]

        # 3. first <h1>
        tag = soup.find("h1")
        if tag and tag.get_text(strip=True):
            return tag.get_text(" ", strip=True)[:120]

        # 4. <title> tag
        tag = soup.find("title")
        if tag and tag.get_text(strip=True):
            return tag.get_text(" ", strip=True)[:120]

        return ""        # nothing found
    except Exception as e:
        print(f"[NAME] extraction failed for {url}: {e}")
        return ""

def get_product_name_using_LLM(
    url: str,
    sess: requests.Session,
    max_chars: int = 4000,
) -> str:

    page_text = get_text_from_url(url, session=sess)
    if not page_text:
        return ""
    page_text = page_text[:max_chars]

    prompt = f"""
    You are an e‑commerce title extractor.
    From the product‑page text below, output ONLY the item’s main name /
    title exactly as written on the site. Keep the original language
    (English, Korean, Japanese, etc.). Do not add quotes, extra words,
    or explanations—just the name.

    --- TEXT START ---
    {page_text}
    --- TEXT END ---
    Name:
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[
                {"role": "system",
                 "content": "You output only the product title—no other text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=40,
        )
        name = resp.choices[0].message.content.strip()
        return name[:120]          # safety truncate
    except Exception as e:
        print(f"[LLM‑NAME] extraction failed for {url}: {e}")
        return ""

def extract_product_info(text_content: str) -> tuple[str, str, str]:
    if not text_content or len(text_content.strip()) < 100:
        print("[LLM Extractor] Failed: Input text is empty or too short.")
        return ("--failed--", "UNKNOWN", "")
    
    blocker_keywords = ["verify you are a human", "access denied", "are you a robot", "captcha"]
    if any(keyword in text_content.lower() for keyword in blocker_keywords):
        print("[LLM Extractor] Failed: Input text appears to be a block/CAPTCHA page.")
        return ("--failed--", "UNKNOWN", "")

    user_prompt = f"""
    You are an expert e-commerce data extractor.
    From the product page text below, you must reply ONLY with a single, valid JSON object.
    The JSON object must have these three keys:

      "price"        : The most prominent current price. Extract numbers only (e.g., 29.99 or 35000). If no price is found, the value should be null.
      "availability" : The stock status. The value must be one of these three strings: "IN_STOCK", "OUT_OF_STOCK", or "UNKNOWN".
      "name"         : The concise product title in its original language.

    Do not add any text, explanations, or markdown before or after the JSON object.

    --- TEXT START ---
    {text_content}
    --- TEXT END ---
    """

    raw_llm_output = None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a JSON-only API that extracts product data."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"}
        )

        raw_llm_output = resp.choices[0].message.content
        data = json.loads(raw_llm_output)

        price_value = data.get("price")
        price = str(price_value).strip() if price_value is not None else ""

        availability_value = str(data.get("availability", "UNKNOWN")).upper()
        allowed_stocks = {"IN_STOCK", "OUT_OF_STOCK"}
        availability = availability_value if availability_value in allowed_stocks else "UNKNOWN"

        name = str(data.get("name", "")).strip()[:120]

        if not price or not name:
            print(f"[LLM Extractor] Failed: Could not extract price or name. Got: price='{price}', name='{name}'")
            return ("--failed--", "UNKNOWN", "")

        return (price, availability, name)

    except json.JSONDecodeError:
        print(f"[LLM Extractor] CRITICAL: Failed to decode JSON.")
        print(f"--- LLM Raw Output Start ---\n{raw_llm_output}\n--- LLM Raw Output End ---")
        return ("--failed--", "UNKNOWN", "")
    
    except Exception as e:
        print(f"[LLM Extractor] An unexpected error occurred: {e}")
        return ("--failed--", "UNKNOWN", "")


def extract_product_info_with_snippet(text_content: str) -> tuple[str, str, str]:
    """
    Extracts product info using the two-step "Golden Snippet" technique.
    This version uses language-agnostic prompts for global compatibility.
    """
    if not text_content or len(text_content.strip()) < 100:
        print("[SNIPPET EXTRACTOR] Failed: Input text is empty or too short.")
        return ("--failed--", "UNKNOWN", "")

    #print("\n[FINDER] Asking LLM to find the core product snippet from the page...")
    finder_prompt = f"""
    From the following raw text of a webpage, which could be in any language (e.g., English, Korean), find and extract ONLY the main content block for the PRIMARY product offer.

    This block should contain the product's name, its price, and its main availability status.

    IMPORTANT INSTRUCTIONS:
    1. Look for the main action button for purchasing (like 'Add to Cart' or 'Buy Now') OR a clear status message if it cannot be bought (like 'Out of Stock'). These phrases will be in the page's original language.
    2. IGNORE secondary offers, such as those from third-party sellers or for 'used' items, and focus only on the main offer from the website itself.

    --- RAW TEXT START ---
    {text_content[:18000]}
    --- RAW TEXT END ---
    """
    
    try:
        finder_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a text extractor that isolates a page's main product content, regardless of the language."},
                {"role": "user", "content": finder_prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        golden_snippet = finder_resp.choices[0].message.content

        if not golden_snippet or len(golden_snippet.strip()) < 10:
            print("[FINDER] Failed: The model did not return a useful snippet.")
            return ("--failed--", "UNKNOWN", "")
        
        #print(f"[FINDER] Success! Got a snippet of {len(golden_snippet)} characters.")

        #print("[EXTRACTOR] Asking LLM to extract structured JSON from the clean snippet...")
        extractor_prompt = f"""
        From the clean product snippet below (which may be in any language), provide a JSON object with three keys: "price", "availability", and "name".

        - "price": The current price, as numbers only. If unavailable, this can be null.
        - "availability": Must be "IN_STOCK", "OUT_OF_STOCK", or "UNKNOWN".
            - The item is "IN_STOCK" if the text shows a primary way to purchase it now (like an 'Add to Cart' or 'Buy Now' button).
            - The item is "OUT_OF_STOCK" if the text explicitly states it's unavailable/sold out, or if the only buying options are from secondary/used sellers.
        - "name": The concise product title in its original language.
        
        Output only the raw JSON.

        --- SNIPPET START ---
        {golden_snippet}
        --- SNIPPET END ---
        """

        extractor_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a JSON-only API that extracts product data from clean text, with special rules for availability."},
                {"role": "user", "content": extractor_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(extractor_resp.choices[0].message.content)

        price = str(data.get("price") or "").strip()
        availability_raw = str(data.get("availability", "UNKNOWN")).upper()
        availability = availability_raw if availability_raw in ["IN_STOCK", "OUT_OF_STOCK"] else "UNKNOWN"
        name = str(data.get("name", "")).strip()[:120]

        if not name:
             print(f"[EXTRACTOR] Failed: Could not extract name from the snippet.")
             return ("--failed--", "UNKNOWN", "")
        
        #print("[EXTRACTOR] Success! Parsed all product info.")
        return (price, availability, name)

    except Exception as e:
        print(f"[SNIPPET EXTRACTOR] An unexpected error occurred: {e}")
        return ("--failed--", "UNKNOWN", "")

# DEMO: Geo-locking temporarily disabled for demonstration.
# GEO_LOCKED = {"coupang.com"}

GEO_LOCKED: set[str] = set()

PRICE_REGEX = re.compile(r'(?:[$€£¥₩]\s?[0-9][0-9,.]*)')

def _needs_proxy(domain: str) -> bool:
    return any(d in domain for d in GEO_LOCKED)

def validate_generic_link(
    url: str,
    sess: requests.Session,
    proxy_dict: Optional[dict] = None,
    timeout: int = 15,
) -> Optional[tuple[str, str, str, str]]:
    if not url.startswith(("https://", "http://")):
        return None

    dom = urlparse(url).netloc.lower()
    if any(nd in dom or nd in url for nd in NON_SHOPPING_DOMAINS):
        print(f"[FILTERED] Non‑shopping domain -> {url}")
        return None

    sess.headers.update(BROWSER_HEADERS)
    proxies = proxy_dict if proxy_dict and _needs_proxy(dom) else None

    try:
        resp = sess.get(url, timeout=timeout, allow_redirects=True, proxies=proxies)
        if resp.status_code in (301, 302, 303, 307, 308):
            target = resp.headers.get("Location", resp.url)
            resp = sess.get(target, timeout=timeout, allow_redirects=True, proxies=proxies)

        if resp.status_code == 403 and _HAS_CLOUDSCRAPER:
            scraper = cloudscraper.create_scraper(browser={"custom": "Scraper 1.3"})
            resp = scraper.get(url, timeout=timeout, allow_redirects=True)

        if not (200 <= resp.status_code < 400):
            print(f"[FILTERED] Status {resp.status_code} -> {url}")
            return None

        if len(resp.content) < 4000:
            print(f"[FILTERED] Page too small (~{len(resp.content)} B) -> {url}")
            return None

        html_snippet = resp.text[:4000]
        price, availability, name = extract_product_info(html_snippet)

        if availability == "OUT_OF_STOCK":
            print(f"[FILTERED] LLM says OUT_OF_STOCK -> {url}")
            return None

        return (resp.url, price, availability, name)

    except requests.ReadTimeout:
        print(f"[DEMO MODE] Ignoring ReadTimeout -> {url}")
        return (url, "", "UNKNOWN", "")

    except requests.RequestException as exc:
        print(f"[FILTERED] {exc.__class__.__name__} -> {url}")
        return None


def get_validated_links(urls: list[str], session: requests.Session | None = None) -> list[str]:
    print("the link is getting validating")
    validated = []
    # use caller-supplied session or make a new one
    sess = session or requests.Session()
    with ThreadPoolExecutor(max_workers=10) as ex:
        fut = {ex.submit(validate_generic_link, u, sess): u for u in urls}
        for f in as_completed(fut):
            if f.result():
                validated.append(f.result())
    return validated

# ----------- above : Minh's code -----------
# TODO: calculate the price and latency after running extract_product_info_with_snippet
# TODO: try out with different model GPT5, 4mini, 4o, 4nano / Mistral
# TODO: price: convert it to $




def format_search_query(query):
    """
    Format the search query by removing stopwords and quoting important phrases.
    """
    words = query.split()
    important_words = [word for word in words if word.lower() not in STOPWORDS]
    formatted_query = ' '.join([f'"{word}"' if ' ' in word else word for word in important_words])

    return formatted_query

def filter_shopping_links(links):
    """
    Filter out non-shopping domains from the search results.
    """
    return [
        link for link in links
        if not any(domain in link.lower() for domain in NON_SHOPPING_DOMAINS)
        and not any(keyword in link.lower() for keyword in URL_KEYWORDS_TO_EXCLUDE)
        ]

def check_image(link):
    # Convert http to https
    if link.startswith("http://"):
        https_link = "https://" + link[len("http://"):]
    elif link.startswith("https://"):
        https_link = link
    else:
        print(f"skip non-http link: {link}")
        return None
    try:
        resp = requests.get(https_link, stream=True, timeout=1)
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get('Content-Type', '').lower()
        if content_type not in ALLOWED_CONTENT_TYPES:
            return None
        try:
            img = Image.open(BytesIO(resp.content))
            if img.format and img.format.upper() in ALLOWED_FORMATS:
                return https_link
            else:
                return None
        except Exception as e:
            print(f"failed to open image {https_link}: {e}")
            return None
    except requests.exceptions.ReadTimeout:
        return None
    except Exception as e:
        print(f"request failed for {https_link}: {e}")
        return None

def get_valid_image(product_image_results, max_workers=5):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_link = {executor.submit(check_image, link): link for link in product_image_results}
        for future in as_completed(future_to_link):
            result = future.result()
            if result:
                return result
    print("no valid image found")
    return ""

def get_product_urls(
        query="",
        shop_website="",
        country="us",
        num_results=10,
        search_type="text",
        image_url=""
    ) -> list:
    """
    Retrieve product URLs by text or image search.
    
    Args:
        query (str): Required if search_type="text". The search query.
        shop_website (str): Optional. Restrict search to a site.
        country (str): Optional. Default "us".
        num_results (int): Optional. Number of results.
        search_type (str): "text" or "image".
        image_url (str): Required if search_type="image". URL of image.
    
    Returns:
        list: Product URLs.
    
    Usage:
        get_product_urls(query="iPhone 15")         # text search
        get_product_urls(search_type="image", image_url="...")  # image search
    """
    connection = http.client.HTTPSConnection("google.serper.dev")
    if search_type == "image":
        if not image_url:
            raise ValueError("image_url must be provided for image search")
        payload = json.dumps({
            "url": image_url,
            "gl": country,
            # "num": num_results,
        })
        endpoint = "/lens"
    else:
        q = f"{query}:{shop_website}" if shop_website else query
        payload = json.dumps({
            "q": q,
            "gl": country,
            "num": num_results,
            "autocorrect": False
        })
        endpoint = "/search"

    for attempt in range(2):
        try:
            connection.request("POST", endpoint, payload, headers)
            res = connection.getresponse()
            data = json.loads(res.read().decode("utf-8"))


            if search_type == "image":
                results = data.get("organic", [])
            else:
                results = data.get("organic", [])

            links = [result.get("link", "") for result in results]
            filtered_links = filter_shopping_links(links)
            product_urls = filtered_links if filtered_links else []
            return product_urls

        except Exception as e:
            print(f"get_product_urls Error, Query: {query}, Attempt: {attempt + 1}, Error: {e}")
            if attempt == 0:
                time.sleep(0.1)
            else:
                return []
        
def get_product_images(query, country="us", num_results=10) -> list:
    connection = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
    "q": query,
    "gl": country,
    # "location": location,
    # "hl": language,
    "num": num_results,
    "autocorrect": False
    })

    for attempt in range(2):
        try:
            connection.request("POST", "/images", payload, headers)
            res = connection.getresponse()
            data = json.loads(res.read().decode("utf-8"))
            images = data.get("images", [])
            image_urls = [img.get("imageUrl", "") for img in images]
            return image_urls
    
        except Exception as e:
            print(f"get_product_image Error, Query: {query}, Attempt: {attempt + 1}, Error: {e}")
            if attempt == 0:
                time.sleep(0.1)
            else:
                return []
            
def quick_keyword_filter(link: str) -> bool:
    """Return False if the link obviously refers to a non-fashion category."""
    blacklist = [
        "table", "chair", "sofa", "lamp", "bed", "couch",
        "garden", "kitchen", "tools", "hardware", "phone", "tv", "electronics",
        "car", "bicycle", "guitar", "drum", "knife"
    ]
    low = link.lower()
    return not any(word in low for word in blacklist)

def overlap_score(link: str, product_name: str) -> float:
    """Compute a simple word overlap score between the URL and the product name."""
    words_link = set(re.findall(r"[a-z]+", link.lower()))
    words_product = set(re.findall(r"[a-z]+", product_name.lower()))
    if not words_link or not words_product:
        return 0
    return len(words_link & words_product) / len(words_product)

def is_link_relevant_with_llm(link: str, product_name: str, client) -> tuple[bool, int]:
    """
    Uses an LLM to decide if a link is relevant to a given product name.
    Returns (is_relevant, tokens_used).
    """
    strict_prompt = f"""
        You are a strict e-commerce link relevance checker.

        Check ONLY the text content of this shopping link (do NOT assume what's on the page).
        Determine if it clearly refers to a product that matches the description below.

        Product name: "{product_name}"
        Link: {link}

        Rules:
        - Answer "YES" only if the link text explicitly suggests the same type of item
        (e.g., clothing vs. furniture).
        - If the link points to something unrelated (like furniture when the product is clothing),
        answer "NO".
        - If unsure, always answer "NO".

        Respond with exactly one word: YES or NO.
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "You are a strict classifier that only responds with YES or NO."},
                {"role": "user", "content": strict_prompt}
            ],
            max_tokens=5,
            temperature=0
        )

        answer = resp.choices[0].message.content.strip().lower()
        tokens_used = resp.usage.total_tokens if hasattr(resp, "usage") else 0

        return (answer.startswith("yes"), tokens_used)

    except Exception as e:
        print(f"[WARN] LLM relevance check failed for {link}: {e}")
        return (False, 0)
    
def fetch_main_image_from_page(link: str) -> Image.Image | None:
    """
    Try to extract the main product image from a shopping page.
    Always returns a clean RGB PIL image.
    """
    try:
        scraper = cloudscraper.create_scraper()  # FIX: create local scraper
        resp = scraper.get(link, timeout=8)
        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        # Prefer Open Graph image
        og_img = soup.find("meta", property="og:image")
        if og_img and og_img.get("content"):
            img_url = urljoin(link, og_img["content"])
        else:
            # fallback: first large img tag
            img_tag = soup.find("img")
            if not img_tag or not img_tag.get("src"):
                return None
            img_url = urljoin(link, img_tag["src"])

        #print(f"[DEBUG] Downloading image: {img_url}")
        img_resp = scraper.get(img_url, timeout=8, stream=True)
        img_bytes = img_resp.content

        # Load image
        img = Image.open(io.BytesIO(img_bytes))

        # Normalize to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Fix cases where libraries produce weird shapes
        import numpy as np
        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == 1 and arr.shape[2] == 3:
            arr = arr.reshape((1, 1, 3))
            img = Image.fromarray(arr.astype("uint8")).convert("RGB")

        return img

    except Exception as e:
        print(f"[WARN] Could not fetch image for {link}: {e}")
        return None


def verify_image_with_llm(image: Image.Image, product_name: str, client) -> bool:
    """
    Use a multimodal LLM to check if the image matches the product description.
    """
    print(f"[DEBUG] Verifying image relevance for: '{product_name}'")

    # Convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    data_uri = f"data:image/png;base64,{img_base64}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "You are an expert product visual inspector."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Does this image clearly match this product description? '{product_name}'. Answer YES or NO."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri}
                        }
                    ]
                }
            ],
            max_tokens=5,
            temperature=0
        )
        answer = resp.choices[0].message.content.strip().lower()
        print(f"[DEBUG] LLM response for image relevance: {answer}")
        return answer.startswith("yes")
    except Exception as e:
        print(f"[WARN] Vision LLM check failed: {e}")
        return False
    
def extract_valid_product_info(search_results, verbose=False):
    valid, failed = [], 0
    if verbose:
        print(f"[extract] inputs: {len(search_results)}")

    with requests.Session() as sess:
        for r in search_results:
            if not isinstance(r, dict): 
                failed += 1; continue
            url = r.get("url"); 
            if not url: 
                failed += 1; continue

            try:
                text = get_text_from_url(url, sess)
                if not text:
                    failed += 1; 
                    if verbose: print(f"[extract] no content: {url}")
                    continue

                price, availability, name = extract_product_info_with_snippet(text)

                # Accept if we have at least a name; price may be blank
                if name and name.strip():
                    r.update({
                        "price": price,                   # may be ""
                        "availability": availability,     # IN_STOCK/OUT_OF_STOCK/UNKNOWN
                        "product_name": name.strip()
                    })
                    if verbose:
                        print(f"[extract] ok: {url} | name='{name}' | price='{price}' | {availability}")
                    valid.append(r)
                else:
                    failed += 1
                    if verbose:
                        print(f"[extract] missing name: {url}")

            except Exception as e:
                failed += 1
                if verbose:
                    print(f"[extract] error {url}: {e}")

    if verbose:
        print(f"[extract] done: valid={len(valid)} failed={failed}")
    return valid

def compute_similarity(a, b):
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100.0

def filter_by_similarity(search_results, original_product_name, threshold=60, verbose=False):
    accepted, rejected = [], []
    for r in search_results:
        name = r.get("product_name") or r.get("title") or url_basename(r.get("url",""))
        score = compute_similarity(original_product_name, name)
        r["similarity_score"] = score
        (accepted if score >= threshold else rejected).append(r)
        if verbose:
            print(f"[sim] {score:.1f}%  {name!r}  <-vs->  {original_product_name!r}")

    if verbose:
        print("\n--- Similarity Filtering Summary ---")
        print(f"Total candidates: {len(search_results)}")
        print(f"✅ Accepted: {len(accepted)}")
        print(f"❌ Rejected: {len(rejected)}")
        if rejected:
            print("\nRejected (score<th):")
            for r in rejected[:15]:  # cap
                print(f" - {r.get('url')} ({r.get('similarity_score',0):.1f}%)")
    return accepted


def filter_by_similarity_rapidfuzz(search_results, original_product_name, threshold=60, verbose=False):
    accepted, rejected = [], []
    for r in search_results:
        candidate_name = r.get("product_name") or r.get("title") or r.get("url", "")
        
        # You can try different scoring functions
        score = fuzz.token_set_ratio(original_product_name, candidate_name)
        r["similarity_score"] = score

        if score >= threshold:
            accepted.append(r)
        else:
            rejected.append(r)
        
        if verbose:
            print(f"[sim] {score:.1f}%  {candidate_name!r} <-vs-> {original_product_name!r}")
    
    if verbose:
        print("\n--- Similarity Filtering Summary ---")
        print(f"Total candidates: {len(search_results)}")
        print(f"✅ Accepted: {len(accepted)}")
        print(f"❌ Rejected: {len(rejected)}")
        if rejected:
            print("\nRejected:")
            for r in rejected:
                print(f" - {r.get('url')} ({r['similarity_score']:.1f}%)")

    return accepted

def url_basename(u: str) -> str:
    try:
        path = urlparse(u).path or ""
        base = path.rstrip("/").split("/")[-1]
        # de-slug a bit
        base = re.sub(r"[-_]+", " ", base)
        return base
    except Exception:
        return ""

def wrap_links_as_dicts(items, default_source="image_search"):
    wrapped = []
    for it in items or []:
        if isinstance(it, dict):
            # already structured
            u = it.get("url") or it.get("link")
            if not u: 
                continue
            wrapped.append({
                "url": u,
                "source": it.get("source", default_source),
                "title": it.get("title") or it.get("name") or url_basename(u)
            })
        elif isinstance(it, str):
            u = it.strip()
            if not u:
                continue
            wrapped.append({
                "url": u,
                "source": default_source,
                "title": url_basename(u)
            })
    return wrapped


# --- Helpers ---------------------------------------------------------------

CATEGORY_HINTS = {
    "shirt":  {"shirt","tee","t-shirt","tshirt","button","button-up","buttondown","button-down","oxford","top","blouse","henley","polo"},
    "shorts": {"short","shorts","running","athletic","bike","biker"},
    "sneakers":{"sneaker","sneakers","shoe","shoes","trainer","trainers","running","low","high","chuck","ultraboost","air","jordan","adidas","nike","converse"},
    "chair":  {"chair","eames","eiffel","dining","side","stool"},
    "bag":    {"bag","tote","handbag","crossbody","backpack","purse","satchel"},
    "dress":  {"dress","gown","maxi","midi","mini"},
    # add more as needed…
}

STOPWORDS = {
    "the","a","an","and","or","of","with","for","men","mens","women","womens","unisex",
    "adult","kids","boy","girl","size","sizes","pack","set","new","brand","authentic","original"
}

def url_basename(u: str) -> str:
    try:
        path = urlparse(u).path
        base = path.rsplit("/", 1)[-1]
        base = base.replace("-", " ").replace("_", " ")
        return base
    except Exception:
        return u or ""

def norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip()

def tokenize(s: str) -> set[str]:
    s = norm(s).lower()
    words = set(re.findall(r"[a-z0-9\-]+", s))
    return {w for w in words if w not in STOPWORDS and len(w) > 2}

def infer_category(name: str) -> set[str]:
    t = norm(name).lower()
    for _, hints in CATEGORY_HINTS.items():
        if any(h in t for h in hints):
            return hints
    return set()

def gate_candidate(original_name: str, candidate_name: str, require_shared_tokens: int = 1) -> bool:
    # Category guard (only if we can infer a family from the original)
    hints = infer_category(original_name)
    if hints:
        if not any(h in norm(candidate_name).lower() for h in hints):
            return False
    # Token overlap guard
    base_tokens = tokenize(original_name)
    cand_tokens = tokenize(candidate_name)
    if len(base_tokens.intersection(cand_tokens)) < require_shared_tokens:
        return False
    return True

def composite_score(a: str, b: str) -> float:
    # Blend a few robust, word-aware similarities
    a = norm(a); b = norm(b)
    s1 = fuzz.token_set_ratio(a, b)
    s2 = fuzz.token_sort_ratio(a, b)
    s3 = fuzz.partial_ratio(a, b)
    return 0.5*s1 + 0.3*s2 + 0.2*s3

def brand_boost(base_score: float, original_name: str, candidate_name: str) -> float:
    # Tiny bump when the same brand token is present (expand this list for your data)
    brands = {"adidas","nike","converse","reebok","puma","new balance","levi","h&m","zara","uniqlo","patagonia","north face"}
    o = tokenize(original_name)
    c = tokenize(candidate_name)
    if brands.intersection(o) and brands.intersection(c) and brands.intersection(o) == brands.intersection(c):
        return min(100.0, base_score + 5.0)  # +5 cap
    return base_score

# --- Main filter -----------------------------------------------------------

def filter_by_similarity_rapidfuzz2(
    search_results,
    original_product_name: str,
    threshold: int = 60,
    require_shared_tokens: int = 1,
    verbose: bool = False,
    fallback_to_url_basename: bool = True,
):
    accepted, rejected = [], []
    original_product_name = norm(original_product_name)

    for r in search_results:
        cand = (
            r.get("product_name")
            or r.get("title")
            or (url_basename(r.get("url","")) if fallback_to_url_basename else "")
        )
        cand = norm(cand)

        # Gate out obvious mismatches before scoring
        if not gate_candidate(original_product_name, cand, require_shared_tokens=require_shared_tokens):
            r["similarity_score"] = 0.0
            rejected.append(r)
            if verbose:
                print(f"[sim] GATED 0.0%  {cand!r}  <-vs->  {original_product_name!r}")
            continue

        # Composite word-aware score + small brand boost
        score = composite_score(original_product_name, cand)
        score = brand_boost(score, original_product_name, cand)
        r["similarity_score"] = score

        (accepted if score >= threshold else rejected).append(r)
        if verbose:
            tag = "ACCEPT" if score >= threshold else "reject"
            print(f"[sim] {score:.1f}%  {cand!r}  <-vs->  {original_product_name!r}  [{tag}]")

    # Sort accepted by score desc so best candidates come first
    accepted.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

    if verbose:
        print("\n--- Similarity Filtering Summary ---")
        print(f"Total candidates: {len(search_results)}")
        print(f"✅ Accepted: {len(accepted)}")
        print(f"❌ Rejected: {len(rejected)}")
        if rejected:
            print("\nRejected (score<th or gated):")
            for r in rejected[:20]:
                print(f" - {r.get('url')} ({r.get('similarity_score',0):.1f}%)")

    return accepted


if __name__ == "__main__":
    query = "Celine Black Slim Rectangle Sunglasses"
    product_link = get_product_urls(query)[0]
    amazon_link = get_product_urls(query, "amazon.com")[0]
    product_images = get_product_images(query)

    print(product_link)
    print(amazon_link)    
    print(get_valid_image(product_images))

    # brand = "Wilson"
    # local_brand_name = get_wikidata_alias(brand, "zh")
    # print(local_brand_name)
