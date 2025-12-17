import http.client
import requests
import json
import time
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import Optional, Any
from dotenv import load_dotenv
import os
from openai import OpenAI

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



headers = {
    'X-API-KEY': '108d0d6e1320908677453002481b0ab25f4c10a7',
    'Content-Type': 'application/json'
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
    'youtube.com', 'linkedin.com', 'instagram.com', 'tiktok.com','domestika.org'
    'blog.naver', 'github.com', 'news.', '/news', 'post.naver', '.wiki',
    '.pdf', '/article', '/photos', '/feed', '.txt', '/view', '.json'
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

ALLOWED_FORMATS = {"JPEG", "PNG", "GIF", "WEBP", "BMP", "WBMP", "HEIF", "AVIF"}

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp",
    "image/vnd.wap.wbmp", "image/heif", "image/avif"
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


# def extract_product_info(text_content: str) -> tuple[str, str, str]:
#     if not text_content:
#         return ("--failed--", "UNKNOWN", "")

#     user_prompt = f"""
#     You are an e‑commerce data extractor.
#     From the product page text below, reply ONLY with a JSON object:

#       "price"        – most prominent current price (numbers only)
#       "availability" – IN_STOCK, OUT_OF_STOCK, or UNKNOWN
#       "name"         – concise product title in the original language

#     No currency symbols, commas, or extra words in "price".
#     Output strictly valid JSON and nothing else.

#     --- TEXT START ---
#     {text_content}
#     --- TEXT END ---
#     """

#     try:
#         resp = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system",
#                  "content": "Return a JSON object with keys price, availability, name — no extra text."},
#                 {"role": "user", "content": user_prompt}
#             ],
#             temperature=0,
#             max_tokens=60,
#             response_format={"type": "json_object"}   # <- forces valid JSON
#         )

#         data = json.loads(resp.choices[0].message.content)

#         price = str(data.get("price", "")).strip()
#         availability = data.get("availability", "UNKNOWN").upper()
#         name = str(data.get("name", "")).strip()[:120]

#         return (price, availability, name)

#     except Exception as e:
#         print(f"[LLM] unified extractor failed: {e}")
#         return ("--failed--", "UNKNOWN", "")

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

    print("\n[FINDER] Asking LLM to find the core product snippet from the page...")
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
        
        print(f"[FINDER] Success! Got a snippet of {len(golden_snippet)} characters.")

        print("[EXTRACTOR] Asking LLM to extract structured JSON from the clean snippet...")
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
        
        print("[EXTRACTOR] Success! Parsed all product info.")
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
) -> Optional[tuple[str, str]]:
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
        return (url, "UNKNOWN")
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
    return [link for link in links if not any(domain in link for domain in NON_SHOPPING_DOMAINS)]

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

def get_product_urls(query, shop_website="", country="us", num_results=10) -> list:
    connection = http.client.HTTPSConnection("google.serper.dev")
    query = f"{query}:{shop_website}" if shop_website else query
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
            connection.request("POST", "/search", payload, headers)
            res = connection.getresponse()
            data = json.loads(res.read().decode("utf-8"))
            search_results = data.get("organic", [])

            links = [result.get("link", "") for result in search_results]
            filtered_links = filter_shopping_links(links)
            product_urls = filtered_links if filtered_links else []
            return product_urls

        except Exception as e:
            print(f"get_product_url Error, Query: {query}, Attempt: {attempt + 1}, Error: {e}")
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

if __name__ == "__main__":
    # For this test, I'll use the previously failing eBay and Bottega Veneta URLs
    SAMPLE_URLS = {
        # KR links
        #"H&M Oversized Shirt": "https://www2.hm.com/ko_kr/productpage.1229356004.html",
        "ebay" : "https://www.ebay.com/itm/176822894614",
        "W‑Concept Dress": "https://www.wconcept.co.kr/Product/306122244",
        #"Coupang AirPods Pro": "https://www.coupang.com/vp/products/4925896972",

        # Global links
        "Amazon # 1": "https://www.amazon.com/Gomyrod-Storage-Colorful-Supporters-Portable/dp/B0DF741998/?_encoding=UTF8&pd_rd_w=nYHaC&content-id=amzn1.sym.a602a706-e4fe-481e-98c3-9b75060fd322%3Aamzn1.symc.abfa8731-fff2-4177-9d31-bf48857c2263&pf_rd_p=a602a706-e4fe-481e-98c3-9b75060fd322&pf_rd_r=V2JDZSQNP1CG1DVFCYK9&pd_rd_wg=J1Ap7&pd_rd_r=76bd380f-14f3-40a0-9e6c-ef341761b0c4&ref_=pd_hp_d_btf_ci_mcx_mr_ca_id_hp_d&th=1",
        #"Amazon": "https://www.amazon.com/Louis-Vuitton-Pre-Loved-Bandouliere-Monogram/dp/B0FJ6BPXKT/ref=sr_1_1?ie=UTF8&sr=1-1&qid=1753116670&content-id=amzn1.sym.d99aee65-6128-42f8-a4b9-4afdcfc41385&pd_rd_w=FWmb0&dib=eyJ2IjoiMSJ9.iVZ_OAU6MjCxCsXNd8ukmm4IzLRfx9iyuEBWxqyco4poG8kSYhp_RI2AJQ4pLOz5SU-TkwkM96-54i1-hL-QGxklzHlNMPg6gZYHcDUntACXpsvJd_3iKPH75rZuLfp73o296MtIbqJ1aE4X02MpxfMKAKhRQZ2imAzhuAcsJCmv4_MUI5m8d5ZFaqPADRXHQ1Kh8TI4wbS1MdtV9GPRnni5XYTfenWwFRuCbT9HAyv7KW6fGCX4shRr5ixFg5s61DkycqdGR9QE9bpd-aHxOdVL5_0K2MMcz10B9d9sO40.-yTmk5RmA8B8oG6EbsHlE-wq6hyfYtdMz65qKc_4MOw&rnid=85457740011&dib_tag=se&pd_rd_r=43c60aa6-d454-4c05-a5fd-0a2c615b00a0&refinements=p_n_feature_fourteen_browse-bin%3A204406298011%2Cp_123%3A423616&ASIN=B0FJ6BPXKT&_encoding=UTF8&pd_rd_wg=2yaw0&ref_=lx_bd",
        "eBay Ray‑Ban Glasses": "https://www.ebay.com/itm/303339913240",
        #"H‑Shop PDF (should fail)": "https://arca.live/b/characterai/124585517",
        "Amazon Category" : "https://www.amazon.com/s?k=box&i=tools&crid=2R3AG1BW298Y7&sprefix=box+%2Ctools%2C482&ref=nb_sb_noss_2",
        "Amazon out of stock" : "https://www.amazon.com/Rumic-Theater-Vol-Rumiko-Takahashi/dp/1569310548",
        "Adidas" : "https://www.adidas.co.kr/superstar-shoes/S79916.html",
        "test1" : "https://www.amazon.com/Leather-Loafers-Classic-Business-Wedding/dp/B0F63CZZH9",
        "test2" : "https://www.bottegaveneta.com/en-en/james-lace-up-shoe-black-837736V2WX01000.html",
    }

    print("\n=== Golden Snippet Extraction Test ===")
    
    with requests.Session() as sess: # or your selenium setup
        for label, url in SAMPLE_URLS.items():
            print(f"\n--- Processing: {label} ---")
            
            text_content = get_text_from_url(url, sess) 
            
            price, availability, name = extract_product_info_with_snippet(text_content)
            
            if price != "--failed--":
                print(f"✅ SUCCESS: {label:<25} -> price: {price:<10} | stock: {availability:<11} | name: {name}")
            else:
                print(f"❌ FAILED: {label:<25} -> Could not extract info.")

    print("\n=== Done ===")
