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
import os

# Attempt to import optional dependencies. ``dotenv`` is used to load
# environment variables from a .env file, and ``openai`` is used for
# language model calls. If these modules are not available (e.g., in a
# restricted environment), we provide safe fallbacks so that the rest
# of the module can still be imported and used for its non‑LLM features.
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        """Fallback no‑op load_dotenv if python‑dotenv is not installed."""
        return None

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    OpenAI = None  # type: ignore
    _HAS_OPENAI = False

try:
    import cloudscraper
    _HAS_CLOUDSCRAPER = True
    print("Cloudscraper library found. Will use for Cloudflare-protected sites.")
except ImportError:
    _HAS_CLOUDSCRAPER = False
    print("Warning: cloudscraper library not found. Install with 'pip install cloudscraper' for better results.")


load_dotenv()

# Configuration for OpenAI API. If the OpenAI module or API key is not
# available, ``client`` will remain ``None``. Downstream functions that
# rely on the client should check for ``None`` and raise or handle
# appropriately.
api_key = os.getenv("OPENAI_API_KEY")
if _HAS_OPENAI and api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None

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
    'youtube.com', 'linkedin.com', 'instagram.com', 'tiktok.com',
    'blog.naver', 'github.com', 'news.', '/news', 'post.naver', '.wiki',
    '.pdf', '/article', '/photos', '/feed', '.txt', '/view', '.json'
]

NON_SHOPPING_DOMAINS += [
    '.pdf', '.ppt', '.xls', '.zip',
    'arca.live', 'personality-database.com',
    'kipo.go.kr', '/download', '/attachment',
    # Blogs and community sites which are unlikely to host direct product pages
    'tistory.',  # filter tistory.com and subdomains
    'blog.',     # generic blog subdomains (e.g. blogspot.com)
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


def get_text_from_url(url: str, session: requests.Session) -> Optional[str]:
    """
    Fetch the textual content from a URL using the provided session.

    This helper will attempt to retrieve the page with a normal ``requests.Session`` and,
    if blocked by common bot mitigations such as HTTP 403 (Forbidden) or 429
    (Too Many Requests), will fall back to using the ``cloudscraper`` library if it
    is available. The returned text has all script/style tags removed and is
    truncated to at most 15,000 characters to avoid overwhelming downstream
    language model calls.

    Parameters
    ----------
    url : str
        The URL to fetch.
    session : requests.Session
        The HTTP session to use for the initial request. The session's headers
        will be updated with a randomly selected User‑Agent to reduce the
        likelihood of being blocked.

    Returns
    -------
    Optional[str]
        A cleaned string of page text or ``None`` if the page could not be loaded.
    """
    try:
        # Rotate the user agent on each request. This helps bypass basic bot
        # filtering that relies on static headers. We update the session's
        # headers directly so that any subsequent redirects inherit the same
        # User‑Agent.
        session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
        response = session.get(url, headers=BROWSER_HEADERS, timeout=15, allow_redirects=True)

        # If the server returns a 403 (Forbidden) or 429 (Too Many Requests),
        # attempt to bypass Cloudflare or other protections using cloudscraper.
        if response.status_code in (403, 429) and _HAS_CLOUDSCRAPER:
            print(f"Standard request failed with {response.status_code}. Retrying with cloudscraper for {url}...")
            scraper = cloudscraper.create_scraper()
            response = scraper.get(url, timeout=15)

        # Raise an HTTPError if the status is still not successful (4xx/5xx)
        response.raise_for_status()

        # Parse the HTML content and remove script/style tags.  Use ``response.text``
        # so that requests can decode the bytes into a string using the
        # appropriate encoding. This helps avoid garbled binary text from
        # compressed responses.
        soup = BeautifulSoup(response.text, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Extract visible text, collapse whitespace and newlines
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = '\n'.join(chunk for chunk in chunks if chunk)

        # Limit text size to avoid sending huge documents to the API.
        # Allow up to 20k characters so that enough context is preserved for the
        # snippet extractor while still keeping within common token limits.
        return clean_text[:20000]

    except requests.exceptions.RequestException as e:
        print(f"[get_text_from_url] Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        print(f"[get_text_from_url] An unexpected error occurred while processing URL {url}: {e}")
        return None
    
def extract_price_from_text(text_content):
    if not text_content:
        return "No text content was provided to analyze."

    # If OpenAI isn't available, we cannot call the LLM. Return None to
    # indicate that no price was extracted via this method.
    if client is None:
        return None

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
        return None

# #for testing {
# def extract_availability_from_text(text_content: str) -> str:
#     if not text_content:
#         return "UNKNOWN"

#     # If OpenAI isn't available, default to UNKNOWN as we cannot classify
#     if client is None:
#         return "UNKNOWN"

#     prompt = f"""
#     You are an e‑commerce availability detector.
#     From the product‑page text below, answer with **only** one word:

#     IN_STOCK       → the main item can be purchased now
#     OUT_OF_STOCK   → the main item is not purchasable (sold out, unavailable) 
#     UNKNOWN        → you cannot decide

#     Do not explain yourself; output just the keyword.

#     --- TEXT START ---
#     {text_content}
#     --- TEXT END ---
#     Answer:
#     """
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4.1-2025-04-14",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0,
#             max_tokens=5,
#         )
#         return response.choices[0].message.content.strip().upper()
#     except Exception as e:
#         print(f"[LLM] Availability extraction failed: {e}")
#         return "UNKNOWN"

# def get_product_name(url: str, sess: requests.Session, timeout: int = 15) -> str:
#     """
#     Attempt to extract a product name/title from a webpage using only basic
#     HTML parsing. This function avoids using heavy language models and
#     intentionally does not pass a custom header set, relying on the session's
#     existing headers (which may have been randomized by ``get_text_from_url``)
#     to reduce the likelihood of being blocked.

#     It checks common locations for a product name in order:
#     1. Open Graph ``og:title`` meta tag
#     2. A ``meta name="title"`` tag
#     3. The first ``<h1>`` element on the page
#     4. The ``<title>`` element in the ``<head>``

#     If none of these are found, an empty string is returned.
#     """
#     try:
#         resp = sess.get(url, timeout=timeout, allow_redirects=True)
#         # If we get blocked, try cloudscraper as a fallback.
#         if resp.status_code == 403 and _HAS_CLOUDSCRAPER:
#             scraper = cloudscraper.create_scraper()
#             resp = scraper.get(url, timeout=timeout, allow_redirects=True)
#         resp.raise_for_status()

#         soup = BeautifulSoup(resp.content, "html.parser")

#         # 1. OpenGraph
#         tag = soup.find("meta", property="og:title")
#         if tag and tag.get("content"):
#             return tag["content"].strip()[:120]

#         # 2. meta name="title"
#         tag = soup.find("meta", attrs={"name": "title"})
#         if tag and tag.get("content"):
#             return tag["content"].strip()[:120]

#         # 3. first <h1>
#         tag = soup.find("h1")
#         if tag and tag.get_text(strip=True):
#             return tag.get_text(" ", strip=True)[:120]

#         # 4. <title> tag
#         tag = soup.find("title")
#         if tag and tag.get_text(strip=True):
#             return tag.get_text(" ", strip=True)[:120]

#         return ""
#     except Exception as e:
#         print(f"[NAME] extraction failed for {url}: {e}")
#         return ""

# def get_product_name_using_LLM(
#     url: str,
#     sess: requests.Session,
#     max_chars: int = 4000,
# ) -> str:

#     page_text = get_text_from_url(url, session=sess)
#     if not page_text:
#         return ""
#     page_text = page_text[:max_chars]

#     prompt = f"""
#     You are an e‑commerce title extractor.
#     From the product‑page text below, output ONLY the item’s main name /
#     title exactly as written on the site. Keep the original language
#     (English, Korean, Japanese, etc.). Do not add quotes, extra words,
#     or explanations—just the name.

#     --- TEXT START ---
#     {page_text}
#     --- TEXT END ---
#     Name:
#     """

#     # Bail out immediately if OpenAI is not available.
#     if client is None:
#         return ""

#     try:
#         resp = client.chat.completions.create(
#             model="gpt-4.1-2025-04-14",
#             messages=[
#                 {"role": "system",
#                  "content": "You output only the product title—no other text."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0,
#             max_tokens=40,
#         )
#         name = resp.choices[0].message.content.strip()
#         return name[:120]          # safety truncate
#     except Exception as e:
#         print(f"[LLM‑NAME] extraction failed for {url}: {e}")
#         return ""

# def extract_product_info(text_content: str) -> tuple[str, str, str]: # for testing
#     if not text_content or len(text_content.strip()) < 100:
#         print("[LLM Extractor] Failed: Input text is empty or too short.")
#         return ("--failed--", "UNKNOWN", "")
    
#     blocker_keywords = ["verify you are a human", "access denied", "are you a robot", "captcha"]
#     if any(keyword in text_content.lower() for keyword in blocker_keywords):
#         print("[LLM Extractor] Failed: Input text appears to be a block/CAPTCHA page.")
#         return ("--failed--", "UNKNOWN", "")

#     user_prompt = f"""
#     You are an expert e-commerce data extractor.
#     From the product page text below, you must reply ONLY with a single, valid JSON object.
#     The JSON object must have these three keys:

#       "price"        : The most prominent current price. Extract numbers only (e.g., 29.99 or 35000). If no price is found, the value should be null.
#       "availability" : The stock status. The value must be one of these three strings: "IN_STOCK", "OUT_OF_STOCK", or "UNKNOWN".
#       "name"         : The concise product title in its original language.

#     Do not add any text, explanations, or markdown before or after the JSON object.

#     --- TEXT START ---
#     {text_content}
#     --- TEXT END ---
#     """

#     raw_llm_output = None
#     # If OpenAI isn't available, we cannot call the unified extractor
#     if client is None:
#         return ("--failed--", "UNKNOWN", "")

#     try:
#         resp = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are a JSON-only API that extracts product data."},
#                 {"role": "user", "content": user_prompt}
#             ],
#             temperature=0,
#             max_tokens=150,
#             response_format={"type": "json_object"}
#         )

#         raw_llm_output = resp.choices[0].message.content
#         data = json.loads(raw_llm_output)

#         price_value = data.get("price")
#         price = str(price_value).strip() if price_value is not None else ""

#         availability_value = str(data.get("availability", "UNKNOWN")).upper()
#         allowed_stocks = {"IN_STOCK", "OUT_OF_STOCK"}
#         availability = availability_value if availability_value in allowed_stocks else "UNKNOWN"

#         name = str(data.get("name", "")).strip()[:120]

#         if not price or not name:
#             print(f"[LLM Extractor] Failed: Could not extract price or name. Got: price='{price}', name='{name}'")
#             return ("--failed--", "UNKNOWN", "")

#         return (price, availability, name)

#     except json.JSONDecodeError:
#         print(f"[LLM Extractor] CRITICAL: Failed to decode JSON.")
#         print(f"--- LLM Raw Output Start ---\n{raw_llm_output}\n--- LLM Raw Output End ---")
#         return ("--failed--", "UNKNOWN", "")
    
#     except Exception as e:
#         print(f"[LLM Extractor] An unexpected error occurred: {e}")
#         return ("--failed--", "UNKNOWN", "")
# #}

def extract_product_info_with_snippet(text_content: str) -> tuple[str, str, str]:
    """
    Extracts product info using the two-step "Golden Snippet" technique.
    Now includes an LLM-based classification in the finder step to decide
    whether the page is a shopping/product page. If it is NOT, returns the
    usual failure sentinel ("--failed--", "UNKNOWN", "") to avoid changing
    any downstream logic.
    """
    if not text_content or len(text_content.strip()) < 100:
        print("[SNIPPET EXTRACTOR] Failed: Input text is empty or too short.")
        return ("--failed--", "UNKNOWN", "")

    # If OpenAI isn't available, keep previous behavior (early fail).
    if client is None:
        return ("--failed--", "UNKNOWN", "")

    # --- Call 1: The Finder (now classifies + returns snippet via JSON) ---
    print("\n[FINDER] Asking LLM to classify and find the core product snippet...")
    finder_prompt = f"""
    You are given raw, unstructured text scraped from a webpage. Determine whether it is a SHOPPING/PRODUCT page
    (i.e., a page that presents a specific product for sale) and, if so, isolate ONLY the main product offer.

    OUTPUT FORMAT (JSON only; no extra text):
    {{
      "is_shopping": <true|false>,
      "snippet": <string or null>   // if is_shopping is true, include a minimal contiguous block of text containing
                                    // the primary product's title/name, its current selling price, and its main
                                    // availability/purchase status (e.g., Add to Cart / Out of Stock). Preserve the
                                    // original language. If you cannot find such a block, set snippet to null.
    }}

    Guidance:
    • Consider cues like "Add to Cart", "Buy Now", size/quantity controls, or explicit "Out of Stock" notices
      (in the page's original language).
    • Ignore third-party offers, used items, instalment plans, shipping fees, loyalty points, or reviews.
    • If it is a blog, news, forum, category/collection page without a specific buyable product, return is_shopping=false.

    --- RAW TEXT START ---
    {text_content[:20000]}
    --- RAW TEXT END ---
    """

    try:
        finder_resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": finder_prompt}
            ],
            temperature=0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        finder_payload = finder_resp.choices[0].message.content
        # Parse the JSON the finder produced
        try:
            finder_data = json.loads(finder_payload)
        except Exception:
            print("[FINDER] Failed to parse JSON; treating as non-shopping.")
            return ("--failed--", "UNKNOWN", "")

        if not bool(finder_data.get("is_shopping", False)):
            print("[FINDER] Classified as NON_SHOPPING.")
            return ("--failed--", "UNKNOWN", "")

        golden_snippet = (finder_data.get("snippet") or "").strip()
        if not golden_snippet:
            print("[FINDER] No usable snippet returned for a shopping page.")
            return ("--failed--", "UNKNOWN", "")

        print(f"[FINDER] Success! Got a snippet of {len(golden_snippet)} characters.")

        # --- Call 2: The Extractor (unchanged logic) ---
        print("[EXTRACTOR] Asking LLM to extract structured JSON from the clean snippet...")
        extractor_prompt = f"""
        From the clean product snippet below (which may be in any language), output a JSON object with exactly three keys: "price", "availability", and "name".

        • "price": Extract the main selling price only. Return numbers and decimals without currency symbols or commas. Ignore list prices, struck-through prices, instalments, shipping, loyalty points, or ratings.
        • "availability": Respond with "IN_STOCK", "OUT_OF_STOCK", or "UNKNOWN". Consider the item "IN_STOCK" only if there is a clear way to purchase it now (e.g., an 'Add to Cart' button). It is "OUT_OF_STOCK" if the text explicitly states it is unavailable/sold out. If you cannot tell, return "UNKNOWN".
        • "name": Provide a concise title for the product exactly as written on the site, preserving the original language. Do not include brand slogans, categories, or extra descriptors.

        Return only raw JSON (no markdown or extra text).

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
            print("[EXTRACTOR] Failed: Could not extract name from the snippet.")
            return ("--failed--", "UNKNOWN", "")

        print("[EXTRACTOR] Success! Parsed all product info.")
        return (price, availability, name)

    except Exception as e:
        print(f"[SNIPPET EXTRACTOR] An unexpected error occurred: {e}")
        return ("--failed--", "UNKNOWN", "")


# TODO: calculate the price and latency after running extract_product_info_with_snippet
# TODO: try out with different model

# DEMO: Geo-locking temporarily disabled for demonstration.
# GEO_LOCKED = {"coupang.com"}

GEO_LOCKED: set[str] = set()

PRICE_REGEX = re.compile(r'(?:[$€£¥₩]\s?[0-9][0-9,.]*)')

# OLD VERSION NO LLM {
def is_shopping_page(text_content: str) -> bool:
    """
    Classify whether the supplied text content appears to come from a shopping
    (product) page. If the OpenAI client is available, a lightweight LLM
    classifier is used. Otherwise, fall back to keyword heuristics.

    Parameters
    ----------
    text_content : str
        A chunk of page text (ideally the first few thousand characters).

    Returns
    -------
    bool
        True if the page appears to be a product/shopping page, False otherwise.
    """
    if not text_content:
        return False

    # Heuristic fallback for when no LLM client is available.
    if client is None:
        lowered = text_content.lower()
        # Fallback heuristic: require at least one strong purchase cue. A single
        # mention of "price" or "product" on a blog may not indicate a shopping page,
        # so we look for explicit action phrases.
        keywords = [
            'add to cart', 'buy now', 'add to basket', 'checkout', 'free shipping',
            'select size', 'choose size', 'quantity', '구매하기', '장바구니', '바로 구매',
        ]
        return any(k in lowered for k in keywords)

    # Use a small classification prompt to decide if this is a shopping page
    prompt = f"""
You are a classifier that decides if a webpage is a shopping/product page.
The user will provide a chunk of text extracted from a webpage. Respond with
only one word:

SHOPPING       – if the text appears to describe a product for sale (with price,
                 add‑to‑cart, size options, etc.)
NON_SHOPPING   – if the text is news, blog, forum, or any non‑product content.

--- TEXT START ---
{text_content[:4000]}
--- TEXT END ---

Answer:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5
        )
        label = response.choices[0].message.content.strip().upper()
        return label == "SHOPPING"
    except Exception as e:
        print(f"[is_shopping_page] LLM classification failed: {e}")
        # Fall back to heuristic if classification fails
        lowered = text_content.lower()
        keywords = [
            'add to cart', 'buy now', 'add to basket', 'checkout', 'free shipping',
            'select size', 'choose size', 'quantity', '구매하기', '장바구니', '바로 구매',
        ]
        return any(k in lowered for k in keywords)

def extract_price_from_html(html: str) -> Optional[str]:
    """
    Attempt to extract a numeric price from raw HTML using structured data
    and simple regex patterns. This function is a lightweight alternative to
    calling the LLM when only the price is missing. It looks for values in
    meta tags and JSON‑LD scripts commonly used on e‑commerce pages. If no
    structured price is found, it falls back to ``PRICE_REGEX`` on the raw
    HTML text.

    Parameters
    ----------
    html : str
        The raw HTML of a product page.

    Returns
    -------
    Optional[str]
        The price as a string containing only digits and decimal points, or
        ``None`` if no price could be confidently extracted.
    """
    if not html:
        return None

    # Parse the HTML
    soup = BeautifulSoup(html, 'html.parser')

    # 1. Check common meta tags (Open Graph, Twitter Card, product schema)
    meta_properties = [
        # Common price meta tags used by Open Graph and product schemas.
        'product:price:amount',
        'product:price:regular_amount',
        'product:price:sale_amount',
        'og:price:amount',
        'og:price:standard_amount',
        # Do not include twitter:data1 – it often holds unrelated values such as ratings.
        'price',
    ]
    for prop in meta_properties:
        for tag in soup.find_all('meta', attrs={'property': prop}):
            content = tag.get('content')
            if content and re.search(r'\d', content):
                # Remove currency symbols and non‑numeric characters except dot/comma
                matches = re.findall(r'[0-9,.]+', content)
                if matches:
                    return matches[0].replace(',', '')
        for tag in soup.find_all('meta', attrs={'name': prop}):
            content = tag.get('content')
            if content and re.search(r'\d', content):
                matches = re.findall(r'[0-9,.]+', content)
                if matches:
                    return matches[0].replace(',', '')

    # 2. Check for itemprop="price"
    for price_tag in soup.find_all(attrs={'itemprop': 'price'}):
        content = price_tag.get('content') or price_tag.get_text()
        if content and re.search(r'\d', content):
            matches = re.findall(r'[0-9,.]+', content)
            if matches:
                return matches[0].replace(',', '')

    # 3. Check JSON-LD structured data
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            data = json.loads(script.string)
        except Exception:
            continue
        # JSON-LD can be a dict or a list of dicts
        data_list = data if isinstance(data, list) else [data]
        for item in data_list:
            if not isinstance(item, dict):
                continue
            # Check Product or Offer schema
            if item.get('@type') in ('Product', 'Offer'):
                # Some Product schemas nest offers under offers/offers\n items
                if 'offers' in item:
                    offers = item['offers']
                    offers_list = offers if isinstance(offers, list) else [offers]
                    for offer in offers_list:
                        price = offer.get('price') or offer.get('priceSpecification', {}).get('price')
                        if price and isinstance(price, (str, int, float)):
                            return str(price)
                # Direct price on the item itself
                price = item.get('price') or item.get('priceSpecification', {}).get('price')
                if price and isinstance(price, (str, int, float)):
                    return str(price)
    # 4. Fall back to simple regex search on the raw HTML
    # As a last resort, scan the raw HTML for currency‑prefixed numbers (e.g., "$99.99").
    # Collect all matches and return the largest value, assuming the highest price is the main product price.
    matches = PRICE_REGEX.findall(html)
    if matches:
        values: list[float] = []
        for m in matches:
            # Extract the numeric portion and convert to float
            numeric = re.findall(r'[0-9,.]+', m)
            if numeric:
                try:
                    values.append(float(numeric[0].replace(',', '')))
                except ValueError:
                    continue
        if values:
            # Return the maximum value as a string (no commas)
            max_val = max(values)
            # Remove any trailing .0 for integer prices
            return (str(int(max_val)) if max_val.is_integer() else str(max_val))

    return None

def _needs_proxy(domain: str) -> bool:
    return any(d in domain for d in GEO_LOCKED)

# }


def validate_generic_link(
    url: str,
    sess: requests.Session,
    proxy_dict: Optional[dict] = None,
    timeout: int = 15,
) -> Optional[tuple[str, str, str, str]]:  # return 4-tuple
    if not url.startswith(("https://", "http://")):
        return None

    dom = urlparse(url).netloc.lower()
    if any(nd in dom or nd in url for nd in NON_SHOPPING_DOMAINS):
        print(f"[FILTERED] Non-shopping domain -> {url}")
        return None

    # Randomise only the UA (and optionally Accept-Language) for this session.
    sess.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
    })
    proxies = proxy_dict if proxy_dict and _needs_proxy(dom) else None

    try:
        resp = sess.get(url, timeout=timeout, allow_redirects=True, proxies=proxies)

        if resp.status_code in (301, 302, 303, 307, 308):
            target = resp.headers.get("Location", resp.url)
            resp = sess.get(target, timeout=timeout, allow_redirects=True, proxies=proxies)

        if resp.status_code in (403, 429) and _HAS_CLOUDSCRAPER:
            scraper = cloudscraper.create_scraper(browser={"custom": "Scraper 1.3"})
            resp = scraper.get(url, timeout=timeout, allow_redirects=True)

        if not (200 <= resp.status_code < 400):
            print(f"[FILTERED] Status {resp.status_code} -> {url}")
            return None

        if len(resp.content) < 4000:
            print(f"[FILTERED] Page too small (~{len(resp.content)} B) -> {url}")
            return None

        page_text = get_text_from_url(resp.url, session=sess)
        if not page_text:
            print(f"[FILTERED] Unable to extract text from {url}")
            return None

        if not is_shopping_page(page_text):
            print(f"[FILTERED] Classified as NON_SHOPPING -> {url}")
            return None

        price, availability, name = extract_product_info_with_snippet(page_text)

        def price_in_text(val: str, text: str) -> bool:
            if not val:
                return False
            p = val.replace(',', '').strip()
            if not p or not any(ch.isdigit() for ch in p):
                return False
            lowered = text.lower()
            if p.lower() in lowered:
                return True
            int_part = p.split('.')[0]
            if int_part and int_part in lowered:
                return True
            return False

        def name_in_text(name_val: str, text: str) -> bool:
            if not name_val:
                return False
            n = ' '.join(name_val.split()).lower()
            t = text.lower()
            return n in t

        snippet_valid = True
        if price and not price_in_text(price, resp.text):
            snippet_valid = False
        if name and not name_in_text(name, page_text):
            snippet_valid = False
        if not price or price == "--failed--":
            snippet_valid = False

        if not snippet_valid:
            price = ""
            availability = ""
            name = ""

        if not price:
            fallback_price = extract_price_from_html(resp.text)
            if fallback_price:
                price = fallback_price
            else:
                extracted = extract_price_from_text(page_text)
                if extracted:
                    numeric = re.findall(r'[0-9,.]+', extracted)
                    if numeric:
                        price = numeric[0].replace(',', '')

        if not availability or availability == "UNKNOWN":
            availability = extract_availability_from_text(page_text)

        if not name:
            name = get_product_name(resp.url, sess)
            if not name:
                try:
                    with requests.Session() as tmp_sess:
                        name = get_product_name(resp.url, tmp_sess)
                except Exception:
                    name = ""
            if not name and client is not None:
                name = get_product_name_using_LLM(resp.url, sess)

        if not name or not price:
            print(f"[FILTERED] Could not extract name or price -> {url}")
            return None

        if availability == "OUT_OF_STOCK":
            print(f"[FILTERED] LLM says OUT_OF_STOCK -> {url}")
            return None

        allowed = {"IN_STOCK", "OUT_OF_STOCK"}
        availability = availability if availability in allowed else "UNKNOWN"

        return (resp.url, price or "", availability, name)

    except requests.ReadTimeout:
        print(f"[validate_generic_link] Ignoring ReadTimeout -> {url}")
        return (url, "", "UNKNOWN", "")  # shape: 4-tuple
    except requests.RequestException as exc:
        print(f"[FILTERED] {exc.__class__.__name__} -> {url}")
        return None



def get_validated_links(
    urls: list[str],
    session: requests.Session | None = None
) -> list[tuple[str, str, str, str]]:
    print("the link is getting validating")
    validated: list[tuple[str, str, str, str]] = []
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
        "test3" : "https://www.ae.com/us/en/p/women/button-up-shirts/button-up-shirts/ae-perfect-button-up-shirt/0355_6246_100",
        "test4" : "https://store.hermanmiller.com/dining-furniture-chairs-stools/eames-molded-plastic-side-chair/1104.html?lang=en_US&sku=100366090",
        "none shopping link" : "https://book-c.tistory.com/234",
        "genshin" : "https://hoyoversemerch.com/product/alhaitham-impression-t-shirt-genshin-impact/",
        "test5" : "https://www.shure.com/en-US/products/microphones/sm7b?variant=SM7B",
    }

    print("\n=== Golden Snippet Extraction Test ===")
    
    # It's best to use a single requests.Session or a single Selenium driver instance
    # if you are processing multiple URLs in a loop.
    with requests.Session() as sess: # or your selenium setup
        for label, url in SAMPLE_URLS.items():
            print(f"\n--- Processing: {label} ---")
            
            # Make sure you are using your BEST scraper function here.
            # A Selenium-based one is recommended for sites like eBay.
            # text_content = get_text_with_selenium(url) # Recommended
            text_content = get_text_from_url(url, sess) # Using your original for this example
            
            # Call the new, powerful snippet-based extractor
            price, availability, name = extract_product_info_with_snippet(text_content)
            
            if price != "--failed--":
                print(f"✅ SUCCESS: {label:<25} -> price: {price:<10} | stock: {availability:<11} | name: {name}")
            else:
                print(f"❌ FAILED: {label:<25} -> Could not extract info.")

    print("\n=== Done ===")
