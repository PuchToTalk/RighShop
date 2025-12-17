import requests
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import time

country_to_lang = {
    "kr": "ko",
    "tw": "zh",
}

def is_valid_wiki_link(url):
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc or "/wiki/" not in url:
        return False
    try:
        response = requests.get(url, timeout=3)
        return (
            response.status_code == 200 and
            "Wikipedia does not have an article with this exact name" not in response.text
        )
    except requests.RequestException:
        return False

def get_standard_title(title):
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "redirects": 1,
        "format": "json"
    }
    try:
        response = requests.get(api_url, params=params)
        data = response.json()
        pages = data["query"]["pages"]
        for page_id, page_data in pages.items():
            return page_data.get("title", title)
        return title
    except Exception as e:
        print(f"get_standard_title error: {e}")
        return title


def validate_links_parallel(people_list):
    results = [None] * len(people_list)
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {
            executor.submit(is_valid_wiki_link, person.get("bio", "")): idx
            for idx, person in enumerate(people_list)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                valid = future.result()
            except Exception:
                valid = False
            results[idx] = valid
    return results

def add_wiki_links(people_list: list[dict], country_code: str = "us"):
    # validate links in parallel
    valid_links = validate_links_parallel(people_list)
    lang_code = country_to_lang.get(country_code, "en")  # fallback to "en"

    # gather valid people and get standard titles
    person_to_canonical = defaultdict(list)
    standard_titles = []
    for idx, person in enumerate(people_list):
        if not valid_links[idx]:
            person["bio"] = ""
            person["photo"] = ""
            continue
        wiki_url = person.get("bio", "")
        path = urlparse(wiki_url).path
        raw_title = unquote(path.split("/wiki/")[-1])
        standard_title = get_standard_title(raw_title)
        if standard_title:
            standard_titles.append(standard_title)
            person_to_canonical[standard_title].append(person)

    # query Wikipedia for images and language links
    if not standard_titles:
        return people_list

    titles_str = "|".join(standard_titles)
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "langlinks|pageimages|thumbnail",
        "lllang": lang_code,
        "piprop": "thumbnail",
        "pithumbsize": 500,
        "titles": titles_str,
        "format": "json"
    }
    retries = 2
    for attempt in range(retries): # try twice
        try:
            r = requests.get(api_url, params=params, timeout=5)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(0.5)
            else:
                print("Max retries reached. Returning unmodified list.")
                return people_list
            
    pages = data.get("query", {}).get("pages", {})
    # print(pages)
    for page in pages.values():
        try:
            standard_title = page.get("title")
            people = person_to_canonical.get(standard_title, [])
            for person in people:
                thumb_url = page.get("thumbnail", {}).get("source", "")
                langlinks = page.get("langlinks", [])
                if country_code in country_to_lang.keys() and langlinks:
                    local_title = langlinks[0]["*"].replace(" ", "_")
                    person["bio"] = f"https://{lang_code}.wikipedia.org/wiki/{local_title}"
                else:
                    # use the standard en link for all other cases
                    person["bio"] = f"https://en.wikipedia.org/wiki/{standard_title.replace(' ', '_')}"
                person["photo"] = thumb_url if thumb_url else ""
        except Exception as e:
            print(f"error processing page: {e}")
            for person in people:
                person["bio"] = ""
                person["photo"] = ""

    return people_list


if __name__ == "__main__":
    wiki_page = [
        {"name": "James harden", "bio": "https://en.wikipedia.org/wiki/James_Harden"},
        {"name": "Hoyeon Jung", "bio": "https://en.wikipedia.org/wiki/HoYeon_Jung"},
        {"name": "Bong JoonHo", "bio": "https://en.wikipedia.org/wiki/Bong_Joon-ho"},
        {"name": "Squid Game", "bio": "https://en.wikipedia.org/wiki/Squid_Game"},
        {"name": "Fake Person", "bio": "https://en.wikipedia.org/wiki/Fake_Person"}
    ]

    country_code = "us"
    enriched = add_wiki_links(wiki_page, country_code=country_code)
    for p in enriched:
        print(p)
