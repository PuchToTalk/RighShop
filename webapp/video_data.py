import http.client
import json
import re
import socket
from google.cloud import firestore
import os
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs


# Google Cloud setup
os.environ['GCLOUD_PROJECT'] = 'righ-snap'
if os.path.isfile('keys/service-account-key.json'):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'keys/service-account-key.json'

collection_name = "video_metadata"
firestore_client = firestore.Client()

headers = {
        'X-API-KEY': "108d0d6e1320908677453002481b0ab25f4c10a7",
        'Content-Type': "application/json"
    }

def get_video_url(video_title, channel_name, website="youtube.com"):
    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        query = f"{video_title} {channel_name}:{website}"
        payload = json.dumps({
            "q": query,
            "num": 10
        })

        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        conn.close()
    except Exception as e:
        print(f"get_video_url Error, Query: {query}, Error: {e}")
        return ""

    if not data.strip():
        raise ValueError("Empty response from API.")

    try:
        response_json = json.loads(data)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw response: {repr(data)}")
        raise

    results = response_json.get("organic", [])
    video_url = ""
    for video in results:
        link = video.get("link", "")
        if "youtube.com/watch" in link:
            video_url = link
            break

    return video_url

def scrape_video_metadata(video_url):
    max_attempts = 3
    for attempt in range(max_attempts):  # try up to 3 times
        try:
            conn = http.client.HTTPSConnection("scrape.serper.dev", timeout=10)
            payload = json.dumps({
                "url": video_url,
            })
            conn.request("POST", "/", payload, headers)
            res = conn.getresponse()
            data = res.read().decode("utf-8")
            conn.close()
        except (socket.timeout, TimeoutError) as e:
            print(f"Request timed out (attempt {attempt + 1}), Query: {video_url}")
            if attempt == max_attempts:
                return "", ""
            continue
        except Exception as e:
            print(f"get_video_metadata Error (attempt {attempt + 1}), Query: {video_url}, Error: {e}")
            if attempt == max_attempts:
                return "", ""
            continue

        if not data.strip():
            print("Empty response from API.")
            if attempt == max_attempts:
                return "", ""
            continue

        try:
            response_json = json.loads(data)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Raw response: {repr(data)}")
            if attempt == max_attempts:
                return "", ""
            continue

        full_text = response_json.get("text", "")
        if not full_text.strip():
            print("No text")
            if attempt == max_attempts:
                return "", ""
            continue
        
        try:
            pattern = r"^(.*?)-{3,}\s*\nTranscript\s*\n(.*)$"
            match = re.search(pattern, full_text, re.DOTALL)
            if match:
                description = match.group(1).strip()
                transcript = match.group(2).strip()
            else:
                description = full_text.strip()
                transcript = ""
            
            return description, transcript

        except Exception as e:
            print(f"Error parsing description block: {e}")
            if attempt == 3:
                return "", ""

    return "", ""

def extract_video_id(url: str) -> str:
    try:
        parsed_url = urlparse(url)
        
        if "youtu.be" in parsed_url.netloc:
            return parsed_url.path.lstrip("/")
        
        if "youtube.com" in parsed_url.netloc:
            query = parse_qs(parsed_url.query)
            return query.get("v", [""])[0]
        
    except Exception as e:
        print(f"Error extract video ID: {e}")
        return ""

def metadata_exists(video_id: str) -> bool:
    doc_ref = firestore_client.collection(collection_name).document(video_id)
    doc = doc_ref.get()
    return doc.exists

def check_video_metadata_change(
    video_id: str,
    video_title: str,
    channel_name: str,
    duration: int
) -> dict:
    """
    Checks if title, channel name, or video duration fields change. (fields we have access to)
    """
    doc_ref = firestore_client.collection(collection_name).document(video_id)

    try:
        doc = doc_ref.get()
        if not doc.exists:
            return {"should_upload": True, "reason": "New video", "diff": {"video_id": video_id}}

        existing = doc.to_dict()
        diff = {}

        # if existing.get("title") != video_title:
        #     diff["title"] = (existing.get("title"), video_title)

        # if existing.get("channel") != channel_name:
        #     diff["channel"] = (existing.get("channel"), channel_name)

        if existing.get("duration") != duration:
            diff["duration"] = (existing.get("duration"), duration)

        return {
            "should_upload": bool(diff),
            "reason": "Changes detected" if diff else "No changes",
            "diff": diff
        }

    except Exception as e:
        return {
            "should_upload": False,
            "reason": f"Error during check: {e}",
            "diff": {}
        }


def upload_video_metadata(
    video_id: str,
    video_title: str,
    channel_name: str,
    duration: int,
    description: str,
    transcript: str
):
    doc_ref = firestore_client.collection(collection_name).document(video_id)
    now = datetime.now(timezone.utc)

    try:
        doc = doc_ref.get()
        new_metadata = {
            "video_id": video_id,
            "title": video_title,
            "channel": channel_name,
            "duration": duration,
            "description": description,
            "transcript": transcript,
            "updated_at": now
        }

        if doc.exists:
            existing = doc.to_dict()

            # prepare version snapshot
            previous_version = {
                "title": existing.get("title", ""),
                "channel": existing.get("channel", ""),
                "duration": existing.get("duration", 0),
                "description": existing.get("description", ""),
                "transcript": existing.get("transcript", ""),
                "updated_at": existing.get("updated_at", now)
            }

            doc_ref.update({
                "metadata_versions": firestore.ArrayUnion([previous_version]),
                **new_metadata
            })
            print(f"Updated metadata for video: {video_id}")
        else:
            new_metadata["created_at"] = now
            new_metadata["metadata_versions"] = []
            doc_ref.set(new_metadata)
            print(f"Uploaded new metadata for video: {video_id}")

    except Exception as e:
        print(f"Error upload_video_metadata: {e}")


def get_video_metadata(video_id: str):
    doc_ref = firestore_client.collection(collection_name).document(video_id)
    try:
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            description = data.get("description", "")
            transcript = data.get("transcript", "")
            print("Retrieved video metadata")
            return description, transcript
        else:
            print(f"No metadata found for video_id: {video_id}")
            return "", ""

    except Exception as e:
        print(f"Error get_video_metadata: {e}")
        return "", ""

    
# convert timestamp to total ms
def timestamp_to_ms(ts: str) -> int:
    try:
        parts = ts.split(":")
        if len(parts) == 2:   # MM:SS
            total_seconds = int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3: # HH:MM:SS
            total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            raise ValueError(f"Unexpected timestamp format: {ts}")
        return total_seconds * 1000
    except ValueError:
        return 0

# get time window of transcript based on time position in video (time position (sec) +- window_s (sec))
def get_transcript_window(transcript_text: str, time_position: int, window_s: int) -> str:
    try:
        # define regex pattern
        timestamp_pattern = re.compile(r"^\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*(.*)$")

        lines = transcript_text.splitlines()
        
        segments = []
        current_ts = None
        current_text_lines = []

        for line in lines:
            line_stripped = line.strip()

            match = timestamp_pattern.match(line_stripped)
            if match:
                if current_ts is not None:
                    full_text = "\n".join(current_text_lines).strip()
                    segments.append((current_ts, full_text))

                ts_str = match.group(1)
                ts_text = match.group(2)

                current_ts = timestamp_to_ms(ts_str)
                current_text_lines = [ts_text] if ts_text else []
            
            else:
                if current_ts is not None:
                    current_text_lines.append(line_stripped)
                else:
                    pass

        if current_ts is not None and current_text_lines:
            full_text = "\n".join(current_text_lines).strip()
            segments.append((current_ts, full_text))

        # deine upper and lower bound
        lower_bound = time_position - window_s
        upper_bound = time_position + window_s

        # filter segments in that window
        filtered = [(ts, txt) for (ts, txt) in segments if lower_bound <= ts <= upper_bound]

        result_lines = []
        for ts, txt in filtered:
            result_lines.append(f"{txt}")

        return "\n".join(result_lines)
    
    except Exception as e:
        print(f"get_transcript_window error: {e}")
        return ""


if __name__ == "__main__":
    video_title = "LE SSERAFIM (르세라핌) 'EASY' OFFICIAL MV"
    channel_name = "HYBE LABELS"
    video_url = get_video_url(video_title, channel_name, website="youtube.com")
    print(video_url)
    video_id = extract_video_id(video_url)
    print(video_id)
    is_metadata = metadata_exists(video_id)
    print(is_metadata)

    if is_metadata:
        video_title, channel_name, description, transcript = get_video_metadata(video_id)

    else:
        description, transcript = scrape_video_metadata(video_url)
        upload_video_metadata(video_id, video_title, channel_name, description, transcript)
        
    print("description:", description)

    transcript_window = get_transcript_window(transcript, time_position=10, window_s=18.9)
    print("transcript:", transcript_window)

    # video_metadata = {"title": video_title, "channel": channel_name, "description": description}
    # print(video_metadata)