from google.cloud import storage, firestore
import os
from datetime import datetime, timezone

# Google Cloud setup
os.environ['GCLOUD_PROJECT'] = 'righ-snap'
if os.path.isfile('keys/service-account-key.json'):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'keys/service-account-key.json'
bucket_name = 'righ-shop-images'
collection_name = 'shop-test'

def upload_image_from_file(image_path: str, collection_name="shop-test") -> str:
    # Initialize GCS client and specify the bucket name
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    bucket_loc = os.path.join(f'{collection_name}/', os.path.basename(image_path))
    blob = bucket.blob(bucket_loc)
    
    try:
        # Upload image
        blob.upload_from_filename(image_path)
        print('Uploaded Image.')
        image_url = blob.public_url
        return image_url
    except Exception as e:
        print(f'Failed to upload image: {e}')
        return ""

def upload_image_from_str(image_bytes: str, upload_id: str, collection_name="alpha-dogfood") -> str:
    # Initialize GCS client and specify the bucket name
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    bucket_loc = os.path.join(f'{collection_name}/', upload_id)
    blob = bucket.blob(bucket_loc)
    
    try:
        # Upload image
        blob.upload_from_string(image_bytes)
        image_url = blob.public_url
        return image_url
    except Exception as e:
        print(f'Failed to upload image: {e}')
        return ""

def upload_response(user_id: str, upload_id: str, image_url: str, final_response: str, llm_responses: dict, latency: float, usage: dict, video_metadata = {}):
    try:
        firestore_client = firestore.Client()
        doc_id = f'{user_id}-{upload_id}'
        doc_ref = firestore_client.collection(collection_name).document(doc_id)

        doc_ref.set({
            'user_id': user_id,
            'image_url': image_url,
            'final_response': final_response,
            'llm_responses': llm_responses,
            'latency': latency,
            'usage': usage,
            'video_metadata': video_metadata,
            'timestamp': datetime.now(timezone.utc),
            'feedback': {}
        })

        print('Uploaded response.')
    except Exception as e:
        print(f"Failed to upload response: {e}")
    
if __name__ == "__main__":
    url = upload_image_from_file("images/pants.jpg")
    print(url)
