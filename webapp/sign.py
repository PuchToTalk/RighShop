from google.cloud import storage
import datetime
import os
import argparse

if os.path.isfile("keys/service-account-key.json"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "keys/service-account-key.json"
storage_client = storage.Client()
BUCKET_NAME = "righ-shop-images/alpha-dogfood"

def generate_signed_url(blob_name: str) -> str:
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    signed_url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(days=7),
        method="GET",
    )
     
    return signed_url


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blob", type=str, required=True)
    args = parser.parse_args()

    signed_url = generate_signed_url(args.blob)
    print(signed_url)