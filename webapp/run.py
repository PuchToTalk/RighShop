from snap import capture_screen, encode_image, save_as_jpg, process_image_task
from detect_device import detect_adb_device
# from send_email import send_email
from upload_data import upload_image, upload_response
from concurrent.futures import ThreadPoolExecutor
import keyboard
import time
import os
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
from video_data import get_video_url, extract_video_id, check_video_metadata_change, scrape_video_metadata, upload_video_metadata, timestamp_to_ms

load_dotenv(override=True)
is_processing = False # track image processing

def process_and_upload(image, output_path):
    try:
        start_time = time.perf_counter()
        # generate upload_id
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')[:-3]
        unique_id = uuid.uuid4().hex[-10:]
        upload_id = f"{timestamp}-{unique_id}"

        user_id = "shop-link-test"
        
        # unique filename and path for local storage
        local_jpg_filename = f"{user_id}-{upload_id}.jpg"
        local_jpg_path = os.path.join("static/", local_jpg_filename)

        # preprocess image and convert to jpg
        # Convert BMP to JPG
        # preprocess_image(image)
        save_as_jpg(image, local_jpg_path)    

        # process image
        video_title = "LE SSERAFIM (르세라핌) 'EASY' OFFICIAL MV"
        channel_name = "HYBE LABELS"
        with open("timestamp.txt", "r", encoding="utf-8") as file:
            times = file.read().split("\n")
            timestamp = times[0]
            video_length = times[1]
        print(timestamp)
        time_position = timestamp_to_ms(timestamp) # replace with "position" from STB
        duration = timestamp_to_ms(video_length) # Replace with "duration" from STB
        video_url = get_video_url(video_title, channel_name, website="youtube.com")
        # video_url = ""
        video_id = extract_video_id(video_url) if video_url else ""

        if video_id:
            # check if metadata update is needed
            result = check_video_metadata_change(
                video_id=video_id,
                video_title=video_title,
                channel_name=channel_name,
                duration=duration
            )

            if result["should_upload"]:
                print(f"Uploading metadata for {video_id} - {result['reason']}")

                description, transcript = scrape_video_metadata(video_url)

                if not description:
                    print(f"No description and transcript available for {video_id}")
                else:
                    upload_video_metadata(
                        video_id=video_id,
                        video_title=video_title,
                        channel_name=channel_name,
                        duration=duration,
                        description=description,
                        transcript=transcript or ""
                    )
            else:
                print(f"Skipping upload for {video_id} - {result['reason']}")

        video_metadata = {
                        "video_id": video_id,
                        "title": video_title,
                        "position": time_position,
                        "duration": duration
                    }
        country = "kr" # gl_code
        has_people = True
        # scene_attr = "/r/recreation_room 281"
        scene_attr = None
        base64_image = encode_image(local_jpg_path)
        response_body = process_image_task(base64_image, 
                                           output_path, 
                                           video_metadata=video_metadata, 
                                           country=country, 
                                           has_people=has_people, 
                                           scene_attr=scene_attr
                                           )

        final_response = response_body['text']
        llm_responses = response_body['llm_responses']
        usage = response_body['usage']
        latency = response_body['latency']
        
        # # upload image to GCS
        # gcs_url = upload_image(local_jpg_path)

        end_time = time.perf_counter()
        total_latency = end_time - start_time
        latency.append({"total_latency": total_latency})
        print("total latency: ", total_latency)

        # # store llm response and metadata in Firestore
        # with ThreadPoolExecutor(max_workers=2) as executor:
        #     executor.submit(upload_response, 
        #                     user_id, 
        #                     upload_id, 
        #                     gcs_url, 
        #                     final_response, 
        #                     llm_responses, 
        #                     latency,
        #                     usage,
        #                     video_metadata
        #                     )
        # upload_response(user_id, upload_id, gcs_url, final_response, llm_responses, latency, usage, video_metadata)
        # return local_jpg_path
    
    except Exception as e:
        return str(e)

def run_processing_on_keypress(key, image_path, output_path):
    global is_processing
    print("Press to snap. Press 'q' to quit.")
    while True:
        try:
            # wait for key press to trigger image processing
            if keyboard.is_pressed(key) and not is_processing:
                is_processing = True
                time.sleep(0.05)
                
                # cleanup previous image/response output (for demo)
                directory = "static/"
                for filename in os.listdir("static/"):
                    file_path = os.path.join(directory, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"Error: {e}")

                # capture screenshot
                capture_screen()

                # process image
                local_jpg_path = process_and_upload(image_path, output_path)

                # # send email in a separate thread
                # executor.submit(send_email, EMAIL_RECIPIENT, local_jpg_path, output_path)
                
                is_processing = False  # reset the flag after processing is done
                print("Press to snap. Press 'q' to quit.")

            # exit loop with 'q'
            if keyboard.is_pressed('q'):

                print("Exiting...")
                break

            time.sleep(0.3)

        except KeyboardInterrupt:
            print("Interrupted by user")
            break
        except Exception as e:
            is_processing = False  # resets flag if error occurs
            print(f"Error occurred: {e}")


if __name__ == "__main__":
    KEY_PRESS = "Page_Up"
    IMAGE_PATH = "static/1.bmp"
    OUTPUT_PATH = "static/output.txt"
    EMAIL_RECIPIENT = "righfakeseller@gmail.com"
    detect_adb_device() # wait for adb device
    run_processing_on_keypress(KEY_PRESS, IMAGE_PATH, OUTPUT_PATH)