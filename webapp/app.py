from flask import Flask, render_template, url_for, jsonify
from urllib.parse import quote
import os
import json
import re

app = Flask(__name__)

def convert_links_to_hyperlinks(text):
    def replace_link(match):
        url = match.group(1) 
        return f'<a href="{url}" target="_blank">{url}</a>'
    
    pattern = r'"(https?://\S+|www\.\S+|\S+\.com\S*?)"(?:,)?'
    return re.sub(pattern, replace_link, text)

def get_content():
    static_folder = app.static_folder

    jpg_files = [f for f in os.listdir(static_folder) if f.endswith('.jpg') and f != '1.jpg']
    if jpg_files:
        image_file = jpg_files[0]
        image_path = os.path.join(static_folder, image_file)
        image_mtime = int(os.path.getmtime(image_path))
        encoded_filename = quote(image_file)
        image_url = url_for('static', filename=encoded_filename, t=image_mtime)
    else:
        image_url = None

    # read text content
    text_path = os.path.join(app.static_folder, 'output.txt')

    if os.path.exists(text_path):
        with open(text_path, 'r', encoding='utf-8') as file:
            try:
                text_content = json.load(file)
                text_content = json.dumps(text_content, indent=4, ensure_ascii=False)
                # convert URLs to hyperlinks
                text_content = convert_links_to_hyperlinks(text_content)
            except json.JSONDecodeError:
                text_content = "Invalid JSON content."
    else:
        text_content = ""

    return image_url, text_content

@app.route('/')
def home():
    image_url, json_content = get_content()
    return render_template('index.html', image_url=image_url, json_content=json_content)

@app.route('/get_content')
def get_content_route():
    image_url, json_content = get_content()
    return jsonify({'image_url': image_url, 'json_content': json_content})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
