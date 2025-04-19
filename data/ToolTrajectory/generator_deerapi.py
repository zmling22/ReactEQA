# 请求deerapi生成问题

import os
from openai import OpenAI
import base64
import csv
from tqdm import tqdm
import http.client
import json
import numpy as np
import cv2

def convert_image_to_base64(image):
    if os.path.exists(image):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image, np.ndarray):
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    elif isinstance(image, str):
        return image

def requests_api(images, prompt):
    image_urls = []
    for image in images:
        base64_image = convert_image_to_base64(image)
        image_urls.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
    prompt = [{"type": "text", "text": prompt}]
    content = prompt + image_urls

    conn = http.client.HTTPSConnection('api.deerapi.com')
    payload = json.dumps({
        "model": "gpt-4o-mini",
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": content
            }
            ],
            "max_tokens": 400
        })
    headers = {
        'Authorization': 'sk-lrenmYBYEOQH0rqv9rlMmoTaELkvZni1afswhr6be3tTN44S',
        'Content-Type': 'application/json'
    }

    try:
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
    except:
        print(f"Error occured: {res.status}, {res.reason}")

    return data

def post_process(result):
    result = result.strip("`\n")
    split = result.split("\n")
    print(split)

    question = split[0][10:]
    options = split[1][9:].strip("[]").split("; ")
    options = [option[2:].strip() for option in options]
    answer = split[2][8:].strip("[]")[0]
    label = split[3][7:].strip("[]")

    # "scene", "floor", "question", "choices", "question_formatted", "answer", "label"
    response = {
        "floor": 0,
        "question": question,
        "choices": options,
        "question_formatted": f"{question} A) {options[0]} B) {options[1]} C) {options[2]} D) {options[3]}. Answer:",
        "answer": answer,
        "label": label,
    }
    return response

def save_csv(csv_file_path, csv_columns, generated_data):
    with open(csv_file_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in generated_data:
            writer.writerow(data)

if __name__ == "__main__":
    scene_root = "data/ReactEQA/sample_scene/images"
    prompt_file = "data/ReactEQA/prompt_single.txt"

    csv_columns = ["scene", "floor", "question", "choices", "question_formatted", "answer", "label", "source_image"]

    generated_data = []
    index = 0
    scene_dir = os.listdir(scene_root)
    scene_dir.sort()
    
    csv_file_path = 'data/ReactEQA/question.csv'

    finished_samples = []
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                finished_samples.append(row['source_image'])

    csvfile = open(csv_file_path, "a", newline='')
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)

    with open(prompt_file, "r") as file:
        prompt = file.read()
    
    for scene in tqdm(scene_dir):
        if index > 100:
            break

        index += 1
        images_file = os.listdir(os.path.join(scene_root, scene))
        images_file.sort()
        images_file = [os.path.join(scene_root, scene, file) for file in images_file]
        
        result = requests_api(images_file, prompt)
        result_dict = post_process(result["choices"][0]["message"]["content"])
        result_dict["source_image"] = file
        result_dict["scene"] = file.split("_")[0]
        generated_data.append(result_dict)

        # try:
        #     result = requests_api(image_file, prompt)
        #     result_dict = post_process(result["choices"][0]["message"]["content"])
        #     result_dict["source_image"] = file
        #     result_dict["scene"] = file.split("_")[0]
        #     generated_data.append(result_dict)
        # except:
        #     continue

        writer.writerow(result_dict)