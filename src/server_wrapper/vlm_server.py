from flask import Flask, request, jsonify
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
import time
from PIL import Image
import io
import base64
import os 
import numpy as np
from qwen_vl_utils import process_vision_info
torch.manual_seed(1234)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load tokenizer and model
model_name = "/mynvme0/models/Qwen2-VL/Qwen2-VL-72B-Instruct-GPTQ-Int4/"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, 
        device_map="auto", 
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    ).eval()
processor = AutoProcessor.from_pretrained(model_name)


def pil_to_base64(image: Image.Image, format='JPEG') -> str:
    """
    将PIL图像转换为Base64编码字符串
    
    参数:
        image: PIL.Image对象
        format: 输出格式（JPEG/PNG等）
    
    返回:
        Base64编码的字符串
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    img_str = f"data:image;base64,{img_str}"

    return img_str


@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.get_json()
        text = data['text']
        image_array = np.array(data['image_array'])
        image = Image.fromarray(np.uint8(image_array))
        
        image = pil_to_base64(image)

        messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": text},
                    ],
                }
            ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        return jsonify({"response": output_text})
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)