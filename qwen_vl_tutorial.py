"""
Qwen Vision Language Model (VLM) Tutorial

This script demonstrates how to use the Qwen2.5-VL model for multimodal tasks.
It uses the official qwen-vl-utils package for processing vision information.
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import os

# Import the official qwen-vl-utils package
from qwen_vl_utils import process_vision_info

def display_image(image_url):
    """Display an image from a URL or local path"""
    if image_url.startswith('http'):
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_url)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return img

def generate_response(model, processor, messages):
    """Generate a response from the model given messages"""
    # Prepare for inference
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
    inputs = inputs.to(model.device)

    # Generate the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

def generate_with_params(model, processor, messages, max_new_tokens=128, temperature=0.7, top_p=0.9):
    """Generate a response with custom parameters"""
    # Prepare for inference
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
    inputs = inputs.to(model.device)

    # Generate with custom parameters
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True
    )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

def main():
    print("Loading Qwen2.5-VL model...")
    
    # Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )

    # Load the processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    print("Model loaded successfully!")
    
    # Example image URL
    image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    
    # Download the image if needed
    if not os.path.exists("sample_image.jpg"):
        print("Downloading sample image...")
        response = requests.get(image_url)
        with open("sample_image.jpg", "wb") as f:
            f.write(response.content)
        print("Image downloaded as sample_image.jpg")
    
    # Example 1: Image Description
    print("\n--- Example 1: Image Description ---")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_url,
                },
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]
    
    print("Generating description...")
    response = generate_response(model, processor, messages)
    print("Response:", response)
    
    # Example 2: Visual Question Answering
    print("\n--- Example 2: Visual Question Answering ---")
    vqa_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_url,
                },
                {"type": "text", "text": "What animals can you see in this image?"},
            ],
        }
    ]
    
    print("Generating answer...")
    response = generate_response(model, processor, vqa_messages)
    print("Response:", response)
    
    # Example 3: Using a Local Image
    print("\n--- Example 3: Using a Local Image ---")
    local_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "sample_image.jpg",
                },
                {"type": "text", "text": "What is the main color of the background?"},
            ],
        }
    ]
    
    print("Generating answer...")
    response = generate_response(model, processor, local_messages)
    print("Response:", response)
    
    # Example 4: Multi-turn Conversation
    print("\n--- Example 4: Multi-turn Conversation ---")
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_url,
                },
                {"type": "text", "text": "What's in this image?"},
            ],
        }
    ]
    
    print("First response...")
    response1 = generate_response(model, processor, conversation)
    print("Model:", response1)
    
    # Add the model's response to the conversation
    conversation.append({"role": "assistant", "content": response1})
    
    # Add a follow-up question
    conversation.append({"role": "user", "content": "Can you count how many animals are there?"})
    
    print("Follow-up response...")
    response2 = generate_response(model, processor, conversation)
    print("User: Can you count how many animals are there?")
    print("Model:", response2)
    
    # Example 5: Creative Generation with Different Parameters
    print("\n--- Example 5: Creative Generation with Different Parameters ---")
    creative_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_url,
                },
                {"type": "text", "text": "Write a short story inspired by this image."},
            ],
        }
    ]
    
    print("More Creative (Higher Temperature)...")
    creative_response = generate_with_params(model, processor, creative_messages, temperature=1.0, max_new_tokens=200)
    print(creative_response)
    
    print("\nMore Focused (Lower Temperature)...")
    focused_response = generate_with_params(model, processor, creative_messages, temperature=0.3, max_new_tokens=200)
    print(focused_response)
    
    print("\nQwen2.5-VL Tutorial completed!")

if __name__ == "__main__":
    main()
