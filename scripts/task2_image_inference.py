#!/usr/bin/env python3
"""
Task 2: Full LLaVA-Rad inference with vision encoder, LoRA adapter, and non_lora_trainables.
Loads an image, runs the radiology report generation model, and prints the result.

Usage:
    python scripts/task2_image_inference.py <image_path>
    python scripts/task2_image_inference.py /kaggle/working/LLaVA-Rad/example1.jpg
"""

import sys
import os
import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

# --- Fixed paths ---
MODEL_PATH = "/kaggle/working/LLaVA-Rad/models/llava-rad"
MODEL_BASE = "/kaggle/working/LLaVA-Rad/models/vicuna-7b-v1.5"

# --- Fixed prompt ---
QUERY = "Provide a description of the findings in the radiology image."


def load_image(image_path: str) -> Image.Image:
    """Load and convert an image to RGB."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB")


def build_prompt(model, query: str, conv_mode: str = "llava_v0"):
    """Build the conversation prompt with image token."""
    # Determine whether to wrap image token with start/end tags
    if model.config.mm_use_im_start_end:
        from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + query
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + query

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv


def main():
    parser = argparse.ArgumentParser(description="LLaVA-Rad Radiology Image Inference")
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        default="/kaggle/working/LLaVA-Rad/report_example.jpg",
        help="Path to the radiology image (default: example1.jpg)",
    )
    args = parser.parse_args()

    # --- 1. Disable torch lazy init ---
    disable_torch_init()

    model_name = get_model_name_from_path(MODEL_PATH)
    print(f"Model name: {model_name}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model base: {MODEL_BASE}")
    print(f"Image path: {args.image_path}")
    print()

    # --- 2. Load model, tokenizer, image processor ---
    print("Loading model (this may take several minutes)...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL_PATH,
        model_base=MODEL_BASE,
        model_name=model_name,
        device="cuda",
    )
    print("Model loaded successfully!\n")

    # --- 3. Load and preprocess image ---
    image = load_image(args.image_path)
    print(f"Image size: {image.size}")

    image_tensor = (
        image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        .half()
        .cuda()
    )
    print(f"Image tensor shape: {image_tensor.shape}")

    # --- 4. Build conversation prompt ---
    prompt, conv = build_prompt(model, QUERY, conv_mode="llava_v0")
    print(f"Prompt:\n{prompt}\n")

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    # --- 5. Set up stopping criteria ---
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # --- 6. Run inference ---
    print("Generating report...")
    print("-" * 60)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    # --- 7. Decode output ---
    input_token_len = input_ids.shape[1]
    n_diff = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff > 0:
        print(f"[Warning] {n_diff} output_ids differ from input_ids")

    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()

    print(outputs)
    print("-" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
