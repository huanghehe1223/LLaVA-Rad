#!/usr/bin/env python3
"""
Task 1: LLM-only multi-turn interactive chat with streaming output.
Uses ONLY the Vicuna-7B-v1.5 base model — no vision encoder, no LoRA adapter.
"""

import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# --- Paths ---
LLM_PATH = "/kaggle/working/LLaVA-Rad/models/vicuna-7b-v1.5"

# --- Vicuna v1.5 conversation template ---
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def build_prompt(history: list[tuple[str, str]], new_user_msg: str) -> str:
    """
    Build the full prompt string for Vicuna v1.5.
    history: list of (user_msg, assistant_msg) from previous turns.
    """
    parts = [SYSTEM_PROMPT]
    for user_msg, assistant_msg in history:
        parts.append(f"USER: {user_msg} ASSISTANT: {assistant_msg}</s>")
    parts.append(f"USER: {new_user_msg} ASSISTANT:")
    return " ".join(parts)


def load_model():
    """Load the base Vicuna-7B-v1.5 model and tokenizer."""
    print(f"Loading tokenizer from {LLM_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, use_fast=False)

    # Set pad token if not present (LLaMA tokenizer often lacks it)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    print(f"Loading model from {LLM_PATH} (this may take a while) ...")
    model = AutoModelForCausalLM.from_pretrained(
        LLM_PATH,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    print("Model loaded successfully!\n")
    return model, tokenizer


def generate_stream(model, tokenizer, prompt: str, max_new_tokens: int = 512):
    """Generate response with streaming output."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    # Decode only the new tokens (skip prompt)
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


def main():
    print("=" * 60)
    print("  Vicuna-7B-v1.5 LLM-only Multi-turn Chat")
    print("  (No vision encoder, No LoRA — pure text)")
    print("  Type /exit to quit, /clear to reset history")
    print("=" * 60)
    print()

    model, tokenizer = load_model()
    history: list[tuple[str, str]] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            print("Goodbye!")
            break

        if user_input.lower() == "/clear":
            history.clear()
            print("[History cleared]\n")
            continue

        # Build prompt with full history
        prompt = build_prompt(history, user_input)

        # Stream generation
        print("Assistant: ", end="", flush=True)
        response = generate_stream(model, tokenizer, prompt)
        print()  # extra newline after streaming

        # Save turn to history
        history.append((user_input, response))


if __name__ == "__main__":
    main()
