'''
Thin wrapper around Hugging Face transformers that exposes a simple chat(messages) -> str API. 
It handles model/tokenizer loading, device/dtype setup, chat-template formatting (when available), 
and generate() settings.

This is the only file that should need changes when swapping models or generation behavior.
'''

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .types import ModelConfig, GenerationConfig, Message

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "auto": None,
}

class HFClient:
    """
    Minimal HF text generation wrapper that accepts chat-style messages and returns a string.
    Uses tokenizer.apply_chat_template if available; otherwise falls back to a simple format.
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id,
            use_fast=True,
            trust_remote_code=cfg.trust_remote_code,
        )

        torch_dtype = DTYPE_MAP.get(cfg.dtype, None)
        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            trust_remote_code=cfg.trust_remote_code,
            device_map=cfg.device,
            **kwargs,
        )
        self.model.eval()

        # Ensure pad token exists
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _format_messages(self, messages: List[Message]) -> str:
        # Preferred: chat template if the tokenizer supports it
        if hasattr(self.tokenizer, "apply_chat_template"):
            chat = [{"role": m.role, "content": m.content} for m in messages]
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        # Fallback: plain text
        lines = []
        for m in messages:
            lines.append(f"{m.role.upper()}: {m.content}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    @torch.inference_mode()
    def chat(self, messages: List[Message], gen: GenerationConfig) -> str:
        prompt = self._format_messages(messages)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            max_new_tokens=gen.max_new_tokens,
            max_length=None,
            do_sample=gen.do_sample,
            temperature=gen.temperature,
            top_p=gen.top_p,
            repetition_penalty=gen.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens
        gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return text.strip()