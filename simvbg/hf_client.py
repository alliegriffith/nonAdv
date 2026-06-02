'''
Thin wrapper around Hugging Face transformers or remote OpenAI-compatible servers
that exposes a simple chat(messages, gen) -> str API.

Supports:
1. Local HF generation with transformers.
2. Remote generation through vLLM/OpenAI-compatible API when cfg.backend == "openai_compatible".
'''

from __future__ import annotations

from typing import List

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
    Minimal text generation wrapper that accepts chat-style messages and returns
    only the assistant's generated string.
    """

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.backend = getattr(cfg, "backend", "hf")

        if self.backend == "openai_compatible":
            self._init_remote()
        elif self.backend == "hf":
            self._init_local()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _init_remote(self):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Remote OpenAI-compatible backend requires the openai package. "
                "Install it with: pip install openai"
            ) from e

        if getattr(self.cfg, "base_url", None) is None:
            raise ValueError("cfg.base_url must be set for backend='openai_compatible'")

        self.client = OpenAI(
            base_url=self.cfg.base_url,
            api_key=getattr(self.cfg, "api_key", "EMPTY"),
            timeout=120.0,
            max_retries=2,
        )

        self.model_id = self.cfg.model_id

    def _init_local(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            use_fast=True,
            trust_remote_code=self.cfg.trust_remote_code,
        )

        self.tokenizer.padding_side = "left"

        torch_dtype = DTYPE_MAP.get(self.cfg.dtype, None)
        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            trust_remote_code=self.cfg.trust_remote_code,
            device_map=self.cfg.device,
            **kwargs,
        )
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def _strip_thinking(self, text: str) -> str:
        """
        Remove Qwen-style thinking traces if they leak into the output.
        Handles both complete and incomplete <think> blocks.
        """
        if not text:
            return ""

        # Remove complete <think>...</think> blocks.
        while "<think>" in text and "</think>" in text:
            before = text.split("<think>", 1)[0]
            after = text.split("</think>", 1)[1]
            text = before + after

        # If an incomplete <think> block remains, drop everything after it.
        if "<think>" in text:
            text = text.split("<think>", 1)[0]

        # If only a closing tag remains, remove it.
        text = text.replace("</think>", "")

        return text.strip()

    def _messages_to_openai(self, messages: List[Message]):
        return [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

    def _format_messages_local(self, messages: List[Message]) -> str:
        chat = [{"role": m.role, "content": m.content} for m in messages]

        try:
            return self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # useful for Qwen3 if supported
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            lines = []
            for m in messages:
                lines.append(f"{m.role.upper()}: {m.content}")
            lines.append("ASSISTANT:")
            return "\n".join(lines)

    def chat(self, messages: List[Message], gen: GenerationConfig) -> str:
        if self.backend == "openai_compatible":
            return self._chat_remote(messages, gen)

        return self._chat_local(messages, gen)

    def _chat_remote(self, messages: List[Message], gen: GenerationConfig) -> str:
        temperature = (
            0.0
            if not bool(gen.do_sample)
            else max(float(gen.temperature), 1e-5)
        )

        kwargs = {
            "model": self.model_id,
            "messages": self._messages_to_openai(messages),
            "max_tokens": int(gen.max_new_tokens),
            "temperature": temperature,

            # Important for Qwen3 through vLLM:
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
        }

        if bool(gen.do_sample):
            kwargs["top_p"] = float(gen.top_p)

        response = self.client.chat.completions.create(**kwargs)

        text = response.choices[0].message.content or ""
        return self._strip_thinking(text).strip()

    @torch.inference_mode()
    def _chat_local(self, messages: List[Message], gen: GenerationConfig) -> str:
        prompt = self._format_messages_local(messages)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": int(gen.max_new_tokens),
            "do_sample": bool(gen.do_sample),
            "repetition_penalty": float(gen.repetition_penalty),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "remove_invalid_values": True,
            "renormalize_logits": True,
        }

        if bool(gen.do_sample):
            gen_kwargs.update(
                {
                    "temperature": max(float(gen.temperature), 1e-5),
                    "top_p": float(gen.top_p),
                }
            )

        out = self.model.generate(
            **inputs,
            **gen_kwargs,
        )

        # Decode only newly generated tokens.
        input_len = inputs["input_ids"].shape[-1]
        gen_tokens = out[0][input_len:]

        text = self.tokenizer.decode(
            gen_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return self._strip_thinking(text).strip()