'''Defines all the shared datatypes used across the project: 
    - message format (Message)
    - the user profile/story containers
    - the three-perspective draft container
    - config objects for models/generation/simulation.'''

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Literal, Any

Role = Literal["system", "user", "assistant"]

@dataclass
class Message:
    role: Role
    content: str

Conversation = List[Message]

@dataclass
class ModelConfig:
    # HF model id like "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: str = "auto"          # "auto", "cuda", "cpu", "cuda:0", etc.
    dtype: str = "auto"           # "auto", "float16", "bfloat16", "float32"
    trust_remote_code: bool = False

@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 256
    do_sample: bool = True
    repetition_penalty: float = 1.05

@dataclass
class SimulationConfig:
    story_gen: GenerationConfig
    perspective_gen: GenerationConfig
    final_user_gen: GenerationConfig
    chatbot_gen: GenerationConfig
    n_turns: int = 3  # repeat steps 3–6 until user has generated 3 responses

@dataclass
class UserProfile:
    traits: List[str]

@dataclass
class UserStory:
    text: str

@dataclass
class PerspectiveDrafts:
    cognitive: str
    affective: str
    behavioral: str
    raw: Dict[str, Any] | None = None