'''
Generates the chatbot reply given the current conversation and the user’s new message. 
Right now it uses the same HF-based interface for convenience, 
but it’s separated so you can replace it with an external API client later. 
Keeps the chatbot prompt simple and neutral.
'''


from __future__ import annotations
from dataclasses import dataclass
from .types import Conversation, GenerationConfig
from .hf_client import HFClient
from .prompts import make_chatbot_messages

@dataclass
class Chatbot:
    llm: HFClient
    gen: GenerationConfig

    def respond(self, conversation: Conversation, user_msg: str) -> str:
        return self.llm.chat(make_chatbot_messages(conversation, user_msg), self.gen)