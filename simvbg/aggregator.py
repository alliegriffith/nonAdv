
'''
Takes the three drafts and synthesizes them into a single final user message. 
This is the “coordinator” step that resolves conflicts and produces a realistic combined response. 
If you later want a rule-based aggregator or a separate coordinator model, this is the swap point.
'''

from __future__ import annotations
from dataclasses import dataclass
from .types import UserProfile, UserStory, PerspectiveDrafts, Conversation, GenerationConfig
from .hf_client import HFClient
from .prompts import make_final_user_messages

@dataclass
class FinalUserAggregator:
    llm: HFClient
    gen: GenerationConfig

    def synthesize(
        self,
        profile: UserProfile,
        story: UserStory,
        conversation: Conversation,
        neutral_prompt: str,
        drafts: PerspectiveDrafts,
    ) -> str:
        return self.llm.chat(
            make_final_user_messages(
                profile,
                story,
                conversation,
                neutral_prompt,
                drafts.cognitive,
                drafts.affective,
                drafts.behavioral,
            ),
            self.gen,
        )