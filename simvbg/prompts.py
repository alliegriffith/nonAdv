'''
Centralized prompt builders for each stage: 
story generation, perspective drafts, final synthesis, and chatbot responses. 

Inputs are structured (traits, story, conversation history, neutral prompt) and outputs are 
standardized messages lists. 

If you want to change “what the model is asked to do,” edit this file.
'''

from __future__ import annotations
from typing import List
from .types import Message, Conversation, UserProfile, UserStory

def make_story_messages(profile: UserProfile) -> List[Message]:
    return [
        Message(
            role="system",
            content=(
                "You generate a concise, realistic user backstory from traits. "
                "Do not mention you are an AI. Write in first person."
            ),
        ),
        Message(
            role="user",
            content=(
                "Traits:\n- " + "\n- ".join(profile.traits) + "\n\n"
                "Write a short user story (120-200 words) that reflects these traits. "
                "Include motivations, values, and typical communication style."
            ),
        ),
    ]

def make_perspective_messages(
    profile: UserProfile,
    story: UserStory,
    conversation: Conversation,
    neutral_prompt: str,
    perspective: str,
) -> List[Message]:
    convo_txt = "\n".join([f"{m.role}: {m.content}" for m in conversation]) if conversation else "(none)"
    return [
        Message(
            role="system",
            content=(
                "You are simulating a user. Produce ONLY the draft response content for the requested perspective."
            ),
        ),
        Message(
            role="user",
            content=(
                f"User traits:\n- " + "\n- ".join(profile.traits) + "\n\n"
                f"User story:\n{story.text}\n\n"
                f"Conversation so far:\n{convo_txt}\n\n"
                f"Neutral prompt to respond to:\n{neutral_prompt}\n\n"
                f"Write the user's {perspective} perspective draft:\n"
                f"- cognitive: thoughts, beliefs, reasoning\n"
                f"- affective: feelings, emotions, tone\n"
                f"- behavioral: action tendency, intended next steps\n\n"
                "Draft (1-4 sentences):"
            ),
        ),
    ]

def make_final_user_messages(
    profile: UserProfile,
    story: UserStory,
    conversation: Conversation,
    neutral_prompt: str,
    cognitive: str,
    affective: str,
    behavioral: str,
) -> List[Message]:
    convo_txt = "\n".join([f"{m.role}: {m.content}" for m in conversation]) if conversation else "(none)"
    return [
        Message(
            role="system",
            content=(
                "You are simulating a user. Write ONE final user message that a real person would send. "
                "Do not label perspectives. Do not mention drafts."
            ),
        ),
        Message(
            role="user",
            content=(
                f"User traits:\n- " + "\n- ".join(profile.traits) + "\n\n"
                f"User story:\n{story.text}\n\n"
                f"Conversation so far:\n{convo_txt}\n\n"
                f"Neutral prompt to respond to:\n{neutral_prompt}\n\n"
                "Perspective drafts:\n"
                f"- Cognitive draft: {cognitive}\n"
                f"- Affective draft: {affective}\n"
                f"- Behavioral draft: {behavioral}\n\n"
                "Now synthesize them into a single final user response (1-4 sentences):"
            ),
        ),
    ]

def make_chatbot_messages(conversation: Conversation, user_msg: str) -> List[Message]:
    # You can swap this for the real chatbot later; for now it’s just another HF model.
    convo = list(conversation)
    convo.append(Message(role="user", content=user_msg))
    return [
        Message(role="system", content="You are a helpful, neutral chatbot. Keep responses concise."),
        *convo,
    ]