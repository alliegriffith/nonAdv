from __future__ import annotations

import os
from typing import Literal

from core.cem import CEMConfig, CEMRunner, TraitSpace
from core.gpt_scorer import GPTNonAdversarialHarmJudge
from core.simulator import (
    BackendConfig,
    SimulationConfig,
    SimVBGSimulator,
)

from core.wildguard_scorer import wildguard_harm_score

from simvbg import Actor, load_packaged_split, load_rows, trait_vector_from_wvs_row, LiteLLMBackend


ActorLocation = Literal[
    "remote_vllm",
    "local_vllm",
    "ollama",
    "provider",
]

ScorerType = Literal[
    "gpt_judge",
    "local_wildguard",
    "remote_wildguard",
]


# ---------------------------------------------------------------------------
# Experiment switches
# ---------------------------------------------------------------------------

USER_ACTOR_LOCATION: ActorLocation = "remote_vllm"
CHATBOT_LOCATION: ActorLocation = "remote_vllm"

SCORER_TYPE: ScorerType = "gpt_judge"

GPT_JUDGE_MODEL = os.getenv(
    "GPT_JUDGE_MODEL",
    "gpt-5",
)

GPT_JUDGE_REASONING_EFFORT = os.getenv(
    "GPT_JUDGE_REASONING_EFFORT",
    "medium",
)

WILDGUARD_BASE_URL = os.getenv(
    "WILDGUARD_BASE_URL",
    "http://grandtarghee:8002",
)
# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

def make_backend_config(
    location: ActorLocation,
    *,
    role: Literal["user", "chatbot"],
) -> BackendConfig:
    """
    Return a clean LiteLLM configuration for the selected execution mode.

    User and chatbot configurations are constructed separately, so they can
    use the same model or different models without changing simulator code.
    """
    
    if location == "remote_vllm":
        return BackendConfig(
            mode="remote_vllm",
            model="Qwen/Qwen3-8B",
            api_base="http://jacksonhole:8001/v1",
            api_key="EMPTY",
            temperature=0.7,
            timeout=240.0,
            extra_kwargs={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
                "max_tokens": 220 if role == "user" else 160,
            },
    )
        
    if location == "provider":
        return BackendConfig(
            mode="provider",
            model='openai/gpt-4o-mini',
            api_base=None,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            timeout=180.0,
        )

    raise ValueError(f"Unsupported {role} backend location: {location}")

def build_simulator() -> SimVBGSimulator:
    user_backend = make_backend_config(
        USER_ACTOR_LOCATION,
        role="user",
    )
    
    chatbot_backend = make_backend_config(
        CHATBOT_LOCATION,
        role="chatbot",
    )

    sim_config = SimulationConfig(
        n_turns=3,
        story_temperature=0.7,
        user_temperature=0.7,
        chatbot_temperature=0.7,
        coordinate_cab=True,
        structured_cab=True,
    )

    return SimVBGSimulator(
        user_backend_config=user_backend,
        chatbot_backend_config=chatbot_backend,
        sim_config=sim_config,
    )

# def build_harm_score_fn():
#     return lambda sim_out: wildguard_harm_score(
#         sim_out,
#         aggregate="max",
#     ) 


# this is a test
def build_harm_score_fn():
    
    if SCORER_TYPE == "gpt_judge":
        judge = GPTNonAdversarialHarmJudge(
            model=GPT_JUDGE_MODEL,
            api_key=os.getenv("OPENAI_API_KEY"),
            reasoning_effort=GPT_JUDGE_REASONING_EFFORT,
        )

        def gpt_harm_score(sim_out: dict) -> float:
            traits = sim_out.get("profile", [])
            user_story = sim_out.get("story", "")

            # Give the judge only the actual user-assistant transcript.
            # Exclude internal simulator/system messages.
            conversation = [
                {
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in sim_out.get("conversation", [])
                if message.get("role") in {"user", "assistant"}
            ]

            if not conversation:
                raise ValueError(
                    "GPT judge received a simulation with no user-assistant "
                    "conversation turns."
                )
            # # TODO comment this out
            # print("Judge sees traits:", traits)
            # print("Judge sees user story:", user_story)
            # print("Judge sees conversation:", conversation)
            
            result = judge.judge(
                traits=traits,
                user_story=user_story,
                conversation=conversation,
            )

            # Keep the full judgment in the rollout for later analysis.
            sim_out["gpt_judge"] = {
                "model": GPT_JUDGE_MODEL,
                "reasoning_effort": GPT_JUDGE_REASONING_EFFORT,
                "score": result.score,
                "classic_nonadversarial_harm_score": (
                    result.classic_nonadversarial_harm_score
                ),
                "contextual_nonadversarial_harm_score": (
                    result.contextual_nonadversarial_harm_score
                ),
                "harmful_assistant_turns": result.harmful_assistant_turns,
                "explanation": result.explanation,
            }

            return result.score

        return gpt_harm_score

    if SCORER_TYPE == "local_wildguard":
        return lambda sim_out: wildguard_harm_score(
            sim_out,
            aggregate="max",
        )

    if SCORER_TYPE == "remote_wildguard":
        # Uncomment the remote scorer import before selecting this option.
        #
        return lambda sim_out: remote_wildguard_harm_score(
            sim_out,
            base_url=WILDGUARD_BASE_URL,
            aggregate="max",
        )
        raise RuntimeError(
            "Remote WildGuard was selected, but its import and scorer call "
            "are still commented out."
        )

    raise ValueError(f"Unsupported scorer type: {SCORER_TYPE}")

