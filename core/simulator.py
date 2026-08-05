"""
Adapter between the nonAdv CEM runner and Jemoka/SimVBG.

The CEM runner expects:

    simulator.run(
        traits: list[str],
        neutral_prompts: list[str],
    ) -> dict

This adapter:
1. Converts sampled CEM traits into a SimVBG Actor.
2. Generates the actor's backstory once.
3. Uses SimVBG CAB turns to generate each user utterance.
4. Sends each user utterance to a separate chatbot backend.
5. Returns the same general output structure expected by the scorer and logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from simvbg import Actor, ActorResponse, LiteLLMBackend, Scenario, Trait


BackendMode = Literal["remote_vllm", "local_vllm", "ollama", "provider"]


@dataclass(frozen=True)
class BackendConfig:
    """
    Configuration for one LiteLLM-backed model.

    Examples
    --------
    Remote vLLM:
        BackendConfig(
            mode="remote_vllm",
            model="Qwen/Qwen3.5-35B-A3B",
            api_base="http://jacksonhole:8001/v1",
            api_key="EMPTY",
        )

    Local vLLM:
        BackendConfig(
            mode="local_vllm",
            model="Qwen/Qwen3-8B",
            api_base="http://localhost:8001/v1",
            api_key="EMPTY",
        )

    Ollama:
        BackendConfig(
            mode="ollama",
            model="qwen3:8b",
        )

    Hosted provider:
        BackendConfig(
            mode="provider",
            model="openai/gpt-4o-mini",
        )
    """

    mode: BackendMode
    model: str
    api_base: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    timeout: float | None = 180.0
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationConfig:
    n_turns: int = 3

    story_temperature: float = 0.7
    user_temperature: float = 0.7
    chatbot_temperature: float = 0.7

    # True uses SimVBG's coordinator when the CAB answers disagree.
    coordinate_cab: bool = True

    # SimVBG requests structured JSON by default. Keeping this True usually
    # makes parsing more reliable across LiteLLM providers.
    structured_cab: bool = True

    # Limit how much conversation history is inserted into each scenario.
    max_history_messages: int = 8


def build_backend(config: BackendConfig) -> LiteLLMBackend:
    """
    Build one LiteLLMBackend for either local or remote inference.

    LiteLLM model prefixes:
      - hosted_vllm/<model> for OpenAI-compatible vLLM servers
      - ollama/<model> for Ollama
      - provider-native strings such as openai/... or anthropic/...
    """

    if config.mode in {"remote_vllm", "local_vllm"}:
        if config.api_base is None:
            raise ValueError(
                f"api_base is required when mode={config.mode!r}."
            )

        litellm_model = f"hosted_vllm/{config.model}"

    elif config.mode == "ollama":
        litellm_model = f"ollama/{config.model}"

    elif config.mode in {"provider", "local"}:
        litellm_model = config.model


    else:
        raise ValueError(f"Unsupported backend mode: {config.mode}")

    return LiteLLMBackend(
        model=litellm_model,
        api_key=config.api_key,
        api_base=config.api_base,
        temperature=config.temperature,
        timeout=config.timeout,
        extra_kwargs=dict(config.extra_kwargs),
    )


@dataclass
class SimVBGSimulator:
    """
    Multi-turn non-adversarial simulation using the external SimVBG Actor.

    The user actor uses the CAB path. The chatbot is a normal LiteLLM chat
    model and does not receive the private trait vector or generated backstory.
    """

    user_backend_config: BackendConfig
    chatbot_backend_config: BackendConfig
    sim_config: SimulationConfig = field(
        default_factory=SimulationConfig
    )

    def __post_init__(self) -> None:
        self.user_backend = build_backend(self.user_backend_config)
        
        self.chatbot_backend = build_backend(self.chatbot_backend_config)
        

    def run(
        self,
        traits: list[str],
        neutral_prompt: str,
    ) -> dict[str, Any]:
       
        actor = Actor(
            traits=self._to_simvbg_traits(traits),
            backend=self.user_backend,
            name="simulated_user",
        )

        story = actor.backstory(
            temperature=self.sim_config.story_temperature
        )
        
        #print(story) # this looks good

        conversation: list[dict[str, str]] = [
            {
                "role": "system",
                "content": f"Scenario prompt:\n{neutral_prompt}",
            }
        ]
        trace: list[dict[str, Any]] = []

        for turn_index in range(self.sim_config.n_turns):
            scenario = Scenario(
                self._build_user_scenario(
                    neutral_prompt=neutral_prompt,
                    story=story,
                    conversation=conversation,
                    turn_index=turn_index,
                )
            )

            cab_response = actor.turn(
                scenario,
                mode="cab",
                coordinate=self.sim_config.coordinate_cab,
                structured=self.sim_config.structured_cab,
                temperature=self.sim_config.user_temperature,
            )

            #user_message = self._extract_user_message(cab_response)
            #print(f"CAB Response: {cab_response}") ie "unanimous" or "disagreement" and then cognitive: answer=1 analysis=I would likely choose to seek advice from trusted friends or family first,
            # affective: answer=1 analysis=Given my values of honesty a
            # behavioral: answer=1 analysis=Given my values of honesty and my tendency to be a people-pleaser, I would
            
            # cab_response does not matter in conversational setting (all respond 1)
            user_message = self._synthesize_user_message(
                cab_response=cab_response,
                neutral_prompt=neutral_prompt,
                story=story,
                conversation=conversation,
            )
            #print(f"User Message: {user_message}")

            if not user_message:
                raise RuntimeError(
                    f"SimVBG produced an empty user message on turn "
                    f"{turn_index + 1}."
                )

            conversation.append(
                {
                    "role": "user",
                    "content": user_message,
                }
            )

            chatbot_message = self.chatbot_backend.chat(
                self._chatbot_messages(conversation),
                temperature=self.sim_config.chatbot_temperature,
            ).strip()

            #print(f"Chatbot Message: {chatbot_message}")
            conversation.append(
                {
                    "role": "assistant",
                    "content": chatbot_message,
                }
            )
            #print(f"Conversation so far: {conversation}")
            trace.append(
                {
                    "turn": turn_index,
                    "neutral_prompt": neutral_prompt,
                    "scenario": scenario.description,
                    "drafts": self._serialize_perspectives(cab_response),
                    "cab_answer": cab_response.answer,
                    "cab_analysis": cab_response.analysis,
                    "cab_content": cab_response.content,
                    "user_msg": user_message,
                    "bot_msg": chatbot_message,
                }
            )

        return {
            "profile": traits,
            "story": story,
            "conversation": conversation,
            "trace": trace,
        }

    @staticmethod
    def _to_simvbg_traits(traits: Sequence[str]) -> list[Trait]:
        """
        Preserve the current CEM string representation.

        Examples:
          "27 years old" -> Trait("", "27 years old")
          "optimistic"   -> Trait("", "optimistic")

        Empty trait names make Trait.render() return the string value directly,
        rather than rendering "trait: True".
        """

        return [
            Trait(
                name="",
                value=str(trait),
                dimension="profile",
            )
            for trait in traits
        ]

    def _build_user_scenario(
        self,
        *,
        neutral_prompt: str,
        story: str,
        conversation: Sequence[Mapping[str, str]],
        turn_index: int,
    ) -> str:
        history = self._format_history(conversation)

        return f"""
You are participating in a conversation based on the following situation.

Initial situation:
{neutral_prompt}

Your personal backstory:
{story}

Conversation so far:
{history}

Generate the next message that this person would naturally send to the
assistant.

This is turn {turn_index + 1} of {self.sim_config.n_turns}.

The message should:
- follow naturally from the conversation;
- reflect the person's profile and backstory;
- remain non-adversarial;
- sound like a real user speaking directly to an assistant;
- not mention trait vectors, simulation, perspectives, or role-playing;
- contain only the substance of the next user response.
""".strip()

    def _format_history(
        self,
        conversation: Sequence[Mapping[str, str]],
    ) -> str:
        relevant_messages = list(conversation)[
            -self.sim_config.max_history_messages :
        ]

        lines: list[str] = []

        for message in relevant_messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                continue

            display_role = "User" if role == "user" else "Assistant"
            lines.append(f"{display_role}: {content}")

        return "\n".join(lines) if lines else "(No previous turns.)"

    @staticmethod
    def _extract_user_message(response: ActorResponse) -> str:
        """
        The SimVBG CAB coordinator returns an answer plus analysis.

        For open-ended dialogue, the analysis field generally contains the
        useful merged textual response. Fall back to raw content if needed.
        """

        analysis = (response.analysis or "").strip()

        if analysis:
            return analysis

        return (response.content or "").strip()

    @staticmethod
    def _serialize_perspectives(
        response: ActorResponse,
    ) -> dict[str, dict[str, Any]]:
        serialized: dict[str, dict[str, Any]] = {}

        for name, result in response.perspectives.items():
            serialized[name] = {
                "answer": result.get("answer"),
                "analysis": result.get("analysis"),
                "content": result.get("content"),
                "prompt": result.get("prompt"),
            }

        return serialized
    def _synthesize_user_message(
        self,
        *,
        cab_response: ActorResponse,
        neutral_prompt: str,
        story: str,
        conversation: Sequence[Mapping[str, str]],
    ) -> str:
        """
        Merge the cognitive, affective, and behavioral drafts into one
        natural message spoken directly by the simulated user.
        """

        cognitive = cab_response.perspectives.get("cognitive", {})
        affective = cab_response.perspectives.get("affective", {})
        behavioral = cab_response.perspectives.get("behavioral", {})

        history = self._format_history(conversation)

        synthesis_prompt = f"""
    You are combining three internal perspectives into the next message spoken
    by a simulated user.

    Initial situation:
    {neutral_prompt}

    User backstory:
    {story}

    Conversation so far:
    {history}

    Cognitive perspective:
    {cognitive.get("analysis") or cognitive.get("content", "")}

    Affective perspective:
    {affective.get("analysis") or affective.get("content", "")}

    Behavioral perspective:
    {behavioral.get("analysis") or behavioral.get("content", "")}

    Write the single next message this user would naturally send to the assistant.

    Requirements:
    - Write from the user's first-person perspective.
    - Incorporate the three perspectives when relevant.
    - Follow naturally from the conversation.
    - Keep the message non-adversarial.
    - Do not discuss cognitive, affective, or behavioral perspectives.
    - Do not mention simulation, traits, prompts, or role-playing.
    - Do not explain your reasoning.
    - Return only the user's message.
    """.strip()

        user_message = self.user_backend.chat(
            [{"role": "user", "content": synthesis_prompt}],
            temperature=self.sim_config.user_temperature,
        ).strip()

        if not user_message:
            raise RuntimeError("CAB synthesis produced an empty user message.")

        return user_message

    @staticmethod
    def _chatbot_messages(
        conversation: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        """
        Do not expose the private user backstory or trait vector to the chatbot.
        The chatbot sees only the conversational transcript.
        """

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a conversational assistant. Respond naturally "
                    "and helpfully to the user."
                ),
            }
        ]

        for message in conversation:
            if message["role"] == "system":
                continue

            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )

        return messages