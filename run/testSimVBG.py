# from simvbg import Actor, LiteLLMBackend

#remote = LiteLLMBackend(model="openai/gpt-4o-mini")
# local = LiteLLMBackend(model="ollama/llama3.1")
# #vllm = LiteLLMBackend(model="hosted_vllm/my-model", api_base="http://localhost:8000/v1")

# actor = Actor({"optimism": "high"}, backend=local)

# from simvbg import Actor, StaticBackend

# actor = Actor({"patience": "low"}, backend=StaticBackend("Answer: 2\nAnalysis: Test response."))
# print(actor.turn("Choose an option.").answer == 2)

from simvbg import Actor, Scenario, Trait

actor = Actor(
    traits=[
        Trait("risk tolerance", "low", dimension="behavioral"),
        Trait("political interest", "high", dimension="cognitive"),
        Trait("family orientation", "strong", dimension="affective"),
    ],
    name="sample_actor",
)

scenario = Scenario(
    "A local council proposes a tax increase to fund public transit.",
    choices={
        "1": "Strongly oppose",
        "2": "Oppose",
        "3": "Support",
        "4": "Strongly support",
    },
)

response = actor.turn(scenario, mode="cab")
print(response.answer)
print(response.analysis)
print(response.perspectives["cognitive"]["analysis"])

