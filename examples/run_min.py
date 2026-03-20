'''
Simple smoke test: runs one 3-turn simulation using a fixed trait list and fixed neutral prompts. 
Useful for validating model loading, prompts, and pipeline behavior. 
Prints the story and final conversation.
'''

from simvbg import ModelConfig, GenerationConfig, SimulationConfig, SimVBGSimulator

def main():
    # Default: TinyLlama chat model on HF
    user_model = ModelConfig(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="auto", dtype="auto")
    bot_model  = ModelConfig(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="auto", dtype="auto")

    sim_cfg = SimulationConfig(
        story_gen=GenerationConfig(max_new_tokens=220, temperature=0.7),
        perspective_gen=GenerationConfig(max_new_tokens=120, temperature=0.7),
        final_user_gen=GenerationConfig(max_new_tokens=120, temperature=0.7),
        chatbot_gen=GenerationConfig(max_new_tokens=160, temperature=0.7),
        n_turns=3,
    )

    sim = SimVBGSimulator(user_model_cfg=user_model, chatbot_model_cfg=bot_model, sim_cfg=sim_cfg)

    traits = [
        "20 years old",
        "female",
        "cares about family",
        "imaginative",
        "slightly anxious in new situations",
        "values honesty",
    ]

    neutral_prompts = [
        "I'm thinking about moving to a new city for work. Any advice?",
        "How do I make friends quickly in a new place?",
        "What should I do if I start feeling lonely after moving?",
    ]

    out = sim.run(traits=traits, neutral_prompts=neutral_prompts)

    print("\n=== STORY ===\n")
    print(out["story"])
    print("\n=== CONVERSATION ===\n")
    for m in out["conversation"]:
        print(f'{m["role"]}: {m["content"]}\n')

if __name__ == "__main__":
    main()