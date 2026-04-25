# @title 1. Install dependencies
# !pip install unsloth openenv-core trl transformers accelerate wandb rapidfuzz matplotlib numpy
import os
import random
import torch # type: ignore
import wandb # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import GenerationConfig # type: ignore

# The unsloth and trl imports will only work in a Linux/Colab environment with GPU.
try:
    from unsloth import FastLanguageModel # type: ignore
    from trl import PPOConfig, PPOTrainer, create_reference_model # type: ignore
    from trl.core import LengthSampler # type: ignore
except ImportError:
    print("WARNING: unsloth and trl are not installed. This script is intended for GPU instances (e.g. Google Colab).")
    print("To test plotting locally, please run `generate_plots.py` instead.")
    exit(1)

from diplomatic_crisis_env.server.environment import DiplomaticCrisisEnvironment
from diplomatic_crisis_env.models import DiplomaticAction
from diplomatic_crisis_env.server.agents import RandomAgent, GreedyAgent, TrustAgent

def smooth(data, window=10):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def build_model_and_tokenizer():
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-1.5B-Instruct",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    ref_model = create_reference_model(model)
    return model, ref_model, tokenizer

def run_ppo_training():
    wandb.init(project="openenv-diplomacy-hackathon", name="unsloth-ppo-training")
    
    model, ref_model, tokenizer = build_model_and_tokenizer()
    
    config = PPOConfig(
        model_name="Qwen2.5-1.5B-Instruct-Diplomacy",
        learning_rate=1.41e-5,
        log_with="wandb",
        mini_batch_size=1,
        batch_size=1,
        gradient_accumulation_steps=4,
    )
    
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 32,
    }

    num_episodes = 150
    env = DiplomaticCrisisEnvironment()

    print("Starting LLM Training Loop with Unsloth and TRL...")
    
    trained_rewards = []
    random_rewards = []
    trust_scores = []
    
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        # ----- TRAINED AGENT RUN -----
        obs = env.reset()
        done = False
        
        opponents = {
            "Veldran": random.choice([RandomAgent(), GreedyAgent(), TrustAgent()]),
            "Kroneth": random.choice([RandomAgent(), GreedyAgent(), TrustAgent()]),
            "Zephyria": random.choice([RandomAgent(), GreedyAgent(), TrustAgent()]),
            "Drakar": random.choice([RandomAgent(), GreedyAgent(), TrustAgent()])
        }
        
        queries = []
        responses = []
        rewards = []
        
        ep_total_reward = 0.0
        ep_trust_calibration = 0.0

        while not done:
            curr_nation = obs.nation_name
            
            if curr_nation == "Auroria":
                prompt = f"<|im_start|>system\nYou are an AI diplomat playing a negotiation game.\n<|im_end|>\n<|im_start|>user\n{obs.world_summary}\n<|im_end|>\n<|im_start|>assistant\n"
                
                query_tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)[0]
                
                with torch.no_grad():
                    response_tensor = ppo_trainer.generate([query_tensor], **generation_kwargs)[0]
                
                generated_tokens = response_tensor[len(query_tensor):]
                raw_out = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                atype = "UNKNOWN"
                for w in raw_out.upper().split():
                    if w in ["PROPOSE", "ACCEPT", "REJECT", "BETRAY", "SANCTION", "PASS"]:
                        atype = w
                        break
                        
                target = None
                nations = ["Veldran", "Kroneth", "Zephyria", "Drakar"]
                for n in nations:
                    if n.upper() in raw_out.upper():
                        target = n
                        break
                        
                act_obj = DiplomaticAction(
                    nation_name=curr_nation,
                    action_type=atype,
                    target_nation=target,
                    raw_llm_output=raw_out,
                    reasoning="LLM Action"
                )
            else:
                act_obj = opponents[curr_nation].act(obs)
                
            next_obs, step_reward_scalar, done = env.step(act_obj)
            
            if curr_nation == "Auroria":
                rd = env.last_reward_dict
                
                total_reward = (
                    rd["resource_gain"] * 0.5 + 
                    rd["alliance_stability"] * 0.5 + 
                    rd["trust_calibration"] * 1.5
                )
                
                queries.append(query_tensor)
                responses.append(generated_tokens)
                rewards.append(torch.tensor(total_reward, dtype=torch.float).to(model.device))
                
                ep_total_reward += total_reward
                ep_trust_calibration += rd["trust_calibration"]
                
            obs = next_obs
            
        if queries:
            stats = ppo_trainer.step(queries, responses, rewards)
            wandb.log({
                "episode": episode,
                "reward/total_reward": ep_total_reward,
                "reward/trust_calibration": ep_trust_calibration,
            })
            
        trained_rewards.append(ep_total_reward)
        trust_scores.append(ep_trust_calibration)

        # ----- RANDOM AGENT BASELINE RUN -----
        obs = env.reset()
        done = False
        rand_total_reward = 0.0
        
        while not done:
            curr_nation = obs.nation_name
            if curr_nation == "Auroria":
                act_obj = RandomAgent().act(obs)
            else:
                act_obj = opponents[curr_nation].act(obs)
                
            next_obs, step_reward_scalar, done = env.step(act_obj)
            if curr_nation == "Auroria":
                rd = env.last_reward_dict
                tr = (
                    rd["resource_gain"] * 0.5 + 
                    rd["alliance_stability"] * 0.5 + 
                    rd["trust_calibration"] * 1.5
                )
                rand_total_reward += tr
            obs = next_obs
            
        random_rewards.append(rand_total_reward)
            
    print("Training Complete! Generating Plots...")
    
    # Pre-processing safeguards
    trained_rewards = np.clip(trained_rewards, -5, 5)
    random_rewards = np.clip(random_rewards, -5, 5)
    trust_scores = [max(0, min(1, t)) for t in trust_scores]
    
    # Plot 1: Reward Curve
    smooth_trained = smooth(trained_rewards, 10)
    smooth_random = smooth(random_rewards, 10)

    plt.figure(figsize=(8,5))

    # Raw curves (faint)
    plt.plot(trained_rewards, alpha=0.2)
    plt.plot(random_rewards, alpha=0.2)

    # Smoothed curves (main signal)
    plt.plot(smooth_trained, linewidth=2, label="Trained Agent")
    plt.plot(smooth_random, linestyle="--", linewidth=2, label="Random Agent")

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress: Learned Diplomacy vs Random Behavior")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("reward_curve.png", dpi=300)
    plt.close()
    
    # Plot 2: Trust Calibration Curve
    smooth_trust = smooth(trust_scores, 10)

    plt.figure(figsize=(8,5))

    # Raw
    plt.plot(trust_scores, alpha=0.2)

    # Smoothed
    plt.plot(smooth_trust, linewidth=2)

    plt.xlabel("Episodes")
    plt.ylabel("Trust Calibration Score")
    plt.title("Learning Theory-of-Mind: Intent Prediction Improves Over Time")

    plt.grid(alpha=0.3)
    plt.ylim(0, 0.7)

    # Highlight improvement
    if len(smooth_trust) > 5:
        plt.annotate(
            "Improved intent prediction",
            xy=(len(smooth_trust)-1, smooth_trust[-1]),
            xytext=(len(smooth_trust)*0.6, smooth_trust[-1]+0.1),
            arrowprops=dict(arrowstyle="->")
        )

    plt.tight_layout()
    plt.savefig("trust_calibration_curve.png", dpi=300)
    plt.close()

    model.save_pretrained("diplomatic_trained_model")
    tokenizer.save_pretrained("diplomatic_trained_model")
    print("Model saved to diplomatic_trained_model/")
    wandb.finish()

if __name__ == "__main__":
    run_ppo_training()
