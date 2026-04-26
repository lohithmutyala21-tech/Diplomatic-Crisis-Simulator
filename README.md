---
title: Diplomatic Crisis Simulator
emoji: 🌍
colorFrom: blue
colorTo: red
sdk: gradio
pinned: false
app_file: app.py
---

# 🧠 Intent & Trust Arena

**[👉 Play with the Live Environment on Hugging Face Spaces!](https://huggingface.co/spaces/lohith-2108/deplomatic-env)**

### Train agents to model trust and intent

## 🌍 Problem Motivation & Environment Explanation
Multi-agent systems often fail to model trust, intent, and strategic behavior in dynamic environments. 

**How the Environment Works:**
- **Geopolitical Simulation with Hidden Intent:** Agents simulate diplomatic decisions (forming alliances, proposing treaties). However, agents can publicly ally while secretly plotting betrayal.
- **Partial Observability:** Agents operate under uncertainty, never fully knowing if a partner is cooperating or defecting until the results are public.
- **OpenEnv Interaction:** OpenEnv strictly manages the interaction loop, routing hidden states and public actions between the environment and the multi-agent policy.
- **RL Training Optimization:** Using Unsloth and TRL (PPO/GRPO), the training script explicitly optimizes the tradeoff between standard reward (survival/resources) and our custom theory-of-mind metric (trust calibration accuracy).

🔥 Key Result:
- Reward improves (1.92 → 2.86)
- Trust calibration increases (~0.22 → ~0.51)

🚀 Run:
```bash
python demo.py
```

📖 Full explanation:
See [Blog.md](Blog.md)
