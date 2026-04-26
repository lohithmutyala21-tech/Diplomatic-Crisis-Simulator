---
title: Diplomatic Crisis Simulator
emoji: 🌍
colorFrom: blue
colorTo: red
sdk: gradio
pinned: false
app_file: app.py
---

# 🔥 Diplomatic Crisis Simulator: Learning Trust, Betrayal, and Strategic Reasoning in Multi-Agent AI Systems

**[👉 Open Hugging Face Space (Live Demo)](https://huggingface.co/spaces/lohith-2108/deplomatic-env)**
**[📒 View Colab Training Notebook](https://colab.research.google.com/drive/1zfvv1OhOC4pzTMPx3cPnfmdbDQPM1vng?usp=sharing)**
**[🎥 View Project Blog](Blog.md)**

---

## 🧩 Problem Motivation

In the pursuit of Artificial General Intelligence (AGI), multi-agent alignment remains one of the hardest unsolved problems. When AI systems interact in the real world, they must navigate environments where incentives are misaligned, communication is unreliable, and actors may act deceptively.

Current reinforcement learning environments heavily index on short-term resource optimization. They fail to model **trust, betrayal, and theory-of-mind**. We built the *Diplomatic Crisis Simulator* to bridge this gap. This project forces Large Language Models to engage in a geopolitical simulation where survival depends on accurately predicting the hidden intents of other agents—mimicking the core challenges of real-world AI alignment and strategic cooperation.

## 🌍 Environment Explanation

Built entirely on top of the **OpenEnv** framework, the environment models a geopolitical standoff between five nations. 

- **Geopolitical Simulation:** Agents must manage resources (Food, Energy, Military, Gold) while forming alliances, enacting treaties, or imposing sanctions.
- **Hidden Intent Modeling:** While actions (e.g., *Propose Alliance*) are public, the true *intent* (e.g., *Sincere Cooperation* vs. *Setup for Betrayal*) is hidden.
- **Partial Observability:** Agents operate under strict uncertainty. They must infer intent from past behavior and reputation.
- **The Diplomacy Loop:** OpenEnv manages a strict state machine, processing public actions and resolving hidden states. If an agent betrays an alliance, devastating penalties are applied delayed in time, forcing agents to model long-horizon consequences.

## 🧠 Training Method (Unsloth + TRL)

Training a model to "understand trust" requires sophisticated reward shaping. We utilized **Unsloth** for rapid 4-bit LoRA training combined with Hugging Face's **TRL (PPO/GRPO)**.

Our core innovation is the **Theory-of-Mind Reward Rubric**:
1. **Resource Gain:** Standard survival optimization.
2. **Alliance Stability:** Rewards for maintaining unbroken treaties.
3. **Trust Calibration (The Core Metric):** The agent is asked to *predict* the hidden intent of its counterpart. If it accurately predicts a betrayal before it happens, it receives a massive reward multiplier. If it blindly trusts a defector, it is severely penalized.

## 📊 Results

The implementation of the trust calibration reward signal drastically improved the agent's ability to survive in adversarial conditions.

- **Reward Improvement:** Average total reward increased from **1.92 (Baseline Heuristic)** to **2.86 (Trained Policy)**.
- **Trust Calibration:** The agent's ability to accurately predict hidden intent improved from **0.22** to **0.51**, proving that theory-of-mind can be successfully optimized.

*(See the generated `reward_curve.png` and `trust_calibration_curve.png` in this repository for visual proof of convergence).*

## 🧠 Final Insight: What the Agents Learned

Through rigorous PPO training, the agents demonstrated emergent strategic behaviors rarely seen in standard benchmarks:
- **Trust Evolution:** Early in training, agents acted randomly or overly aggressively. By episode 100, they learned to build "coalition blocks" with highly reputable nations.
- **Betrayal Detection:** The agents developed a mathematical intuition for deception. They learned to sanction nations that rapidly amassed military power *before* the betrayal occurred, proving that the trust calibration metric successfully imparted a rudimentary theory-of-mind.

---
*Built for the OpenEnv India Hackathon 2026. This project proves that intent modeling is both measurable and trainable.*
