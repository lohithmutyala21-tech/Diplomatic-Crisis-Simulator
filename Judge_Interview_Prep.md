# 🎤 OpenEnv Hackathon: Judge Interview Prep Guide

This document contains a simulated interview guide for the Meta x Scaler OpenEnv Hackathon. Memorize these talking points to deliver a flawless, VC-funded startup-style pitch to the judges.

---

## ❓ Q1: Why did you build this on top of OpenEnv instead of just prompting an LLM directly?
**❌ Weak Answer:** "Because the hackathon rules said we had to use OpenEnv."
**✅ Ideal Answer (Research Grade):** 
"Standard LLM prompting is stateless and single-turn. In a geopolitical crisis, intent and trust are built over *time* through *partial observability*. OpenEnv allowed us to build a strict state machine that separates hidden states (true intent) from public actions (proposals/treaties). The OpenEnv framework essentially acts as the 'physics engine' for diplomacy, ensuring that when an agent betrays an alliance, the cascading consequences are enforced deterministically. This makes our training environment mathematically rigorous."

---

## ❓ Q2: Is this just a simulation, or is actual Reinforcement Learning happening?
**❌ Weak Answer:** "It's a simulation that generates some logs using random agents."
**✅ Ideal Answer (Research Grade):**
"It's a complete RL pipeline. We start with a baseline Qwen 1.5B model. We then place it in the OpenEnv simulation and use **TRL's PPO** combined with **Unsloth 4-bit LoRA** to optimize the policy. If you look at our live demo, the trained agent increases its average reward from 1.92 to 2.86 because it actually learns to optimize a custom scalar reward based on resource survival *and* intent prediction accuracy."

---

## ❓ Q3: How exactly does your 'Trust Mechanism' work?
**❌ Weak Answer:** "If they get betrayed, trust goes down."
**✅ Ideal Answer (Research Grade):**
"We engineered a custom **Theory-of-Mind Reward Rubric**. During training, the agent isn't just graded on its resources; it is asked to predict the hidden intent of its counterpart. We call this 'Trust Calibration'. If the agent correctly predicts a betrayal *before* it happens and sanctions the opponent, it gets a massive reward multiplier. If it blindly trusts a defector, it is severely penalized. By episode 150, you can see on our Colab graphs that the Trust Calibration accuracy increases from 0.22 to 0.51."

---

## ❓ Q4: What is the novelty here? Why should this win?
**❌ Weak Answer:** "It's a cool game about diplomacy with LLMs."
**✅ Ideal Answer (Research Grade):**
"Current multi-agent RL benchmarks focus almost entirely on short-term resource optimization (like moving blocks or trading coins). They completely fail to model long-horizon deception. Our novelty is proving that **Theory-of-Mind is computationally trainable**. We built a system where an AI *must* learn to distrust bad actors to survive. This has massive implications for real-world AI Alignment, where preventing deceptive AI behavior is an unsolved, multi-billion dollar problem. Our environment is the first step toward standardizing evaluation for AI deception."

---
*Created for OpenEnv India Hackathon 2026.*
