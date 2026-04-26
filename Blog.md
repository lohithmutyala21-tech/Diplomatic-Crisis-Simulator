# 🧠 Intent & Trust Arena — Hackathon Submission

## 🚀 Overview
We built a multi-agent diplomatic simulation environment where agents interact through negotiation, cooperation, and competition under partial observability.

## 🎯 Problem Statement
Multi-agent systems often fail to model trust, intent, and strategic behavior in dynamic environments. LLMs are notoriously poor at maintaining long-term reasoning when faced with deception, shifting alliances, and hidden variables.

## 🧠 Solution
We designed an OpenEnv-based environment where:
- Agents simulate diplomatic decisions and alliances.
- Trust is dynamically computed based on intent prediction.
- Rewards strictly depend on cooperation and advanced theory-of-mind strategy.

## ⚙️ Environment Design
- **OpenEnv Framework**: Leveraging strict state transitions and action spaces.
- **Betrayal Mechanics**: Agents can publicly ally but secretly sanction or defect.
- **Trust Dynamics**: Theory-of-mind is measured by how well an agent predicts the intent of others.

## 📊 Results
Trained agents consistently outperform baselines, proving that they are internalizing the mechanics rather than acting randomly.
Trust calibration improves significantly over time as the agent learns to predict hidden intent.

![Reward Curve](reward_curve.png)
*Trained agent outperforms baseline.*

![Trust Calibration Curve](trust_calibration_curve.png)
*Trust calibration improves over time.*

## 🔥 Key Innovation
- **Intent-aware rewards**: Directly optimizing theory-of-mind.
- **Delayed betrayal modeling**: Actions have long-term consequences.
- **Measurable theory-of-mind**: Trust calibration gives a hard mathematical score to intuition.

## 🧪 Scientific Validation
- **Ablation Study**: Confirmed that removing the trust signal destroys the agent's ability to maintain alliances.
- **Multi-Seed Evaluation**: Demonstrated low variance and robust convergence across multiple runs.
- **Generalization Test**: Agent scales gracefully when the environment expands with more nations.

## 🎥 Demo
[Add your video link here]

## 🏁 Conclusion
This proves theory-of-mind can be trained and measured. This project demonstrates how intent and trust can be modeled in multi-agent reinforcement learning environments, directly satisfying Hackathon Theme #1.
