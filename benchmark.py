import argparse
import random
import numpy as np
import sys
from tqdm import tqdm

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from diplomatic_crisis_env.server.agents import RandomAgent, HeuristicAgent
from generate_plots import run_episode, LightweightQAgent

class Colors:
    HEADER = '\033[95m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def main():
    parser = argparse.ArgumentParser(description="Diplomatic Crisis Simulator Benchmark")
    parser.add_argument("--agent", type=str, choices=["random", "heuristic", "trained"], required=True, help="Agent to benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (5 episodes)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    num_episodes = 5 if args.quick else 150
    print(f"{Colors.OKCYAN}Running Benchmark for {args.agent.capitalize()} Agent (Episodes: {num_episodes})...{Colors.ENDC}")

    if args.agent == "random":
        agent = None # run_episode handles random internally if agent is None
    elif args.agent == "heuristic":
        agent = HeuristicAgent()
    else:
        actions = ["PROPOSE", "ACCEPT", "REJECT", "BETRAY", "SANCTION", "PASS"]
        agent = LightweightQAgent(actions)

    rewards = []
    trust_scores = []
    alliances = 0

    for _ in tqdm(range(num_episodes)):
        metrics = run_episode(agent, agent_type=args.agent)
        rewards.append(metrics["total_reward"])
        trust_scores.append(metrics["trust_calibration"])
        alliances += metrics["alliances_formed"]

    avg_reward = np.mean(rewards)
    avg_trust = np.mean(trust_scores)
    avg_alliances = alliances / num_episodes

    print(f"\n{Colors.HEADER}{Colors.BOLD}=============================={Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}📊 BENCHMARK RESULTS{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}=============================={Colors.ENDC}")
    print(f"Agent:             {Colors.BOLD}{args.agent.capitalize()}{Colors.ENDC}")
    print(f"Avg Reward:        {Colors.OKGREEN}{avg_reward:.2f}{Colors.ENDC}")
    print(f"Trust Calibration: {Colors.WARNING}{avg_trust:.2f}{Colors.ENDC}")
    print(f"Alliances:         {Colors.OKGREEN}{avg_alliances:.1f}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}=============================={Colors.ENDC}\n")

if __name__ == "__main__":
    main()
