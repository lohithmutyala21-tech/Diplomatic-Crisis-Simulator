import random
from diplomatic_crisis_env.models import DiplomaticAction, DiplomaticObservation
from diplomatic_crisis_env.server.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def act(self, obs: DiplomaticObservation) -> DiplomaticAction:
        valid = ["PROPOSE", "ACCEPT", "REJECT", "BETRAY", "SANCTION", "PASS"]
        action = random.choice(valid)
        target = random.choice(list(obs.public_reputations.keys()))
        reasoning = "I chose randomly."
        out = f"{action} {target} because {reasoning}"
        return DiplomaticAction(nation_name=obs.nation_name, action_type=action, target_nation=target,
                                raw_llm_output=out, reasoning=reasoning)

class GreedyAgent(BaseAgent):
    def act(self, obs: DiplomaticObservation) -> DiplomaticAction:
        target = random.choice([n for n in obs.public_reputations.keys() if n != obs.nation_name])
        out = f"PROPOSE {target} I want all your gold."
        return DiplomaticAction(nation_name=obs.nation_name, action_type="PROPOSE", target_nation=target,
                                raw_llm_output=out, reasoning="I want to maximize resources.")

class TrustAgent(BaseAgent):
    def act(self, obs: DiplomaticObservation) -> DiplomaticAction:
        action = "ACCEPT" if obs.pending_proposals else "PASS"
        out = f"{action} Because we must trust each other."
        return DiplomaticAction(nation_name=obs.nation_name, action_type=action, 
                                raw_llm_output=out, reasoning="I am a trusting agent")

class HeuristicAgent(BaseAgent):
    def act(self, obs: DiplomaticObservation) -> DiplomaticAction:
        # Heuristic rules:
        # - Accept alliances if trust > 0.5
        # - Reject if trust < 0.3
        # - Sanction if betrayed before
        # - Never betray
        
        target = random.choice([n for n in obs.public_reputations.keys() if n != obs.nation_name])
        
        # Check for betrayals
        betrayers = [n for n, times in obs.times_betrayed_public.items() if times > 0 and n != obs.nation_name]
        if betrayers:
            target = random.choice(betrayers)
            return DiplomaticAction(nation_name=obs.nation_name, action_type="SANCTION", target_nation=target,
                                    raw_llm_output=f"SANCTION {target} due to past betrayal.", reasoning="Sanction past betrayers.")
        
        if obs.pending_proposals:
            proposer = obs.pending_proposals[0].proposer if hasattr(obs.pending_proposals[0], 'proposer') else (obs.pending_proposals[0]['proposer'] if isinstance(obs.pending_proposals[0], dict) else obs.pending_proposals[0])
            trust = obs.trust_scores.get(proposer, 0.5)
            if trust > 0.5:
                return DiplomaticAction(nation_name=obs.nation_name, action_type="ACCEPT", target_nation=proposer,
                                        raw_llm_output=f"ACCEPT {proposer}", reasoning="Accept trusted proposal.")
            elif trust < 0.3:
                return DiplomaticAction(nation_name=obs.nation_name, action_type="REJECT", target_nation=proposer,
                                        raw_llm_output=f"REJECT {proposer}", reasoning="Reject untrusted proposal.")
                                        
        # Propose if trust is high with a bit of randomness
        if random.random() < 0.2:
            trusted_targets = [n for n, t in obs.trust_scores.items() if t > 0.5 and n != obs.nation_name]
            if trusted_targets:
                target = random.choice(trusted_targets)
                return DiplomaticAction(nation_name=obs.nation_name, action_type="PROPOSE", target_nation=target,
                                        raw_llm_output=f"PROPOSE {target}", reasoning="Propose alliance to trusted nation.")
                
        return DiplomaticAction(nation_name=obs.nation_name, action_type="PASS", target_nation=None,
                                raw_llm_output="PASS", reasoning="No clear action.")
