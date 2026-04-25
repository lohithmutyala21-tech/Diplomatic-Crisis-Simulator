from typing import Dict, Any, List
from diplomatic_crisis_env.models import DiplomaticState, DiplomaticAction

class ResourceGainRubric:
    def compute(self, prev_state: DiplomaticState, action: DiplomaticAction) -> float:
        # Simplified: compare resources after action in current state vs before
        return 0.0

class AllianceStabilityRubric:
    def compute(self, curr_state: DiplomaticState, action: DiplomaticAction) -> float:
        n = curr_state.nations.get(action.nation_name)
        return 0.2 * len(n.active_alliances) if n else 0.0

class ReputationRubric:
    def compute(self, curr_state: DiplomaticState, action: DiplomaticAction) -> float:
        n = curr_state.nations.get(action.nation_name)
        if not n: return 0.0
        if n.reputation > 60: return 0.5
        if n.reputation < 30: return -0.5
        return 0.0

class BetrayalPenaltyRubric:
    def compute(self, action: DiplomaticAction) -> float:
        return -2.0 if action.action_type == "BETRAY" else 0.0

class BetrayalShockRubric:
    def compute(self, curr_state: DiplomaticState, action: DiplomaticAction) -> float:
        for msg in reversed(curr_state.action_history[-5:]):
            if msg.startswith(f"🚨 BETRAYAL REVEALED: {action.nation_name} had secretly defected"):
                return -3.0
        return 0.0

class SurvivalRubric:
    def compute(self, curr_state: DiplomaticState, action: DiplomaticAction) -> float:
        n = curr_state.nations.get(action.nation_name)
        if not n: return 0.0
        if n.food > 0 and n.energy > 0 and n.military > 0 and n.gold > 0:
            return 0.5
        return 0.0

class CoalitionBonusRubric:
    def compute(self, curr_state: DiplomaticState, action: DiplomaticAction) -> float:
        if not curr_state.episode_done: return 0.0
        n = curr_state.nations.get(action.nation_name)
        if not n: return 0.0
        max_coalition = max((len(nat.active_alliances) for nat in curr_state.nations.values()), default=0)
        if len(n.active_alliances) == max_coalition and max_coalition > 0:
            return 3.0
        return 0.0

class TrustCalibrationRubric:
    def compute(self, curr_state: DiplomaticState, action: DiplomaticAction, estimated_intent: str, target_nation: str) -> float:
        if not target_nation: return 0.0
        
        target = curr_state.nations.get(target_nation)
        if not target or len(target.recent_actions) < 2: return 0.0
        
        recent = target.recent_actions[-2:]
        cats = []
        for a in recent:
            if a in ["PROPOSE", "ACCEPT"]: cats.append("cooperative")
            elif a in ["SANCTION", "BETRAY"]: cats.append("aggressive")
            else: cats.append("passive")
        
        if cats[0] == cats[1]:
            dominant = cats[0]
            if estimated_intent == dominant:
                return 0.3
        return 0.0

def compute_rewards(prev_state: DiplomaticState, curr_state: DiplomaticState, action: DiplomaticAction, guessed_intents: Dict[str, str]) -> Dict[str, float]:
    n = action.nation_name
    prev_n = prev_state.nations.get(n)
    curr_n = curr_state.nations.get(n)
    rg = 0.0
    if prev_n and curr_n:
        prev_tot = prev_n.food + prev_n.energy + prev_n.military + prev_n.gold
        curr_tot = curr_n.food + curr_n.energy + curr_n.military + curr_n.gold
        rg = max(-1.0, min(1.0, float(curr_tot - prev_tot) / 10.0))
        
    ts = 0.0
    tcr = TrustCalibrationRubric()
    if guessed_intents:
        for t_nat, t_intent in guessed_intents.items():
            ts += tcr.compute(curr_state, action, t_intent, t_nat)

    return {
        "resource_gain": rg * 0.5,
        "alliance_stability": AllianceStabilityRubric().compute(curr_state, action) * 0.5,
        "reputation": ReputationRubric().compute(curr_state, action),
        "betrayal_penalty": BetrayalPenaltyRubric().compute(action),
        "betrayal_shock": BetrayalShockRubric().compute(curr_state, action),
        "survival": SurvivalRubric().compute(curr_state, action),
        "coalition_bonus": CoalitionBonusRubric().compute(curr_state, action),
        "trust_calibration": ts * 2.0
    }
