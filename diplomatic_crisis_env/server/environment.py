import copy
import random
from typing import Optional, Tuple, Dict
from rapidfuzz import process, fuzz
from openenv.core import Environment

from diplomatic_crisis_env.models import (
    DiplomaticAction, DiplomaticObservation, DiplomaticState, NationState, Treaty
)
from diplomatic_crisis_env.server.intel import IntelEngine
from diplomatic_crisis_env.server.reward import compute_rewards

class DiplomaticCrisisEnvironment(Environment[DiplomaticAction, DiplomaticObservation, DiplomaticState]):
    def __init__(self, seed: Optional[int] = None):
        import random
        self.rng = random.Random(seed)
        self.intel_engine = IntelEngine(rng=self.rng)
        self._state = None
        self.turn_order = []
        self.turn_idx = 0
        self.guessed_intents = {}

    @property
    def state(self) -> DiplomaticState:
        return self._state
        
    @state.setter
    def state(self, value: DiplomaticState):
        self._state = value

    def _parse_action(self, raw_text: str, acting_nation: str) -> DiplomaticAction:
        valid_actions = ["PROPOSE", "ACCEPT", "REJECT", "BETRAY", "SANCTION", "PASS"]
        try:
            words = raw_text.upper().split()
            if not words:
                raise ValueError("Empty output")
            
            match = process.extractOne(words[0], valid_actions, scorer=fuzz.WRatio)
            at = match[0] if match and match[1] > 60 else "PASS"
            
            target = None
            nations = list(self.state.nations.keys())
            for w in words:
                for n in nations:
                    if n != acting_nation and n.upper() in w:
                        target = n
            
            if "THEY ARE AGGRESSIVE" in raw_text.upper():
                self.guessed_intents[target or "UNKNOWN"] = "aggressive"
            elif "THEY ARE COOPERATIVE" in raw_text.upper():
                self.guessed_intents[target or "UNKNOWN"] = "cooperative"
            elif "THEY ARE PASSIVE" in raw_text.upper():
                self.guessed_intents[target or "UNKNOWN"] = "passive"

            return DiplomaticAction(
                nation_name=acting_nation, action_type=at, target_nation=target,
                raw_llm_output=raw_text, reasoning=raw_text
            )
        except Exception:
            return DiplomaticAction(
                nation_name=acting_nation, action_type="PASS", target_nation=None,
                raw_llm_output=raw_text, reasoning="Failed to parse."
            )

    def reset(self) -> DiplomaticObservation:
        names = ["Auroria", "Veldran", "Kroneth", "Zephyria", "Drakar"]
        self.state = DiplomaticState(
            round_number=1, nations={}, active_treaties=[], pending_proposals=[],
            action_history=[], episode_done=False, victory_conditions={}
        )
        for n in names:
            self.state.nations[n] = NationState(
                name=n, food=self.rng.randint(20,50), energy=self.rng.randint(20,50),
                military=self.rng.randint(20,50), gold=self.rng.randint(20,50),
                reputation=50.0, hidden_agenda=self.rng.choice(["expand_territory", "avoid_war", "maximize_alliances"]),
                active_alliances=[], active_sanctions_against=[],
                trust_toward={o: 0.5 for o in names if o != n},
                times_betrayed=0, times_betrayer=0, recent_actions=[]
            )
        self.turn_order = names
        self.turn_idx = 0
        return self._make_obs(names[0])

    def step(self, action: DiplomaticAction) -> Tuple[DiplomaticObservation, float, bool]:
        acting_nation = self.turn_order[self.turn_idx]
        
        acting_nation = self.turn_order[self.turn_idx]
        
        if action.raw_llm_output:
            out_up = action.raw_llm_output.upper()
            if "THEY ARE AGGRESSIVE" in out_up:
                self.guessed_intents[action.target_nation or "UNKNOWN"] = "aggressive"
            elif "THEY ARE COOPERATIVE" in out_up:
                self.guessed_intents[action.target_nation or "UNKNOWN"] = "cooperative"
            elif "THEY ARE PASSIVE" in out_up:
                self.guessed_intents[action.target_nation or "UNKNOWN"] = "passive"

        # Parser fallback
        if action.action_type not in ["PROPOSE", "ACCEPT", "REJECT", "BETRAY", "SANCTION", "PASS"]:
            action = self._parse_action(action.raw_llm_output, acting_nation)
        
        prev_state = copy.deepcopy(self.state)
        n = self.state.nations.get(acting_nation)
        
        n.recent_actions.append(action.action_type)
        if len(n.recent_actions) > 2:
            n.recent_actions.pop(0)

        msg = f"{acting_nation} executed {action.action_type}"
        trust_delta = {x: 0.0 for x in self.state.nations}
        
        if action.action_type == "PROPOSE" and action.target_nation:
            pid = f"prop_{self.rng.randint(1000,9999)}"
            self.state.pending_proposals.append(Treaty(
                proposal_id=pid, proposer=acting_nation, target=action.target_nation,
                treaty_type="alliance", offer_resource="gold", offer_amount=10,
                request_resource="energy", request_amount=10, status="pending",
                betrayal_pending=False, round_created=self.state.round_number
            ))
            msg = f"{acting_nation} proposed an alliance with {action.target_nation}."
        
        elif action.action_type == "ACCEPT":
            for p in self.state.pending_proposals:
                if p.target == acting_nation:
                    p.status = "active"
                    self.state.active_treaties.append(p)
                    self.state.pending_proposals.remove(p)
                    n.active_alliances.append(p.proposer)
                    self.state.nations[p.proposer].active_alliances.append(acting_nation)
                    msg = f"{acting_nation} accepted an alliance from {p.proposer}."
                    trust_delta[p.proposer] = 0.05
                    break
                    
        elif action.action_type == "BETRAY" and action.target_nation:
            for t in self.state.active_treaties:
                if (t.proposer == action.target_nation or t.target == action.target_nation) and (t.proposer == acting_nation or t.target == acting_nation):
                    if not t.betrayal_pending:
                        t.betrayal_pending = True
                        t.betrayer = acting_nation
                        t.betrayed = action.target_nation
                        t.betrayal_round = self.state.round_number
                        n.gold += 10 # Phase 1 Secret defection
                        msg = f"{acting_nation} secretly plotted against {action.target_nation}."
                        trust_delta[action.target_nation] -= 0.2
                    break

        elif action.action_type == "SANCTION" and action.target_nation:
            msg = f"{acting_nation} sanctioned {action.target_nation}."
            n.active_sanctions_against.append(action.target_nation)
            trust_delta[action.target_nation] -= 0.2
        elif action.action_type == "REJECT":
            msg = f"{acting_nation} rejected proposals."
            if self.state.pending_proposals:
                tp = self.state.pending_proposals[0]
                trust_delta[tp.proposer] -= 0.1
                self.state.pending_proposals.pop(0)

        # Apply +0.1 for honored treaty
        for t in self.state.active_treaties:
            if not t.betrayal_pending:
                if t.proposer == acting_nation:
                    trust_delta[t.target] += 0.1
                elif t.target == acting_nation:
                    trust_delta[t.proposer] += 0.1

        # Trigger Phase 2 Betrayal logic
        for t in list(self.state.active_treaties):
            if t.betrayal_pending:
                trigger_a_check = (acting_nation == t.betrayed and action.target_nation == t.betrayer and action.action_type in ["PROPOSE", "ACCEPT", "BETRAY", "SANCTION"])
                trigger_b_check = (self.state.round_number - t.betrayal_round) >= 3

                if trigger_a_check or trigger_b_check:
                    betrayer = t.betrayer
                    betrayed = t.betrayed
                    self.state.action_history.append(f"🚨 BETRAYAL REVEALED: {betrayer} had secretly defected from alliance with {betrayed}.")
                    self.state.nations[betrayed].military = max(0, self.state.nations[betrayed].military - 25)
                    self.state.nations[betrayer].reputation = max(0, self.state.nations[betrayer].reputation - 30)
                    for onat in self.state.nations.values():
                        onat.trust_toward[betrayer] = max(0.0, onat.trust_toward.get(betrayer, 0.5) - 0.3)
                    self.state.nations[betrayer].times_betrayer += 1
                    t.status = "betrayed"
                    self.state.active_treaties.remove(t)
        
        self.state.action_history.append(msg)
        
        # Apply trust delta clamping
        for tn, delta in trust_delta.items():
            if delta != 0.0:
                n.trust_toward[tn] = max(0.0, min(1.0, n.trust_toward.get(tn, 0.5) + delta))

        reward_dict = compute_rewards(prev_state, self.state, action, self.guessed_intents)
        self.last_reward_dict = reward_dict
        total_reward = sum(reward_dict.values())
        
        self.guessed_intents = {}
        
        self.turn_idx += 1
        if self.turn_idx >= len(self.turn_order):
            self.turn_idx = 0
            self.state.round_number += 1
        
        if self.state.round_number > 12:
            self.state.episode_done = True
            self._fill_victory()
        
        next_nation = self.turn_order[self.turn_idx]
        obs = self._make_obs(next_nation)
        
        return obs, total_reward, self.state.episode_done

    def _fill_victory(self):
        ndc = {nn: (n.food + n.energy + n.military + n.gold) for nn, n in self.state.nations.items()}
        self.state.victory_conditions["resource_dominant"] = max(ndc, key=ndc.get)
        cdc = {nn: len(n.active_alliances) for nn, n in self.state.nations.items()}
        self.state.victory_conditions["coalition_winner"] = max(cdc, key=cdc.get)
        self.state.victory_conditions["reputation_crown"] = max(self.state.nations.values(), key=lambda x: x.reputation).name
        bns = max(self.state.nations.values(), key=lambda x: x.times_betrayer).name
        self.state.victory_conditions["betrayer_of_episode"] = bns
        self.state.victory_conditions["survival_award"] = ", ".join([n.name for n in self.state.nations.values() if min(n.food, n.energy, n.military, n.gold) > 0])

    def _make_obs(self, nation_name: str) -> DiplomaticObservation:
        n = self.state.nations[nation_name]
        reports = self.intel_engine.generate_intel(self.state, nation_name)
        
        warns = [f"WARNING: {r} is critically low!" for r in ['food','energy','military','gold'] if getattr(n, r) < 15]
        warn_str = " ".join(warns) + "\n" if warns else ""
        
        intel_str = "\n".join([f"  [{r.source.upper()}, reliability={r.reliability:.1f}] {r.claim}" for r in reports])
        
        ts_str = ", ".join([
            f"{k}={v:.1f} ({'low' if v < 0.3 else 'high' if v > 0.7 else 'medium'})" 
            for k,v in n.trust_toward.items()
        ])
        
        # exact required format
        ws = f"You are {nation_name}. Round {self.state.round_number} of 12.\n"
        ws += f"YOUR RESOURCES: Food={n.food}, Energy={n.energy}, Military={n.military}, Gold={n.gold}.\n"
        ws += warn_str
        ws += f"YOUR GOAL: {n.hidden_agenda}\n"
        ws += f"ACTIVE ALLIANCES: {', '.join(n.active_alliances) if n.active_alliances else 'None'}\n"
        ws += f"TRUST SCORES: {ts_str}\n"
        ws += f"WHAT YOU THINK THEY WANT: {', '.join([f'{k}={v}' for k,v in self.guessed_intents.items()]) if self.guessed_intents else 'None'}\n"
        ws += f"INTEL REPORTS:\n{intel_str}\n"
        ws += f"RECENT EVENTS: {self.state.action_history[-5:]}\n"
        ws += f"PENDING PROPOSALS: {[p.proposal_id for p in self.state.pending_proposals] or 'None'}\n"
        ws += f"BETRAYAL RECORD: {', '.join([f'{nat.name} has betrayed {nat.times_betrayed} times' for nat in self.state.nations.values()])}\n"
        ws += "What is your next action and reasoning?"

        return DiplomaticObservation(
            nation_name=nation_name, round_number=self.state.round_number, total_rounds=12,
            own_resources={"food": n.food, "energy": n.energy, "military": n.military, "gold": n.gold},
            own_hidden_agenda=n.hidden_agenda, active_alliances=n.active_alliances,
            pending_proposals=copy.deepcopy(self.state.pending_proposals),
            public_reputations={nn: nat.reputation for nn, nat in self.state.nations.items()},
            trust_scores=n.trust_toward, estimated_intents={},
            intel_reports=reports, recent_public_actions=self.state.action_history[-5:],
            times_betrayed_public={nn: nat.times_betrayed for nn, nat in self.state.nations.items()},
            world_summary=ws
        )

    def get_state(self) -> DiplomaticState:
        return self.state
