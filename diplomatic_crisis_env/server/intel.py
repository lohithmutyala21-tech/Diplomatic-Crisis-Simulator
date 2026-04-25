import random
from typing import List
from diplomatic_crisis_env.models import DiplomaticState, IntelReport

class IntelEngine:
    def __init__(self, rng=None):
        import random
        self.rng = rng if rng else random.Random()

    def generate_intel(self, state: DiplomaticState, for_nation: str) -> List[IntelReport]:
        reports = []
        num_reports = self.rng.randint(1, 3)
        
        for _ in range(num_reports):
            target = self.rng.choice([n for n in state.nations.values() if n.name != for_nation]) if len(state.nations) > 1 else None
            if not target: continue
            
            stype = self.rng.choice(["spy", "rumor", "public_broadcast"])
            res = self.rng.choice(['food', 'energy', 'military', 'gold'])
            claim = f"{target.name} has ~{int(getattr(target, res))} {res}."
            is_true = True
            rel = self.rng.uniform(0.7, 0.9)
            
            if stype == "spy":
                rel = self.rng.uniform(0.7, 0.9)
                if target.hidden_agenda == "expand_territory" and self.rng.random() < 0.4:
                    is_true, rel, claim = False, self.rng.uniform(0.4, 0.7), f"{target.name} is critically low on military."
                if target.hidden_agenda == "avoid_war":
                    rel = min(1.0, rel + 0.1)
            elif stype == "rumor":
                is_true, rel = self.rng.random() < 0.6, self.rng.uniform(0.4, 0.6)
                if target.hidden_agenda == "avoid_war":
                    is_true, rel = self.rng.random() < 0.7, min(1.0, rel + 0.1)
                elif target.hidden_agenda == "maximize_alliances":
                    is_true, claim = True, f"{target.name} is looking for new allies."
            else: # public_broadcast
                rel, claim = 1.0, f"Public records show {target.name} has {int(target.reputation)} reputation."

            reports.append(IntelReport(
                source=stype, about_nation=target.name, claim=claim,
                reliability=min(1.0, rel), is_true=is_true
            ))
            
        return reports
