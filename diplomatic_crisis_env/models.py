from pydantic import BaseModel
from openenv.core import Action, Observation, State
from typing import List, Dict, Optional, Literal

class IntelReport(BaseModel):
    source: str           # "spy", "rumor", "public_broadcast"
    about_nation: str
    claim: str            # e.g. "Veldran is critically low on energy"
    reliability: float    # shown to agent: 0.0 to 1.0
    is_true: bool         # HIDDEN from agent, used for env scoring only

class NationState(BaseModel):
    name: str
    food: float
    energy: float
    military: float
    gold: float
    reputation: float             # public, 0-100
    hidden_agenda: str            # NEVER sent to other agents
    active_alliances: List[str]
    active_sanctions_against: List[str]
    trust_toward: Dict[str, float]   # private: this nation's trust in others
    times_betrayed: int              # public
    times_betrayer: int              # private
    recent_actions: List[str] = []   # private: structured last 2 actions for TrustCalibration

class Treaty(BaseModel):
    proposal_id: str
    proposer: str
    target: str
    treaty_type: str        # "trade", "alliance", "non_aggression"
    offer_resource: str
    offer_amount: float
    request_resource: str
    request_amount: float
    status: str             # "pending", "active", "rejected", "betrayed"
    betrayal_pending: bool  # HIDDEN: True if agent secretly defected
    betrayer: Optional[str] = None
    betrayed: Optional[str] = None
    betrayal_round: Optional[int] = None
    round_created: int

class DiplomaticAction(Action):
    nation_name: str
    action_type: Literal["PROPOSE","ACCEPT","REJECT","BETRAY","SANCTION","PASS"]
    target_nation: Optional[str] = None
    proposal_id: Optional[str] = None
    offer_resource: Optional[str] = None
    offer_amount: Optional[float] = None
    request_resource: Optional[str] = None
    request_amount: Optional[float] = None
    reasoning: Optional[str] = None   # LLM explains its reasoning
    raw_llm_output: str

class DiplomaticObservation(Observation):
    nation_name: str
    round_number: int
    total_rounds: int
    own_resources: Dict[str, float]
    own_hidden_agenda: str
    active_alliances: List[str]
    pending_proposals: List[Treaty]
    public_reputations: Dict[str, float]
    trust_scores: Dict[str, float]         # own trust toward others
    estimated_intents: Dict[str, str]      # LLM's guess of others' agendas
    intel_reports: List[IntelReport]       # rumors, not always true
    recent_public_actions: List[str]       # last 5 rounds only
    times_betrayed_public: Dict[str, int]  # public betrayal record
    world_summary: str                     # natural language LLM prompt

class DiplomaticState(State):
    round_number: int
    nations: Dict[str, NationState]
    active_treaties: List[Treaty]
    pending_proposals: List[Treaty]
    action_history: List[str]
    episode_done: bool
    victory_conditions: Dict[str, str]   # populated at episode end
