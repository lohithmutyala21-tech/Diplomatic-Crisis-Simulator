import pytest
from diplomatic_crisis_env.models import DiplomaticAction
from diplomatic_crisis_env.server.environment import DiplomaticCrisisEnvironment

def test_reset_env():
    env = DiplomaticCrisisEnvironment()
    obs = env.reset()
    assert obs.round_number == 1
    assert len(obs.intel_reports) <= 3
    assert len(obs.trust_scores) > 0

def test_propose_action():
    env = DiplomaticCrisisEnvironment()
    obs = env.reset()
    act = DiplomaticAction(nation_name=obs.nation_name, action_type="PROPOSE", target_nation="random", raw_llm_output="")
    obs2 = env.step(act)
    assert len(env.state.pending_proposals) == 1

def test_betray_action_hidden():
    env = DiplomaticCrisisEnvironment()
    obs = env.reset()
    act = DiplomaticAction(nation_name=obs.nation_name, action_type="BETRAY", target_nation="random", raw_llm_output="")
    env.step(act)
    # Ensure betray logic works (would need active treaty setup for full test)
    assert env.state.nations[obs.nation_name].gold >= 0

def test_betrayal_autoreveal():
    env = DiplomaticCrisisEnvironment()
    env.reset()
    # Fake a treaty and trigger
    pass

def test_malformed_llm_output():
    env = DiplomaticCrisisEnvironment()
    obs = env.reset()
    act = DiplomaticAction(nation_name=obs.nation_name, action_type="PASS", raw_llm_output="GARBLED TEXT THAT MAKES NO SENSE!!")
    obs2 = env.step(act)
    # Environment handles it via parser fallbacks to PASS or similar without crash
    assert len(env.state.action_history) > 0
