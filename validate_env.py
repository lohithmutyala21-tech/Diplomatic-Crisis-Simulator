import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from diplomatic_crisis_env.server.environment import DiplomaticCrisisEnvironment
from diplomatic_crisis_env.models import DiplomaticAction, DiplomaticObservation
from diplomatic_crisis_env.server.agents import RandomAgent

def main():
    print("Running Sanity Check...")
    env = DiplomaticCrisisEnvironment(seed=42)
    agent = RandomAgent()
    
    betrayal_triggered = False
    
    tests_passed = 0
    
    # Test 1: No crashes in full rollout
    try:
        for ep in range(5):
            obs = env.reset()
            done = False
            
            while not done:
                curr_nation = obs.nation_name
                act_obj = agent.act(obs)
                
                # Test 2: step() always returns (obs, reward, done)
                step_res = env.step(act_obj)
                assert len(step_res) == 3, "step() did not return 3 elements"
                next_obs, reward, done = step_res
                assert isinstance(next_obs, DiplomaticObservation), "Returned obs is not DiplomaticObservation"
                assert isinstance(reward, (int, float)), "Returned reward is not numeric"
                assert isinstance(done, bool), "Returned done is not boolean"
                
                # Test 3: Trust values stay within [0,1]
                for trust in next_obs.trust_scores.values():
                    assert 0.0 <= trust <= 1.0, f"Trust value out of bounds: {trust}"
                        
                if act_obj.action_type == "BETRAY":
                    betrayal_triggered = True
                    
                obs = next_obs
        tests_passed += 4 # Rollout, step signature, trust bounds, no crashes
        
    except Exception as e:
        print(f"Crash during rollout: {e}")
        sys.exit(1)
        
    # Test 5: Betrayal actually triggers at least once in 5 episodes
    assert betrayal_triggered, "Betrayal never triggered in 5 episodes (random agent)"
    tests_passed += 1
    
    if tests_passed == 5:
        print("✅ Environment validation passed (5/5 tests)")

if __name__ == "__main__":
    main()
