class BaseAgent:
    """
    Base class for all Diplomatic Crisis Simulator agents.
    Any custom agent must inherit from this class and implement the act() method.
    """
    def act(self, obs):
        raise NotImplementedError("Agents must implement the act method.")
