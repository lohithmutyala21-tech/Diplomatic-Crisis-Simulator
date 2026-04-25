from openenv.core import GenericEnvClient as EnvClient
from diplomatic_crisis_env.models import (
    DiplomaticAction, DiplomaticObservation, DiplomaticState
)

class DiplomaticCrisisEnv(EnvClient[DiplomaticAction, DiplomaticObservation, DiplomaticState]):
    pass
