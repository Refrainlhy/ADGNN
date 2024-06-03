__version__ = "0.1.2"

__all__ = [
    "BaseInfluenceModule",
    "BaseObjective",
    "AutogradInfluenceModule",
    "CGInfluenceModule",
    "LiSSAInfluenceModule",
    "log_regression",
    "evaluation",
    "LogReg",
    "influence_score_computation"
]

from torch_influence.base import BaseInfluenceModule, BaseObjective
from torch_influence.modules import AutogradInfluenceModule, CGInfluenceModule, LiSSAInfluenceModule
from torch_influence.encoder import log_regression, evaluation, LogReg
from torch_influence.score import influence_score_computation