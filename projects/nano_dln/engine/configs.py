from dataclasses import dataclass
from .scorer import FullStackScorer, VIScorer, Scorer
from .sampler import MultiActionPromptSampler, PriorHiddenSampler, PromptSampler, HiddenSampler


class BackwardEngineConfiguration:
    def __init__(self):
        pass


class VIBackwardEngine(BackwardEngineConfiguration):
    def __init__(
        self,
        memory_size: int = 0,
        logp_penalty: float = 1.0,
    ):
        self.prompt_sampler = MultiActionPromptSampler(memory_size)
        self.hidden_sampler = PriorHiddenSampler()
        self.scorer = VIScorer(logp_penalty=logp_penalty)
