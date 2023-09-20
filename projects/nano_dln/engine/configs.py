from .scorer import LogProbsScorer
from .sampler import BackpropHiddenSampler, MultiActionPromptSampler, PriorHiddenSampler


class BackwardEngineConfiguration:
    def __init__(self):
        pass


class BackwardLogProbsEngine(BackwardEngineConfiguration):
    def __init__(
        self,
        memory_size: int = 5,
        logp_penalty: float = 1.0,
    ):
        self.prompt_sampler = MultiActionPromptSampler(memory_size=memory_size)
        self.hidden_sampler = PriorHiddenSampler()
        self.scorer = LogProbsScorer(logp_penalty=logp_penalty)


class BackpropLogProbsEngine(BackwardEngineConfiguration):
    def __init__(
        self,
        memory_size: int = 5,
        logp_penalty: float = 1.0,
    ):
        self.prompt_sampler = MultiActionPromptSampler(memory_size=memory_size)
        self.hidden_sampler = BackpropHiddenSampler()
        self.scorer = LogProbsScorer(logp_penalty=logp_penalty)
