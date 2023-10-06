from .scorer import FullStackScorer, LogProbsScorer
from .sampler import BackpropHiddenSampler, MixedPriorPosteriorHiddenSampler, MultiActionPromptSampler, PriorHiddenSampler


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
        self.hidden_sampler = MixedPriorPosteriorHiddenSampler()
        self.scorer = LogProbsScorer(logp_penalty=logp_penalty)


class FullStackEngine(BackwardEngineConfiguration):
    def __init__(
        self,
        memory_size: int = 5,
    ):
        self.prompt_sampler = MultiActionPromptSampler(memory_size=memory_size)
        self.hidden_sampler = None
        self.scorer = FullStackScorer()


class BackpropLogProbsEngine(BackwardEngineConfiguration):
    def __init__(
        self,
        memory_size: int = 5,
        logp_penalty: float = 1.0,
    ):
        self.prompt_sampler = MultiActionPromptSampler(memory_size=memory_size)
        self.hidden_sampler = BackpropHiddenSampler()
        self.scorer = LogProbsScorer(logp_penalty=logp_penalty)
