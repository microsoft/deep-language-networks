from dataclasses import dataclass
from .scorer import FullStackScorer, VIScorer, Scorer
from .sampler import MultiActionPromptSampler, PriorHiddenSampler, PromptSampler, HiddenSampler


@dataclass
class EngineConfiguration:
    hidden_sampler: HiddenSampler
    prompt_sampler: PromptSampler
    scorer: Scorer


vi_engine_configuration = EngineConfiguration(
    hidden_sampler=PriorHiddenSampler,
    prompt_sampler=MultiActionPromptSampler,
    scorer=VIScorer,
)

full_stack_engine_configuration = EngineConfiguration(
    hidden_sampler=None,
    prompt_sampler=MultiActionPromptSampler,
    scorer=FullStackScorer,
)
