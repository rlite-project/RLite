from __future__ import annotations

import ray
from loguru import logger
from vllm import RequestOutput, SamplingParams

from rlite.interface.base_engine import BaseEngine, abort_if_no_executor
from rlite.interface.inference.inference_executor import (
    INFERENCE_EXECUTOR_REGISTRY,
    BaseInferenceExecutor
)
from rlite.nn.inference_module import BaseInferenceModule
from rlite.utils.need_parallel import NeedParallel


class RliteInferenceEngine(BaseEngine):
    def __init__(
        self,
        module_or_path: str | BaseInferenceModule,
        executor: BaseInferenceExecutor | str = "vllm",
        **kwargs
    ):
        super().__init__(**kwargs)

        if isinstance(module_or_path, BaseInferenceModule):
            self._module_path = module_or_path.get_checkpoint_path()
        else:
            self._module_path = module_or_path

        if isinstance(executor, str):
            self._executor_cls = INFERENCE_EXECUTOR_REGISTRY[executor]
        else:
            self._executor_cls = executor

    @property
    def executor_cls(self) -> type[BaseInferenceExecutor]:
        return self._executor_cls

    @abort_if_no_executor
    def generate(
        self,
        prompts: list[str],
        sampling_params: dict | SamplingParams | None = None,
        *,
        use_tqdm: bool = True,
        tqdm_desc: str = "Processed prompts"
    ) -> list[RequestOutput]:
        # RliteInferenceEngine.generate() will spread the prompts evenly to all
        # the workers. It is the caller's responsibility to handle the load
        # balance issue.

        if isinstance(sampling_params, dict):
            sampling_params = SamplingParams(**sampling_params)

        if isinstance(prompts, NeedParallel):
            prompts = prompts.unwrap()
            logger.info(
                "NOTE: rlite by default uses data parallel during "
                "inference, and the sequence parallel is handled by the "
                "backend inference engine."
            )

        # Assign prompts to each dp group.
        # Example 1:
        #   len(prompts) = 10, num_dp_groups = 3
        #   num_prompts_for_each_dp = [4, 3, 3]
        # Example 2:
        #   len(prompts) = 2, num_dp_groups = 4
        #   num_prompts_for_each_dp = [1, 1, 0, 0]
        num_dp_groups = len(self.executors)
        num_prompts_for_each_dp = [len(prompts) // num_dp_groups] * num_dp_groups
        for i in range(len(prompts) % num_dp_groups):
            num_prompts_for_each_dp[i] += 1
        dp_group_prompt_indeces = [0] + num_prompts_for_each_dp
        for i in range(1, len(dp_group_prompt_indeces)):
            dp_group_prompt_indeces[i] += dp_group_prompt_indeces[i - 1]

        prompts_for_each_dp = [
            prompts[dp_group_prompt_indeces[i]:dp_group_prompt_indeces[i + 1]]
            for i in range(num_dp_groups)
        ]

        # Generate responses from each dp group
        futures = [
            executor.generate(
                prompts=prompts_for_each_dp[i],
                sampling_params=sampling_params,
                use_tqdm=use_tqdm,
                tqdm_desc=tqdm_desc
            )
            for i, executor in enumerate(self.executors)
        ]
        responses: list[list[RequestOutput]] = [ray.get(future) for future in futures]

        # Deduplicate responses, for each executor, only the response from
        # the first worker is used, as all workers are running the same
        # prompt.
        deduped_responses = sum([x[0] for x in responses], [])
        return deduped_responses

    def pre_build_executors_hook(self, **kwargs) -> dict:
        return {"model_path": self._module_path}
