from vllm import LLM as BaseLLM

from collections.abc import Sequence
from typing import Any, Callable, Optional, cast, Union

from tqdm.auto import tqdm
from vllm.entrypoints.utils import _validate_truncation_size
from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import deprecate_kwargs


class LLM(BaseLLM):
    @deprecate_kwargs(
        "prompt_token_ids",
        is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
        additional_message="Please use the 'prompts' parameter instead.",
    )
    def generate(
        self,
        prompts: Union[Union[PromptType, Sequence[PromptType]],
                       Optional[Union[str, list[str]]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        prompt_token_ids: Optional[Union[list[int], list[list[int]]]] = None,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        lora_request: Optional[Union[list[LoRARequest], LoRARequest]] = None,
        priority: Optional[list[int]] = None,
        tqdm_desc: str = "Processed prompts",
    ) -> list[RequestOutput]:
        """Generates the completions for the input prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See [PromptType][vllm.inputs.PromptType]
                for more details about the format of each prompts.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
                When it is a single value, it is applied to every prompt.
                When it is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: If `True`, shows a tqdm progress bar.
                If a callable (e.g., `functools.partial(tqdm, leave=False)`),
                it is used to create the progress bar.
                If `False`, no progress bar is created.
            lora_request: LoRA request to use for generation, if any.
            priority: The priority of the requests, if any.
                Only applicable when priority scheduling policy is enabled.

        Returns:
            A list of `RequestOutput` objects containing the
            generated completions in the same order as the input prompts.

        Note:
            Using `prompts` and `prompt_token_ids` as keyword parameters is
            considered legacy and may be deprecated in the future. You should
            instead pass them via the `inputs` parameter.
        """
        model_config = self.llm_engine.model_config
        runner_type = model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.generate() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model.")

        if prompt_token_ids is not None:
            parsed_prompts = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, list[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            parsed_prompts = cast(Union[PromptType, Sequence[PromptType]],
                                  prompts)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = self.get_default_sampling_params()

        tokenization_kwargs: dict[str, Any] = {}
        truncate_prompt_tokens = None
        if isinstance(sampling_params, SamplingParams):
            truncate_prompt_tokens = sampling_params.truncate_prompt_tokens

        _validate_truncation_size(model_config.max_model_len,
                                  truncate_prompt_tokens, tokenization_kwargs)

        # Add any modality specific loras to the corresponding prompts
        lora_request = self._get_modality_specific_lora_reqs(
            parsed_prompts, lora_request)

        self._validate_and_add_requests(
            prompts=parsed_prompts,
            params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            priority=priority,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm, tqdm_desc=tqdm_desc)
        return self.engine_class.validate_outputs(outputs, RequestOutput)

    def _run_engine(
        self,
        *,
        use_tqdm: Union[bool, Callable[..., tqdm]] = True,
        tqdm_desc: str = "Processed prompts",
    ) -> list[Union[RequestOutput, PoolingRequestOutput]]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            pbar = tqdm_func(
                total=num_requests,
                desc=tqdm_desc,
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )

        # Run the engine.
        outputs: list[Union[RequestOutput, PoolingRequestOutput]] = []
        total_in_toks = 0
        total_out_toks = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            n = len(output.outputs)
                            assert output.prompt_token_ids is not None
                            total_in_toks += len(output.prompt_token_ids) * n
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(
                                len(stp.token_ids) for stp in output.outputs)
                            out_spd = (total_out_toks /
                                       pbar.format_dict["elapsed"])
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s")
                            pbar.update(n)
                        else:
                            pbar.update(1)
                        if pbar.n == num_requests:
                            pbar.refresh()

        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))
