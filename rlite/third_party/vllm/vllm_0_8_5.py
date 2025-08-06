import warnings
from typing import Optional, Sequence, Union, cast

from tqdm import tqdm
from vllm import LLM as BaseLLM
from vllm.inputs import PromptType
from vllm.model_executor.guided_decoding.guided_fields import (
    GuidedDecodingRequest
)
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams


class LLM(BaseLLM):
    def generate(
        self,
        prompts=None,
        sampling_params=None,
        prompt_token_ids=None,
        use_tqdm=True,
        lora_request=None,
        prompt_adapter_request=None,
        guided_options_request=None,
        priority=None,
        tqdm_desc="Processed prompts",
    ):
        runner_type = self.llm_engine.model_config.runner_type
        if runner_type not in ["generate", "transcription"]:
            messages = [
                "LLM.generate() is only supported for (conditional) generation "
                "models (XForCausalLM, XForConditionalGeneration).",
            ]

            supported_runner_types = self.llm_engine.model_config \
                .supported_runner_types
            if "generate" in supported_runner_types:
                messages.append(
                    "Your model supports the 'generate' runner, but is "
                    f"currently initialized for the '{runner_type}' runner. "
                    "Please initialize vLLM using `--task generate`.")

            raise ValueError(" ".join(messages))

        if prompt_token_ids is not None:
            parsed_prompts = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, list[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            parsed_prompts = cast(Union[PromptType, Sequence[PromptType]],
                                  prompts)

        if isinstance(guided_options_request, dict):
            if len(guided_options_request) > 1:
                raise ValueError(
                    "You can only use one guided decoding but multiple is "
                    f"specified: {guided_options_request}")
            guided_options_request = GuidedDecodingRequest(
                **guided_options_request)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = self.get_default_sampling_params()

        self._validate_and_add_requests(
            prompts=parsed_prompts,
            params=sampling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            guided_options=guided_options_request,
            priority=priority)

        outputs = self._run_engine(use_tqdm=use_tqdm, tqdm_desc=tqdm_desc)
        return self.engine_class.validate_outputs(outputs, RequestOutput)

    def _run_engine(self, *, use_tqdm: bool, tqdm_desc: str):
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc=tqdm_desc,
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0: .2f} toks/s, "
                         f"output: {0: .2f} toks/s"),
            )

        # Run the engine.
        outputs = []
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
                                f"est. speed input: {in_spd: .2f} toks/s, "
                                f"output: {out_spd: .2f} toks/s")
                            pbar.update(n)
                        else:
                            pbar.update(1)

        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))

    def _add_request(self, *args, **kwargs) -> str:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id, *args, **kwargs)
        return request_id

    def _validate_and_add_requests(
        self,
        prompts,
        params,
        lora_request,
        prompt_adapter_request,
        guided_options: Optional[GuidedDecodingRequest] = None,
        priority: Optional[list[int]] = None,
    ) -> None:
        if guided_options is not None:
            warnings.warn(
                "guided_options_request is deprecated, use "
                "SamplingParams.guided_decoding instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(prompts, (str, dict)):
            # Convert a single prompt to a list.
            prompts = [prompts]

        num_requests = len(prompts)
        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")
        if isinstance(lora_request,
                      list) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request "
                             "must be the same.")

        for sp in params if isinstance(params, list) else (params, ):
            if isinstance(sp, SamplingParams):
                self._add_guided_params(sp, guided_options)

                # We only care about the final output
                sp.output_kind = RequestOutputKind.FINAL_ONLY

        # Add requests to the engine.
        request_ids = []
        for i, prompt in enumerate(prompts):
            request_id = self._add_request(
                prompt,
                params[i] if isinstance(params, Sequence) else params,
                lora_request=lora_request[i] if isinstance(
                    lora_request, Sequence) else lora_request,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority[i] if priority else 0,
            )
            request_ids.append(request_id)

        return request_ids
