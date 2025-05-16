import torch
from vllm import RequestOutput, SamplingParams

from rlite.interface.inference.inference_worker import BaseWorker
from rlite.third_party.vllm import LLM
from rlite.utils.distributed import CUDAIPCHandle


class VllmWorker(BaseWorker):
    def __init__(
        self,
        *,
        model_path: str,
        seed: int,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        dtype: str = "bfloat16",
        enforce_eager: bool = True,
        gpu_memory_utilization: float = 0.8,
        trust_remote_code: bool = True,
        enable_chunked_prefill: bool = True,
        enable_prefix_caching: bool = True,
        max_model_len: int = 32768,
        use_sleep_mode: bool = False,
        **vllm_kwargs
    ):
        # Discard non-vllm kwargs
        for key in ["dp_rank", "data_parallel_size"]:
            vllm_kwargs.pop(key, None)

        self.llm = LLM(
            # These parameters are fixed
            enable_sleep_mode=use_sleep_mode,
            disable_custom_all_reduce=True,
            distributed_executor_backend="external_launcher",
            # Customizable parameters
            model=model_path,
            seed=seed,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            dtype=dtype,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=enable_prefix_caching,
            max_model_len=max_model_len,
            **vllm_kwargs
        )
        # NOTE: the GPU memory of vLLM is controlled by cuMemAllocator,
        #       which bypasses the pytorch's memory allocator. So the device
        #       of the model weight will always be cuda, even if the memory is
        #       released. To address this issue, we manually record the current
        #       device.
        self._device = "cuda"

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | dict | None = None,
        use_tqdm: bool = True,
        tqdm_desc: str = "Processed prompts",
        return_none: bool = False,
    ) -> list[RequestOutput]:
        if isinstance(prompts, str):
            prompts = [prompts]

        if len(prompts) == 0:
            return []

        if sampling_params is None:
            sampling_params = self.llm.get_default_sampling_params()
        elif isinstance(sampling_params, dict):
            sampling_params = SamplingParams(**sampling_params)

        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            tqdm_desc=tqdm_desc,
        )

        if return_none:
            return
        return outputs

    def cpu(self):
        self.llm.sleep(level=1)
        self.release_memory_cache(gpu=True)
        self._device = "cpu"

    def meta(self):
        self.llm.sleep(level=2)
        self.release_memory_cache(gpu=True)
        self._device = "meta"

    def cuda(self, tags: str | list[str] | None = None):
        if tags is None:
            tags = ["weights", "kv_cache"]
        if isinstance(tags, str):
            tags = [tags]
        if isinstance(tags, list):
            tags = set(tags)

        self.llm.wake_up(tags=tags)
        self._device = "cuda"

    def get_device(self) -> str:
        return self._device

    def update_weight(self, key: str, value: torch.Tensor | CUDAIPCHandle | None = None):
        if key == "END":
            return

        model = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model

        if isinstance(value, CUDAIPCHandle):
            value = value.rebuild()

        try:
            model.load_weights(weights=[(key, value)])
        except KeyError:
            # Ignore the unrecognized weights.
            pass

    def load_state_dict(self, state_dict: dict):
        model = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
        model.load_weights(weights=state_dict.items())

    def state_dict(self) -> dict:
        raise NotImplementedError("Currently VllmWorker does not support state_dict()")

    def sync_weights_to(self, worker):
        raise NotImplementedError("Currently VllmWorker does not support sync_weights_to()")
