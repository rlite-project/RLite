import torch
from ray.util.queue import Empty, Queue
from vllm import SamplingParams

from rlite.inference.utils import IndexedGenerationHistory, StopGeneration
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
        max_num_seqs: int = 128,
        use_sleep_mode: bool = False,
        **vllm_kwargs
    ):
        # Discard non-vllm kwargs
        for key in ["dp_rank", "data_parallel_size"]:
            vllm_kwargs.pop(key, None)

        self.max_num_seqs = max_num_seqs

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
        input_queue: Queue,
        output_queue: Queue,
        sampling_params: SamplingParams | dict | None = None,
        is_io_worker: bool = False,
    ):
        index_book = {}
        while True:
            self._keep_stepping(
                is_io_worker,
                output_queue,
                index_book,
                break_if_any_finished=True
            )

            new_inputs = self._fetch_new_inputs(
                input_queue, sampling_params, index_book, is_io_worker
            )

            if new_inputs is not None and isinstance(new_inputs[-1], StopGeneration):
                break

        self._keep_stepping(is_io_worker, output_queue, index_book)

    def _keep_stepping(
        self,
        is_io_worker: bool,
        output_queue: Queue,
        index_book: dict,
        break_if_any_finished: bool = False
    ):
        while self.llm.llm_engine.has_unfinished_requests():
            step_output = self.llm.llm_engine.step()

            any_finished = False
            for output in step_output:
                if output.finished:
                    any_finished = True
                    if is_io_worker:
                        data_item = index_book.pop(output.request_id)
                        data_item.outputs.append(output)
                        output_queue.put(data_item)
            if any_finished and break_if_any_finished:
                return

    def _fetch_new_inputs(
        self,
        input_queue: Queue,
        sampling_params: SamplingParams,
        index_book: dict,
        is_io_worker: bool = False
    ) -> bool:
        if is_io_worker:
            new_inputs: list[IndexedGenerationHistory] = []
            num_unfinished = self.llm.llm_engine.get_num_unfinished_requests()

            while len(new_inputs) + num_unfinished < self.max_num_seqs:
                try:
                    new_inputs.append(input_queue.get(block=False))
                    if isinstance(new_inputs[-1], StopGeneration):
                        break
                except Empty:
                    break

            to_broadcast = [new_inputs]
        else:
            to_broadcast = [None]

        torch.distributed.broadcast_object_list(to_broadcast, src=0)

        new_inputs = to_broadcast[0]

        if not new_inputs:
            return

        request_ids = self.llm._validate_and_add_requests(
            prompts=self._prepare_vllm_prompts(new_inputs),
            params=sampling_params,
            lora_request=None,
            prompt_adapter_request=None,
        )

        for input_item, request_id in zip(new_inputs, request_ids):
            # NOTE: StopGeneration will always be the last, and the request_ids
            #       will have a length of `len(new_inputs) - 1`.
            index_book[request_id] = input_item

        return new_inputs

    def _prepare_vllm_prompts(self, new_inputs: list[IndexedGenerationHistory]) -> list[str]:
        prompts = []
        for item in new_inputs:
            if isinstance(item, IndexedGenerationHistory):  # Omit StopGeneration
                prompts.append(item.inputs[-1])
        return prompts

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
