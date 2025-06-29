import abc

from vllm import RequestOutput, SamplingParams

from rlite.interface import BaseWorker


class BaseInferenceWorker(BaseWorker, abc.ABC):
    @abc.abstractmethod
    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | dict | None = None,
        use_tqdm: bool = True,
        tqdm_desc: str = "Processed prompts"
    ) -> list[RequestOutput]:
        raise NotImplementedError()
