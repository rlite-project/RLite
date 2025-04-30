from enum import Enum


class ParallelType(Enum):
    DP = "dp"
    SEQ = "seq"


class NeedParallel:
    def __init__(self, data: any, /, *, type: str = "dp"):
        allowed_types = [
            "dp", "data parallel", "data_parallel", "data-parallel",
            "seq", "sequence", "sequence parallel", "sequence_parallel", "sequence-parallel",
            "ctx", "context", "context parallel", "context_parallel", "context-parallel",
        ]
        assert type in allowed_types, f"Invalid parallel type: {type}"
        if type in ["dp", "data parallel", "data_parallel", "data-parallel"]:
            self.parallel_type = ParallelType.DP
        else:
            self.parallel_type = ParallelType.SEQ

        self.data = data
