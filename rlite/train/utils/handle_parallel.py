import torch

from rlite.utils.need_parallel import NeedParallel, ParallelType


def _get_chunk_indices(data_size: int, parallel_size: int) -> list[int]:
    # Eg: len(arg)=7, dp_size=4 : [2, 2, 2, 1]
    chunk_sizes = [data_size // parallel_size] * parallel_size
    for i in range(data_size % parallel_size):
        chunk_sizes[i] += 1
    chunk_indeces = [0] * (parallel_size + 1)
    for i in range(1, len(chunk_indeces)):
        chunk_indeces[i] = chunk_sizes[i - 1] + chunk_indeces[i - 1]
    return chunk_indeces


def _handle_need_parallel(
    arg: NeedParallel,
    unwrap: bool = True,
    parallel_size: int = 1
) -> list[any]:
    if arg.parallel_type == ParallelType.DP:
        assert len(arg.data) >= parallel_size, "Data size smaller than parallel_size!"

        if isinstance(arg.data, torch.Tensor):
            data_chunks = [*torch.chunk(arg.data, parallel_size, dim=0)]
        else:
            chunk_indeces = _get_chunk_indices(len(arg.data), parallel_size)
            data_chunks = [
                arg.data[chunk_indeces[i]:chunk_indeces[i + 1]]
                for i in range(parallel_size)
            ]

        if not unwrap:
            data_chunks = [NeedParallel(data_chunk, type="dp") for data_chunk in data_chunks]

        return data_chunks

    elif arg.parallel_type == ParallelType.SEQ:
        assert len(arg.data[0]) >= parallel_size, "Sequence length smaller than parallel_size!"
        assert all(len(x) == len(arg.data[0]) for x in arg.data), (
            "All data chunks must have the same sequence length!"
        )

        if isinstance(arg.data, torch.Tensor):
            data_chunks = [*torch.chunk(arg.data, parallel_size, dim=1)]
        else:
            chunk_indeces = _get_chunk_indices(len(arg.data[0]), parallel_size)
            data_chunks = [
                [
                    arg.data[i][chunk_indeces[j]:chunk_indeces[j + 1]]
                    for i in range(len(arg.data))
                ]
                for j in range(parallel_size)
            ]

        if not unwrap:
            data_chunks = [NeedParallel(data_chunk, type="seq") for data_chunk in data_chunks]

        return data_chunks


def prepare_parallel_args(
    args: tuple,
    unwrap: bool = True,
    dp_size: int = 1
) -> list[list[any]]:
    if len(args) == 0:
        return [[]] * dp_size
    arg_list = []
    for arg in args:
        if isinstance(arg, NeedParallel):
            arg_list.append(_handle_need_parallel(arg, unwrap, dp_size))
        else:
            arg_list.append([arg] * dp_size)
    return [[arg_list[j][i] for j in range(len(arg_list))] for i in range(dp_size)]


def prepare_parallel_kwargs(
    kwargs: dict[str, any],
    unwrap: bool = True,
    dp_size: int = 1
) -> list[dict[str, any]]:
    if len(kwargs) == 0:
        return [{} for _ in range(dp_size)]
    new_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, NeedParallel):
            new_kwargs[key] = _handle_need_parallel(value, unwrap, dp_size)
        else:
            new_kwargs[key] = [value] * dp_size
    return [{key: value[i] for key, value in new_kwargs.items()} for i in range(dp_size)]
