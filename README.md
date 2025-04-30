# RLite

RLite aims to provide a unified interface for LLM training and inference, regardless of the backend parallelism or model used.

Check the [documents](//doc) for more details.

## Installation

#### Start Locally

We use `conda` to manage our computation environment.

1. Create a conda environment:

```
conda create -n rlite python==3.12
conda activate rlite
```

2. Install CUDA if not exist

3. Install common dependencies

```bash
# install vllm
pip install vllm accelerate

# flash attention 2 (make sure you have 64 CPU cores)
MAX_JOBS=64 pip install flash-attn --no-build-isolation

# Install fashinfer for faster inference
pip install flashinfer-python==0.2.2.post1 -i https://flashinfer.ai/whl/cu124/torch2.6
```

4. Install `rlite`

```bash
pip install -e .
```

<details>
<summary>Developer's guide.</summary>

We use `pre-commit` and `git cz` to sanitize the commits. You can run `pre-commit` before `git cz` to avoid repeatedly input the commit messages.

```bash
pip install pre-commit
# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
# Install this emoji-style tool
sudo npm install -g git-cz --no-audit --verbose --registry=https://registry.npmmirror.com

# Install rlite
pip install -e ".[dev]"
```

##### Code Style

- Single line code length is 99 characters, comments and documents are 79 characters.
- Write unit tests for atomic capabilities to ensure that `pytest` does not throw an error.

Run `pre-commit` to automatically lint the code:

```
pre-commit run --all-files
```

##### Run Unit Tests:

```bash
# Only run tests
pytest

# Run tests and output test code coverage report
pytest --cov=rlite
```

</details>

Debug in `rlite` is very simple thanks to `ray debugger`. You can add breakpoint anywhere in your code by inserting `breakpoint()`. This will trigger debug at the insertion point.
