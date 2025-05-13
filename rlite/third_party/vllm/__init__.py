from importlib.metadata import PackageNotFoundError, version

from packaging import version as pkg_version


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = "vllm"
package_version = get_version(package_name)


supported_versions = ["0.8.3", "0.8.4", "0.8.5", "0.8.5.post1"]


if package_version not in supported_versions:
    raise ImportError(f"vllm 版本 {package_version} 不支持。支持的版本为: {supported_versions}。")


if package_version in ["0.8.3", "0.8.4"]:
    from .vllm_0_8_34 import LLM

elif package_version in ["0.8.5", "0.8.5.post1"]:
    from .vllm_0_8_5 import LLM


__all__ = [
    "LLM",
]
