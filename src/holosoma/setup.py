import platform
import sys
from os import getenv

from setuptools import setup

UNITREE_VERSION = "0.1.2"
BOOSTER_VERSION = "0.1.0"

PLATFORM_MAP = {
    "x86_64": "linux_x86_64",
    "aarch64": "linux_aarch64",
}

pyvers = f"cp{sys.version_info.major}{sys.version_info.minor}"
platform_str = PLATFORM_MAP.get(platform.machine(), "linux_x86_64")


def sdk_dependency(package_name: str, version: str, repo_env_var: str) -> str:
    repo = getenv(repo_env_var, "").strip().rstrip("/")
    if not repo:
        return package_name
    wheel = f"{package_name}-{version}-{pyvers}-{pyvers}-{platform_str}.whl"
    return f"{package_name} @ {repo}/releases/download/{version}/{wheel}"


unitree_dep = sdk_dependency("unitree_sdk2", UNITREE_VERSION, "HOLOSOMA_UNITREE_SDK2_REPO")
booster_dep = sdk_dependency("booster_robotics_sdk", BOOSTER_VERSION, "HOLOSOMA_BOOSTER_SDK_REPO")

setup(
    extras_require={
        "unitree": [unitree_dep],
        "booster": [booster_dep],
    },
    # Entry points are declared in pyproject.toml [project.entry-points.*]
)
