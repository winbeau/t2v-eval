"""
Environment setup: paths, logger, config loading, and dependency checks.

This is the leaf module — no internal imports from sibling modules.
"""

import logging
import os
import sys
import warnings
from pathlib import Path

import yaml

try:
    from .dimensions.registry import (
        CLIP_REQUIRED_DIMENSIONS,
        LONG_DIMENSION_SET,
        PYIQA_REQUIRED_DIMENSIONS,
        default_long_subtasks,
        normalize_subtasks,
        supported_long_subtasks,
    )
except ImportError:
    from vbench_runner.dimensions.registry import (
        CLIP_REQUIRED_DIMENSIONS,
        LONG_DIMENSION_SET,
        PYIQA_REQUIRED_DIMENSIONS,
        default_long_subtasks,
        normalize_subtasks,
        supported_long_subtasks,
    )

# =============================================================================
# Paths
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
VBENCH_ROOT = PROJECT_ROOT / "third_party" / "VBench"

# =============================================================================
# Logger
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vbench_runner")


# =============================================================================
# Logging helpers
# =============================================================================
def setup_log_file_handler(
    log_path: Path,
    rank: int = 0,
    world_size: int = 1,
) -> Path:
    """Attach a file handler for current process logs."""
    target_path = log_path
    if world_size > 1 and rank > 0:
        target_path = log_path.with_name(f"{log_path.stem}.rank{rank}{log_path.suffix}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(target_path.resolve())

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == resolved:
            return target_path

    file_handler = logging.FileHandler(target_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    return target_path


# =============================================================================
# VBench installation & path
# =============================================================================
def check_vbench_installation() -> bool:
    """Check if VBench submodule is properly initialized."""
    vbench_init_file = VBENCH_ROOT / "vbench" / "__init__.py"
    if not vbench_init_file.exists():
        logger.error("=" * 70)
        logger.error("VBench submodule not found or not initialized!")
        logger.error("")
        logger.error("Please run the following commands from project root:")
        logger.error("  git submodule update --init --recursive")
        logger.error("")
        logger.error("Or if cloning fresh:")
        logger.error("  git clone --recurse-submodules <repo_url>")
        logger.error("=" * 70)
        return False
    return True


def setup_vbench_path():
    """Add VBench to Python path."""
    vbench_path = str(VBENCH_ROOT)
    if vbench_path not in sys.path:
        sys.path.insert(0, vbench_path)


# =============================================================================
# Config loading
# =============================================================================
def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str | None) -> Path | None:
    """Resolve path against project root when relative."""
    if not path_str:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def use_vbench_long(config: dict) -> bool:
    """Whether to run VBench-Long backend."""
    vbench_config = config.get("metrics", {}).get("vbench", {})
    backend = str(vbench_config.get("backend", "vbench")).lower()
    return bool(vbench_config.get("use_long", False) or backend in {"long", "vbench_long"})


def get_vbench_subtasks(config: dict) -> list:
    """Get configured subtasks with mode-specific defaults."""
    vbench_config = config.get("metrics", {}).get("vbench", {})
    if "subtasks" in vbench_config:
        configured = normalize_subtasks(vbench_config["subtasks"])
        if configured:
            unknown = sorted(set(configured) - LONG_DIMENSION_SET)
            if unknown:
                logger.warning(
                    "Configured subtasks include unregistered dimensions: %s",
                    unknown,
                )
            return configured

    if use_vbench_long(config):
        profile = vbench_config.get("dimension_profile", "long_6")
        subtasks = default_long_subtasks(profile=profile)
        if str(profile).strip().lower() in {"long_16", "16", "16d", "full", "full_16"}:
            logger.info("Using VBench-Long full 16-dimension profile.")
            logger.debug("Supported long subtasks: %s", supported_long_subtasks())
        else:
            logger.info("Using VBench-Long recommended 6-dimension profile.")
        return subtasks

    return normalize_subtasks(
        vbench_config.get(
            "subtasks",
            ["temporal_flickering", "motion_smoothness"],
        )
    )


# =============================================================================
# Warning / logging filters
# =============================================================================
def configure_warning_filters() -> None:
    """Suppress noisy but non-actionable warnings from third-party libs."""
    warnings.filterwarnings(
        "ignore",
        message=r"The video decoding and encoding capabilities of torchvision are deprecated.*",
        category=UserWarning,
        module=r"torchvision\.io\._video_deprecation_warning",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Importing from timm\.models\.layers is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Importing from timm\.models\.registry is deprecated.*",
        category=FutureWarning,
    )


def configure_third_party_loggers() -> None:
    """Reduce verbose INFO logs from VBench internals."""
    noisy_loggers = [
        "vbench2_beta_long",
        "vbench2_beta_long.subject_consistency",
        "vbench2_beta_long.background_consistency",
        "vbench2_beta_long.utils",
        "vbench",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# =============================================================================
# Compatibility shims
# =============================================================================
def ensure_moviepy_editor_compat() -> None:
    """
    Ensure `moviepy.editor` import path works for VBench-Long.

    Some MoviePy versions expose `VideoFileClip` without the legacy
    `moviepy.editor` module path. VBench-Long imports `moviepy.editor`.
    """
    try:
        import moviepy.editor  # noqa: F401
        return
    except ModuleNotFoundError as e:
        if e.name == "moviepy":
            raise ModuleNotFoundError(
                "moviepy is not installed in current Python env. "
                'Please run: python -m pip install "moviepy<2"'
            ) from e
        if e.name != "moviepy.editor":
            raise

    try:
        import moviepy  # noqa: F401
    except Exception as e:
        raise ModuleNotFoundError(
            "moviepy is not installed in current Python env. "
            'Please run: python -m pip install "moviepy<2"'
        ) from e

    try:
        from moviepy import VideoFileClip
    except Exception:
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip
        except Exception as e:
            raise ModuleNotFoundError(
                "moviepy is installed but VideoFileClip is unavailable. "
                'Try: python -m pip install --upgrade "moviepy<2"'
            ) from e

    import types

    editor_module = types.ModuleType("moviepy.editor")
    editor_module.VideoFileClip = VideoFileClip
    sys.modules["moviepy.editor"] = editor_module
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        logger.info(
            "Applied compatibility shim for `moviepy.editor` "
            "(VBench-Long expects legacy import path)."
        )


# =============================================================================
# Dependency checks
# =============================================================================
def ensure_clip_dependency(subtasks: list[str]) -> None:
    """Ensure CLIP Python package is available for CLIP-based dimensions."""
    extra_clip_dims = {
        "appearance_style",
        "human_action",
        "overall_consistency",
        "temporal_style",
    }
    required = sorted(set(subtasks).intersection(CLIP_REQUIRED_DIMENSIONS.union(extra_clip_dims)))
    if not required:
        return

    try:
        import clip  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name == "pkg_resources":
            raise ModuleNotFoundError(
                "OpenAI CLIP requires `pkg_resources` from `setuptools`. "
                "Install it in current env: `uv pip install -U setuptools` "
                "or `python -m pip install -U setuptools`."
            ) from e
        if e.name != "clip":
            raise ModuleNotFoundError(
                f"`clip` import failed due to missing dependency `{e.name}`. "
                "Please install/repair CLIP environment first. "
                "Suggested order: `uv pip install -U setuptools` then "
                "`uv pip install openai-clip`."
            ) from e
        dims = ", ".join(required)
        raise ModuleNotFoundError(
            "`clip` module is required for VBench subtasks: "
            f"{dims}. Install it in the same environment, e.g. "
            "`uv pip install openai-clip` or "
            "`python -m pip install openai-clip`. "
            "If your mirror does not provide it, use "
            "`python -m pip install git+https://github.com/openai/CLIP.git`."
        ) from e


def ensure_pyiqa_dependency(subtasks: list[str]) -> None:
    """Ensure pyiqa is installed when imaging-quality subtask is enabled."""
    required = sorted(set(subtasks).intersection(PYIQA_REQUIRED_DIMENSIONS))
    if not required:
        return
    try:
        import pyiqa  # noqa: F401
    except Exception as e:
        dims = ", ".join(required)
        raise ModuleNotFoundError(
            "`pyiqa` is required for VBench subtasks: "
            f"{dims}. Install it in current env, e.g. `uv pip install pyiqa`, "
            "and ensure NumPy compatibility (recommend numpy<2 for older pyiqa stacks)."
        ) from e


def ensure_extended_dimension_dependencies(subtasks: list[str]) -> None:
    """Ensure optional heavy dependencies exist for full 16-dimension run."""
    detectron_dims = {
        "object_class",
        "multiple_objects",
        "spatial_relationship",
        "color",
    }
    if set(subtasks) & detectron_dims:
        try:
            import detectron2  # noqa: F401
        except Exception as e:
            missing = ", ".join(sorted(set(subtasks) & detectron_dims))
            raise ModuleNotFoundError(
                "detectron2 is required for VBench subtasks: "
                f"{missing}. Install with:\n"
                "  uv pip install --no-build-isolation "
                '"detectron2 @ git+https://github.com/facebookresearch/detectron2.git"'
            ) from e

    if "scene" in set(subtasks):
        try:
            import fairscale  # noqa: F401
        except Exception as e:
            raise ModuleNotFoundError(
                "fairscale is required for VBench `scene` subtask. Install with:\n"
                "  uv pip install fairscale"
            ) from e


def ensure_pyav_dependency(long_mode: bool) -> None:
    """Ensure PyAV is available for VBench-Long preprocessing video writes."""
    if not long_mode:
        return
    try:
        import av  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name != "av":
            raise
        raise ModuleNotFoundError(
            "PyAV is required by torchvision video IO in VBench-Long. "
            "Install in current env: `uv pip install av` "
            "or `python -m pip install av`."
        ) from e
