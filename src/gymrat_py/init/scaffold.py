"""Write the gymrat config, runbook stub, and skill file for ``init``.

The config dict is validated before anything is written, so a config that would
fail the schema or the cross-field loop-key checks leaves no files behind. The
bundled skill is read next (only when installing), so a broken install fails
before the runbook stub is created. Among the writes the config lands last and
with an exclusive create, so a failure partway through never leaves a
``gymrat.toml`` pointing at artifacts that were not written, and an existing
config is never silently overwritten.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tomli_w

from gymrat_py.bundled_skill import read_bundled_skill
from gymrat_py.config import CONFIG_FILENAME, validate_config_dict
from gymrat_py.init.wizard import DEFAULT_RUNBOOK_PATH, WizardResult

#: Path, relative to the project root, where init writes and doctor checks for the skill.
SKILL_RELATIVE_PATH = ".claude/skills/gymrat/SKILL.md"

type ArtifactStatus = Literal["created", "exists", "declined"]

_RUNBOOK_STUB = """# Optimization Runbook

## Goal

<!-- Describe the optimization goal here. -->

## Gating metrics

<!-- List the metrics that must not regress. -->

## Constraints

<!-- List any constraints on the optimization. -->

## Approaches to try

<!-- List strategies for the agent to explore. -->

`gymrat supervise` injects this file into the agent's instructions.
"""


@dataclass(frozen=True, slots=True)
class ScaffoldArtifact:
    """The outcome of one scaffold artifact: its path and whether it was written."""

    path: str
    status: ArtifactStatus


@dataclass(frozen=True, slots=True)
class ScaffoldResult:
    """One entry per artifact :func:`scaffold` may write."""

    config: ScaffoldArtifact
    runbook: ScaffoldArtifact
    skill: ScaffoldArtifact


def _build_config(result: WizardResult) -> dict[str, object]:
    """Build the config dict, in declared key order, from the settled wizard answers."""
    config: dict[str, object] = {"bench": result.bench}
    if result.adapter is not None:
        config["adapter"] = result.adapter
    if result.checks is not None:
        config["checks"] = result.checks
    if result.primary is not None:
        config["primary"] = result.primary
    stop: dict[str, object] = {}
    if result.stop_target is not None:
        stop["targetValue"] = result.stop_target
    if result.stop_max_iterations is not None:
        stop["maxIterations"] = result.stop_max_iterations
    if stop:
        config["stop"] = stop
    if result.runbook is not False:
        config["runbook"] = result.runbook.path
    return config


def _write_runbook(base_dir: str, result: WizardResult) -> ScaffoldArtifact:
    if result.runbook is False:
        return ScaffoldArtifact(path=DEFAULT_RUNBOOK_PATH, status="declined")

    runbook_path = result.runbook.path
    # `Path("/base") / "/abs"` yields "/abs", so an absolute runbook path is
    # honored verbatim while a relative one anchors at the base dir.
    full_path = Path(base_dir) / runbook_path
    if full_path.exists():
        return ScaffoldArtifact(path=runbook_path, status="exists")

    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(_RUNBOOK_STUB, encoding="utf-8")
    return ScaffoldArtifact(path=runbook_path, status="created")


def _write_skill(base_dir: str, content: str) -> ScaffoldArtifact:
    full_path = Path(base_dir) / SKILL_RELATIVE_PATH
    if full_path.exists():
        return ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="exists")

    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    return ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="created")


def _write_config(base_dir: str, config: dict[str, object]) -> ScaffoldArtifact:
    full_path = Path(base_dir) / CONFIG_FILENAME
    # tomli_w emits scalar keys before table headers, so a scalar declared after the
    # ``stop`` dict (e.g. ``runbook``) stays top-level instead of nesting under ``[stop]``.
    payload = tomli_w.dumps(config)
    # Exclusive create: an existing gymrat.toml must never be silently overwritten.
    with full_path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return ScaffoldArtifact(path=CONFIG_FILENAME, status="created")


def scaffold(base_dir: str, wizard_result: WizardResult) -> ScaffoldResult:
    """Write the config, runbook stub, and skill file for ``base_dir``.

    Returns a :class:`ScaffoldResult` describing each artifact. Raises a
    :class:`GymratError` when the config fails validation or the bundled skill
    cannot be read, and :class:`FileExistsError` when ``gymrat.toml`` already
    exists — in every failure case no partial scaffold is left behind.
    """
    config = _build_config(wizard_result)
    validate_config_dict(config)

    skill_content = read_bundled_skill() if wizard_result.install_skill else None

    runbook = _write_runbook(base_dir, wizard_result)
    skill = (
        ScaffoldArtifact(path=SKILL_RELATIVE_PATH, status="declined")
        if skill_content is None
        else _write_skill(base_dir, skill_content)
    )
    config_artifact = _write_config(base_dir, config)

    return ScaffoldResult(config=config_artifact, runbook=runbook, skill=skill)
