import re
from pathlib import Path

import pytest

from gymrat.config import CONFIG_DEFAULTS
from tests._ansi import normalize
from tests._markdown import md_section

README_PATH = Path(__file__).resolve().parents[1] / "README.md"
QUICK_EXAMPLE = "## Quick example"
CONFIGURATION = "## Configuration"


def _section(heading: str) -> str:
    """The README text from ``heading`` up to the next second-level heading."""
    return md_section(README_PATH.read_text(encoding="utf-8"), heading, r"\n## ")


def _paragraph(heading: str, marker: str) -> str:
    """The paragraph under ``heading`` containing ``marker``, unwrapped to one line."""
    paragraphs = (normalize(item) for item in _section(heading).split("\n\n"))
    return next((item for item in paragraphs if marker in item), "")


def test_readme_configuration_example_when_read_does_set_samples_to_the_config_default():
    section = _section(CONFIGURATION)

    stated = re.findall(r"^samples = (\d+)\b", section, re.MULTILINE)

    assert stated == [str(CONFIG_DEFAULTS.samples)]


@pytest.mark.parametrize(
    ("heading", "template"),
    [
        pytest.param(QUICK_EXAMPLE, "{} times per side", id="quick-example-runs"),
        pytest.param(QUICK_EXAMPLE, "{} paired samples", id="quick-example-header"),
        pytest.param(CONFIGURATION, "{}-sample measurements", id="probe"),
    ],
)
def test_readme_when_section_counts_default_samples_does_use_the_config_default(
    heading: str, template: str
):
    text = normalize(_section(heading))

    assert template.format(CONFIG_DEFAULTS.samples) in text


@pytest.mark.parametrize(
    ("marker", "phrases"),
    [
        pytest.param(
            "every position",
            (
                "after every other target",
                "multiple of",
                "even",
                "`N`",
                "`2N`",
                "refuses",
                "before a bench or hook runs",
                "`gymrat doctor`",
            ),
            id="balanced-order",
        ),
        pytest.param(
            "`detectable_pct`",
            ("within noise", "smallest change", "JSON", "session log"),
            id="detectable-change",
        ),
    ],
)
def test_readme_how_verdicts_work_when_paragraph_inspected_does_state_its_rule(
    marker: str, phrases: tuple[str, ...]
):
    text = _paragraph("## How verdicts work", marker)

    missing = [phrase for phrase in phrases if phrase not in text]
    assert missing == []
