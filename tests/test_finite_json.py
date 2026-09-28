"""Behavioral tests for nulling non-finite floats before JSON serialization.

``null_non_finite`` walks a JSON-bound value and replaces every NaN and
infinity with ``None`` at any depth, so ``json.dumps(..., allow_nan=False)``
accepts the result.
"""

import json
import math

from hypothesis import example, given
from hypothesis import strategies as st

from gymrat.finite_json import null_non_finite

_json_children = st.recursive(
    st.floats(allow_nan=True, allow_infinity=True) | st.text() | st.integers() | st.none(),
    lambda children: (
        st.lists(children) | st.lists(children).map(tuple) | st.dictionaries(st.text(), children)
    ),
)


def _expected_after_null_non_finite(value: object) -> object:
    """Independently compute the transformation ``null_non_finite`` should produce."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _expected_after_null_non_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_expected_after_null_non_finite(item) for item in value]
    return value


@given(value=_json_children)
@example((math.nan, math.inf, 1.5))
@example({"excluded": (-math.inf, 2.0, math.nan)})
@example([("name", math.nan)])
@example(((math.inf, (math.nan,)),))
def test_null_non_finite_when_value_holds_non_finite_floats_at_any_depth_does_serialize_with_allow_nan_false(
    value: object,
):
    round_tripped = json.loads(json.dumps(null_non_finite(value), allow_nan=False))

    assert round_tripped == _expected_after_null_non_finite(value)
