import re

import pytest

from gymrat.errors import GymratError
from gymrat.sample_balance import require_balanced_samples


def test_require_balanced_samples_when_count_balances_does_not_raise():
    require_balanced_samples(12, 3)


def test_require_balanced_samples_when_count_is_unbalanced_does_refuse_with_the_nearest_counts():
    with pytest.raises(GymratError) as caught:
        require_balanced_samples(10, 3)

    assert re.findall(r"\d+", str(caught.value)) == ["10", "3", "6", "6", "12"]
