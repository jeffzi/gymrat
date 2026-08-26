"""Constants shared between ``config.py`` and ``config_env.py``.

Splitting these out of ``config.py`` breaks a circular import: the config-file
schema in ``config.py`` needs the env-var readers ``config_env.py`` defines,
while ``config_env.py``'s timeout reader needs the timeout cap ``config.py``
would otherwise define.
"""

MAX_TIMEOUT_SECONDS = 2_147_483
"""Largest ``timeout_seconds`` a 32-bit millisecond timer can represent."""
