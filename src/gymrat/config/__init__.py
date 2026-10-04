"""Config-file schema and loading for gymrat.

The on-disk ``gymrat.toml`` file is validated by pydantic against the frozen
dataclasses themselves. Consumers receive plain frozen dataclasses and values;
the pydantic metadata lives only in their class definitions, in field
annotations and a ``__pydantic_config__`` class attribute. Two entry points
share one read/parse/validate pipeline:

- :func:`~gymrat.config.resolve.load_config_file` raises a :class:`~gymrat.errors.GymratError`
  on the first problem.
- :func:`~gymrat.config.resolve.load_config_file_collecting` returns every
  problem alongside an ``exists`` flag, for callers that want to report all
  issues at once.

Config keys are snake_case (``timeout_seconds``, ``unstable_noise_pct``,
``stop.max_iterations``), matching the frozen dataclass attributes. Validation
error paths always name the snake_case key the user wrote.
"""
