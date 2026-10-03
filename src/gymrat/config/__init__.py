"""Config-file schema and loading for gymrat.

The on-disk ``gymrat.toml`` file is validated by pydantic against the frozen
dataclasses themselves. Consumers receive plain frozen dataclasses and values;
the pydantic metadata lives only in their class definitions, in field
annotations and a ``__pydantic_config__`` class attribute.
:func:`~gymrat.config.resolve.load_config_file_collecting` reads, parses, and
validates the file, returning every problem alongside an ``exists`` flag so a
caller can report all issues at once.

Config keys are snake_case (``timeout_seconds``, ``unstable_noise_pct``,
``stop.max_iterations``), matching the frozen dataclass attributes. Validation
error paths always name the snake_case key the user wrote.
"""
