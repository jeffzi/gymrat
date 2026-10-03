"""The OTLP endpoint every tracing entry point reads, and the rule for reading it.

Command tracing, the tracing provider, and ``gymrat export`` all name the same
environment variable and trim it the same way. This module imports nothing, so
a command can read the endpoint without loading the provider.
"""

ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
"""Environment variable every tracing entry point reads the OTLP endpoint from."""


def otlp_endpoint(value: str | None) -> str | None:
    """Apply the endpoint rule shared by command tracing, the provider, and ``export``.

    Args:
        value: A raw endpoint, from ``--endpoint`` or ``OTEL_EXPORTER_OTLP_ENDPOINT``.

    Returns:
        The endpoint with surrounding whitespace trimmed, or ``None`` when
        nothing is left, which means "no endpoint".
    """
    return (value or "").strip() or None
