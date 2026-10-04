"""The environment protocol between the supervisor and the commands it spawns.

The supervisor marks every command the agent runs through its tool host, and
hands it the trace context the command's span joins. The command side reads the
same names back. Both sides import them from here so the protocol is declared
once; this module imports nothing, so a command can read it without loading the
supervisor.
"""

COMMAND_ORIGIN_ENV = "GYMRAT_COMMAND_ORIGIN"
"""Set on a command the supervised agent runs through the in-process tool host."""

TOOL_ORIGIN = "tool"
"""The :data:`COMMAND_ORIGIN_ENV` value naming a run the agent made through a tool."""

TRACEPARENT_ENV = "GYMRAT_TRACEPARENT"
"""The W3C ``traceparent`` of the supervisor's run span, for the command's span to join."""
