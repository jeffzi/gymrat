"""The names gymrat's MCP tools go by inside the agent SDK.

The SDK names each tool ``mcp__<server key>__<tool name>``. The server key is the
``mcp_servers`` key the driver registers the tool host under, and the tool names
are the ones the tool host defines, so the composed names below match what the
agent calls only while all three come from here.
"""

MCP_SERVER = "gymrat"
"""The key the driver registers gymrat's tool host under."""

ITERATE_TOOL_NAME = "iterate"
"""The tool name the tool host gives its iterate tool."""

PROBE_TOOL_NAME = "probe"
"""The tool name the tool host gives its probe tool."""

ITERATE_TOOL = f"mcp__{MCP_SERVER}__{ITERATE_TOOL_NAME}"
"""The iterate tool's name as the agent calls it."""

PROBE_TOOL = f"mcp__{MCP_SERVER}__{PROBE_TOOL_NAME}"
"""The probe tool's name as the agent calls it."""
