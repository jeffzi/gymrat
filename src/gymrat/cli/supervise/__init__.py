"""The ``gymrat supervise`` subpackage: pre-flight, progress reporter, frame, and state.

The command itself is :mod:`gymrat.cli.commands.supervise`.

Import the submodules directly.  Re-exporting them here would make importing any
one of them — including the Rich-free :mod:`~gymrat.cli.supervise.reducer` and
:mod:`~gymrat.cli.supervise.text` — pull in the terminal view layer.
"""
