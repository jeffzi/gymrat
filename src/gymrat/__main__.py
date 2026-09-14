"""Allow ``python -m gymrat`` to launch the CLI.

Calls the typer app with an explicit ``prog_name`` so the usage line reads
``python -m gymrat.cli.app``, matching the canonical entry module.
"""

from gymrat.cli.app import app

if __name__ == "__main__":
    app(prog_name="python -m gymrat.cli.app")
