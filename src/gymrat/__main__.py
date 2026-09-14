"""Allow ``python -m gymrat`` to launch the CLI.

Calls the typer app through the shared console-script entry so usage lines and
``Try ... --help`` hints name ``gymrat``, the command users actually type.
"""

from gymrat.cli.app import main

if __name__ == "__main__":
    main()
