"""The assembled ``gymrat`` command-line program and its console-script entry.

Building the app wires the ``compare`` and ``measure`` subcommands, the root
``--version`` and ``--debug`` options, and the example epilogues, all without
importing the comparison or measurement engines — those load lazily inside each
command so ``gymrat --help`` stays as cheap as importing the package.
"""

from __future__ import annotations

import importlib.metadata
from typing import Annotated

import typer

from gymrat_py.cli.compare_cmd import compare
from gymrat_py.cli.measure_cmd import measure
from gymrat_py.cli.shared import BUGS_URL, DebugOption, set_debug_mode

_DOCS_URL = "https://github.com/jeffzi/gymrat#readme"

_ROOT_EPILOGUE = f"""
Examples:

  • gymrat compare main my-branch --bench "npm run bench"

  • gymrat compare old=main new=perf/decode --bench "npm run bench" --fail-on regressed

  • gymrat measure --bench "npm run bench"

Docs: {_DOCS_URL}
Bugs: {BUGS_URL}
"""

_COMPARE_EPILOGUE = """
Examples:

  • gymrat compare main perf/faster-decode --bench "npm run bench"

  • gymrat compare main perf/simd perf/lookup-table --bench "npm run bench"

  • gymrat compare old=main new=perf/faster-decode --bench "npm run bench"

  • gymrat compare main my-branch --bench "npm run bench" --fail-on geomean:2 --format json
"""

_MEASURE_EPILOGUE = """
Examples:

  • gymrat measure --bench "npm run bench"

  • gymrat measure release=v2.0.0 --bench "npm run bench" --adapter mitata

  • gymrat measure main --bench "npm run bench" --record
"""


def _version_callback(*, value: bool) -> None:
    """Print the installed package version and exit when ``--version`` is given."""
    if value:
        typer.echo(importlib.metadata.version("gymrat-py"))
        raise typer.Exit


_VersionOption = Annotated[
    bool,
    typer.Option(
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="show the version and exit",
    ),
]


app = typer.Typer(
    name="gymrat",
    help="Performance comparison tool for benchmarks",
    epilog=_ROOT_EPILOGUE,
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


@app.callback()
def _root(*, debug: DebugOption = False, version: _VersionOption = False) -> None:
    """Route the shared ``--debug`` flag, tolerated before or after the subcommand."""
    _ = version  # consumed eagerly by its callback; declared so --version is a root option
    set_debug_mode(debug)


app.command("compare", epilog=_COMPARE_EPILOGUE)(compare)
app.command("measure", epilog=_MEASURE_EPILOGUE)(measure)


def main() -> None:
    """Console-script entry: run the assembled app."""
    app()


if __name__ == "__main__":
    main()
