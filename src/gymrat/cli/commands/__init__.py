"""The command bodies ``gymrat.cli.app`` registers, one module per command family.

Import the submodules directly. Each one loads its engine lazily, so importing
the package that registers them stays cheap.
"""
