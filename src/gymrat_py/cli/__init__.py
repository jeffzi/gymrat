"""Command-line infrastructure: parsing, exit routing, error rendering, progress.

The submodules here carry no dependency on the heavy statistics stack or the
command bodies, so importing them stays cheap. The command bodies that wire
these pieces into a runnable CLI live alongside them.
"""
