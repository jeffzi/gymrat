"""Local-only parity harness: diff the pinned oracle CLI against the port.

This package is gitignored and never referenced from committed files. It builds
the reference CLI from a pinned checkout, runs it against deterministic fixtures,
and diffs the parsed JSON output. Until the port ships its own CLI, only
self-diff mode (reference binary on both sides) is exercisable end to end.
"""
