"""The ``init`` feature: the interactive wizard and the file scaffold helper.

Consumers import from the submodules directly — ``gymrat_py.init.wizard`` and
``gymrat_py.init.scaffold`` — rather than through this package. Re-exporting the
``scaffold`` function here would shadow the same-named submodule as a package
attribute, breaking attribute-path patching of ``scaffold``'s own imports.
"""
