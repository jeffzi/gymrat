"""The ``init`` feature: the file scaffold helper.

Consumers import from ``gymrat_py.init.scaffold`` directly rather than through
this package. Re-exporting the ``scaffold`` function here would shadow the
same-named submodule as a package attribute, breaking attribute-path patching of
``scaffold``'s own imports.
"""
