"""Single source of truth for the package version.

Imported by ``ecp/__init__.py`` (public ``__version__``), by ``ecp/store.py``
(stamped into every serialized Evidence record) and by ``pyproject.toml``
(``[tool.setuptools.dynamic]``). Bump it here and nowhere else: a proof object
whose version metadata disagrees with the code that produced it is not audit
evidence, it is a puzzle.
"""

__version__ = "0.4.1"
