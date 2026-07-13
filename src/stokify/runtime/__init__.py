"""Hand-written exemplar of the transpiler's output for the linear stokify recipe.

`tasks.py`, `runner.py`, and `cli.py` are written by hand to look exactly like
what `hip-cargo transpile` would emit from `recipes/stokify.yml` (RFC §12
stage 2). They are plain, mechanical Python — no clever abstractions — so a
reviewer can compare the recipe (the would-be input) against these modules (the
would-be output) and see that the code generation is mechanical.
"""
