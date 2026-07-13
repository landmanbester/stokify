"""The restricted-grammar recipe must parse to the expected linear DAG.

This is the would-be *input* to the transpiler; the hand-written ``runtime/``
modules are the would-be *output*. The same parser the monitoring stack uses
(``parse_recipe``) recovers the DAG.
"""

from pathlib import Path

from hip_cargo.monitoring.recipe_parser import parse_recipe

RECIPE = Path(__file__).resolve().parents[1] / "src" / "stokify" / "recipes" / "stokify.yml"


def test_recipe_steps_and_edges():
    dag = parse_recipe(RECIPE, resolve_cabs=False)
    assert dag.step_names() == ["init", "process", "image"]
    assert dag.edges == [("init", "process"), ("process", "image")]


def test_recipe_uses_only_restricted_bindings():
    """Every binding is a plain =recipe.x reference — no =IF/=IFSET/arithmetic."""
    dag = parse_recipe(RECIPE, resolve_cabs=False)
    for step in dag.steps:
        for param in step.params:
            if param.is_binding:
                assert param.binding_expr.startswith("recipe."), param.binding_expr


def test_recipe_resolves_cabs():
    dag = parse_recipe(RECIPE, resolve_cabs=True)
    assert set(dag.cab_schemas) == {"init", "process", "image"}
    assert dag.cab_schemas["init"]["command"] == "stokify.core.init.init"
