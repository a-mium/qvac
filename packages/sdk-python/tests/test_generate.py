"""Tests for scripts/generate.py: the Python client's typed-surface generator.

Mirrors the SDK's own contract-export test rigor (packages/sdk/test/unit/
contract-export.test.ts): manifest/schema coverage, deterministic
regeneration, and that the committed output matches a fresh build.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PACKAGE_ROOT / "scripts"

spec = importlib.util.spec_from_file_location("generate", SCRIPTS_DIR / "generate.py")
generate = importlib.util.module_from_spec(spec)
sys.modules["generate"] = generate
spec.loader.exec_module(generate)


@pytest.fixture(scope="module")
def manifest_methods() -> list[dict]:
    return generate.load_manifest_methods()


@pytest.fixture(scope="module")
def fresh_build() -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        output_root = Path(tmp) / "_generated"
        generate.build(output_root)
        yield output_root


def test_committed_output_matches_a_fresh_build(fresh_build: Path) -> None:
    assert generate.compare_dirs(fresh_build, generate.GENERATED_DIR)


def test_build_is_deterministic_across_runs(fresh_build: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        second = Path(tmp) / "_generated"
        generate.build(second)
        assert generate.compare_dirs(fresh_build, second)


def test_every_manifest_method_resolves_to_a_request_and_response_class(
    manifest_methods: list[dict],
) -> None:
    resolved = generate.resolve_titles(generate.MODELS_DIR, manifest_methods)
    for method in manifest_methods:
        name = method["name"]
        request_title = f"{generate.pascal_case(name)}Request"
        response_title = f"{generate.pascal_case(name)}Response"
        assert request_title in resolved, f"{name} has no resolvable request class"
        assert response_title in resolved, f"{name} has no resolvable response class"


def test_index_reexports_every_resolved_title(manifest_methods: list[dict]) -> None:
    resolved = generate.resolve_titles(generate.MODELS_DIR, manifest_methods)
    rendered = generate.render_index(resolved)
    for title in resolved:
        assert title in rendered, f"{title} missing from rendered index"


def test_methods_module_has_one_function_per_manifest_entry_with_matching_shape(
    manifest_methods: list[dict],
) -> None:
    import ast

    rendered = generate.render_methods_module(manifest_methods)
    tree = ast.parse(rendered)
    functions_by_name = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    for method in manifest_methods:
        func_name = generate.snake_case(method["name"])
        assert func_name in functions_by_name, f"missing stub for {method['name']}"

        params = [arg.arg for arg in functions_by_name[func_name].args.args]
        shape = generate.CALL_SHAPE_ANNOTATION[method["callShape"]]
        if shape == "duplex":
            assert params == [
                "transport",
                "params",
                "up",
            ], f"{method['name']} is duplex, expected (transport, params, up), got {params}"
        else:
            assert params == [
                "transport",
                "params",
            ], f"{method['name']} is {shape}, expected (transport, params), got {params}"


def test_manifest_method_count_matches_sdk_contract() -> None:
    manifest_methods = generate.load_manifest_methods()
    manifest = json.loads(generate.MANIFEST_PATH.read_text())
    assert len(manifest_methods) == len(manifest["methods"])
    assert len(manifest_methods) > 0
