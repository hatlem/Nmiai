import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from executor import resolve_ref, resolve_refs


def test_resolve_ref_simple_id():
    results = {0: {"data": {"value": {"id": 42}}}}
    assert resolve_ref("$step_0.id", results) == 42


def test_resolve_ref_nested_field():
    results = {0: {"data": {"value": {"id": 42, "name": "Test"}}}}
    assert resolve_ref("$step_0.name", results) == "Test"


def test_resolve_ref_no_match():
    assert resolve_ref("plain_string", {}) == "plain_string"


def test_resolve_refs_in_dict():
    results = {0: {"data": {"value": {"id": 10}}}}
    body = {"customer": {"id": "$step_0.id"}, "name": "Test"}
    resolved = resolve_refs(body, results)
    assert resolved == {"customer": {"id": 10}, "name": "Test"}


def test_resolve_refs_in_path():
    results = {1: {"data": {"value": {"id": 99}}}}
    path = "/order/$step_1.id/:invoice"
    resolved = resolve_ref(path, results)
    assert resolved == "/order/99/:invoice"


def test_resolve_ref_array_indexing():
    results = {0: {"data": {"values": [{"id": 7, "description": "Cash"}, {"id": 8}]}}}
    assert resolve_ref("$step_0.values[0].id", results) == 7


def test_resolve_ref_deep_nested():
    results = {0: {"data": {"value": {"customer": {"id": 5}}}}}
    assert resolve_ref("$step_0.customer.id", results) == 5


def test_resolve_ref_non_string():
    assert resolve_ref(42, {}) == 42
    assert resolve_ref(True, {}) is True
    assert resolve_ref(None, {}) is None
