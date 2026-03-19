import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from templates import TEMPLATES


def test_all_templates_have_required_fields():
    for task_type, template in TEMPLATES.items():
        assert "steps" in template, f"{task_type} missing steps"
        assert "relevant_schemas" in template, f"{task_type} missing relevant_schemas"
        assert "description" in template, f"{task_type} missing description"
        if task_type != "unknown":
            assert len(template["steps"]) > 0, f"{task_type} has empty steps"


def test_all_methods_are_valid():
    valid_methods = {"GET", "POST", "PUT", "DELETE"}
    for task_type, template in TEMPLATES.items():
        for step in template["steps"]:
            assert step["method"] in valid_methods, f"{task_type}: invalid method {step['method']}"


def test_minimum_task_types():
    """We need at least 15 templates to cover common task types."""
    assert len(TEMPLATES) >= 15, f"Only {len(TEMPLATES)} templates, need at least 15"


def test_all_paths_start_with_slash():
    for task_type, template in TEMPLATES.items():
        for i, step in enumerate(template["steps"]):
            path = step["path"]
            assert path.startswith("/") or path.startswith("{{"), \
                f"{task_type} step {i}: path '{path}' doesn't start with /"
