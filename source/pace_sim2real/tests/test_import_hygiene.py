# © 2026 Kushagra Srivastava
# Licensed under the Apache License 2.0

"""Layering rules: the core and the MuJoCo backend must work without Isaac Lab, the core without any simulator."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "pace_sim2real"
ISAAC_MODULES = ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic", "isaacsim", "omni",
                 "pxr", "carb")


def _module_imports(path: Path) -> set[str]:
    """Absolute names of all modules imported by ``path``, ignoring ``if TYPE_CHECKING:`` blocks."""
    package = ".".join(path.relative_to(PACKAGE_DIR.parent).with_suffix("").parts[:-1])
    imports = set()

    class Visitor(ast.NodeVisitor):
        def visit_If(self, node):
            if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                return
            self.generic_visit(node)

        def visit_Import(self, node):
            imports.update(alias.name for alias in node.names)

        def visit_ImportFrom(self, node):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - node.level + 1]
                imports.add(".".join(base + ([node.module] if node.module else [])))
            else:
                imports.add(node.module)

    Visitor().visit(ast.parse(path.read_text()))
    return imports


def _files(*relative: str) -> list[Path]:
    files = []
    for item in relative:
        path = PACKAGE_DIR / item
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return files


def _top_level(name: str) -> str:
    return name.split(".")[0]


@pytest.mark.parametrize("path", _files("core", "optim", "backends"), ids=lambda p: str(p.relative_to(PACKAGE_DIR)))
def test_no_isaac_imports(path):
    isaac = {name for name in _module_imports(path) if _top_level(name) in ISAAC_MODULES}
    assert not isaac, f"{path.name} imports Isaac modules {isaac}"


@pytest.mark.parametrize("path", _files("core"), ids=lambda p: str(p.relative_to(PACKAGE_DIR)))
def test_core_is_simulator_independent(path):
    forbidden = {name for name in _module_imports(path)
                 if _top_level(name) == "mujoco" or name.startswith(("pace_sim2real.backends", "pace_sim2real.tasks",
                                                                     "pace_sim2real.utils", "pace_sim2real.optim"))}
    assert not forbidden, f"{path.name} imports {forbidden}"


@pytest.mark.parametrize("path", _files("backends/base.py", "backends/actuator.py", "backends/isaac"),
                         ids=lambda p: str(p.relative_to(PACKAGE_DIR)))
def test_backends_do_not_import_other_simulators(path):
    assert not {name for name in _module_imports(path) if _top_level(name) == "mujoco" or ".mujoco" in name}


@pytest.mark.parametrize("path", _files("backends/mujoco"), ids=lambda p: str(p.relative_to(PACKAGE_DIR)))
def test_mujoco_backend_does_not_import_isaac_backend(path):
    assert not {name for name in _module_imports(path) if name.startswith("pace_sim2real.backends.isaac")}


@pytest.mark.parametrize("path", _files("."), ids=lambda p: str(p.relative_to(PACKAGE_DIR)))
def test_python_310_syntax(path):
    # Isaac Sim 4.5 ships Python 3.10
    ast.parse(path.read_text(), feature_version=(3, 10))


BLOCK_ISAAC = f"""
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {ISAAC_MODULES!r}:
            raise ImportError("blocked " + name)
sys.meta_path.insert(0, Block())
"""


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", BLOCK_ISAAC + code], cwd=PACKAGE_DIR.parent, capture_output=True, text=True)


def test_runtime_imports_without_isaac():
    result = _run("""
import pace_sim2real, pace_sim2real.core, pace_sim2real.backends, pace_sim2real.backends.isaac, pace_sim2real.optim
import pace_sim2real.utils
assert pace_sim2real.utils.project_root().is_dir()
assert hasattr(pace_sim2real.CMAESOptimizer, "update_simulator")
loaded = [m for m in sys.modules if m.split(".")[0] in {isaac!r}]
assert not loaded, loaded
for name in ("PaceCfg", "PaceSim2realEnvCfg"):
    try:
        getattr(pace_sim2real, name)
    except ImportError:
        pass
    else:
        raise AssertionError(name + " should need Isaac Lab")
try:
    pace_sim2real.utils.PaceDCMotorCfg
except ImportError:
    pass
else:
    raise AssertionError("PaceDCMotorCfg should need Isaac Lab")
print("ok")
""".replace("{isaac!r}", repr(ISAAC_MODULES)))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_runtime_mujoco_backend_without_isaac():
    pytest.importorskip("mujoco")
    result = _run("import pace_sim2real.backends.mujoco, pace_sim2real.backends.mujoco.robots; print('ok')")
    assert result.returncode == 0, result.stderr


def test_core_does_not_load_simulators():
    result = _run("import pace_sim2real.core; print(sorted(m for m in sys.modules if m.split('.')[0] == 'mujoco'))")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
