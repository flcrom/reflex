"""Tests for the reflex CLI command tree."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time

import click
import click.testing
import psutil
import pytest
from pytest_mock import MockerFixture

from reflex import reflex
from reflex.testing import DEFAULT_TIMEOUT

_CLI_STARTUP_DENIED_MODULES = frozenset({
    "PIL",
    "alembic",
    "fastapi",
    "granian",
    "httpx",
    "numpy",
    "pandas",
    "plotly",
    "redis",
    "reflex.app",
    "reflex.compiler",
    "reflex.custom_components.custom_components",
    "reflex.model",
    "reflex.state",
    "reflex.utils.frontend_skeleton",
    "reflex.utils.prerequisites",
    "reflex_cli.v2.deploy",
    "reflex_cli.v2.deployments",
    "sqlalchemy",
    "sqlmodel",
    "starlette",
    "uvicorn",
})
_COMPONENT_HELP_DENIED_MODULES = _CLI_STARTUP_DENIED_MODULES - {
    "reflex.custom_components.custom_components"
}


def _run_cli_probe(probe: str) -> dict[str, object]:
    """Run a CLI import probe in a fresh interpreter.

    Args:
        probe: The Python source to execute.

    Returns:
        The JSON object written by the probe.
    """
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
        env={
            **os.environ,
            "REFLEX_CHECK_LATEST_VERSION": "false",
            "REFLEX_TELEMETRY_ENABLED": "false",
        },
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize(
    ("argv", "denied_modules"),
    [
        (["--help"], _CLI_STARTUP_DENIED_MODULES),
        (["--version"], _CLI_STARTUP_DENIED_MODULES),
        (["run", "--help"], _CLI_STARTUP_DENIED_MODULES),
        (["component", "--help"], _COMPONENT_HELP_DENIED_MODULES),
        (
            ["deploy", "--help"],
            _CLI_STARTUP_DENIED_MODULES - {"reflex_cli.v2.deploy"},
        ),
        (
            ["cloud", "--help"],
            _CLI_STARTUP_DENIED_MODULES - {"reflex_cli.v2.deployments"},
        ),
    ],
    ids=[
        "help",
        "version",
        "run-help",
        "component-help",
        "deploy-help",
        "cloud-help",
    ],
)
def test_cli_startup_does_not_import_runtime_modules(
    argv: list[str], denied_modules: frozenset[str]
):
    """Keep informational CLI paths independent of app and optional runtimes.

    Args:
        argv: The informational command-line arguments to invoke.
        denied_modules: Modules that the command must not import.
    """
    probe = f"""
import json
import sys

from click.testing import CliRunner
from reflex.reflex import cli

result = CliRunner().invoke(cli, {argv!r})
denied = {denied_modules!r}
loaded = sorted(
    module
    for module in denied
    if module in sys.modules
    or any(name.startswith(module + ".") for name in sys.modules)
)
print(json.dumps({{"exit_code": result.exit_code, "loaded": loaded}}))
"""
    outcome = _run_cli_probe(probe)

    assert outcome["exit_code"] == 0
    assert outcome["loaded"] == []


def test_backend_launcher_does_not_import_compiler_or_state() -> None:
    """The backend supervisor must not load the worker's compiler and state."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from reflex import reflex
from reflex.istate.manager import reset_disk_state_manager
from reflex.utils import build, exec, telemetry

unexpected = {"reflex.state", "reflex.compiler.utils", "sqlalchemy"} & sys.modules.keys()
assert not unexpected, unexpected
""",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_compile_app_worker_flushes_telemetry(mocker):
    """Flush the completed compile span before an isolated worker exits."""
    app_task = mocker.Mock(return_value=True)
    flush = mocker.patch("reflex_base.otel.flush")

    assert reflex._compile_app_worker(app_task, (True,), {"trigger": "initial"})

    app_task.assert_called_once_with(True, trigger="initial")
    flush.assert_called_once_with()


def test_compile_app_worker_flushes_telemetry_on_failure(mocker):
    """Flush telemetry even when the worker's compile task raises."""
    app_task = mocker.Mock(side_effect=RuntimeError("compile failed"))
    flush = mocker.patch("reflex_base.otel.flush")

    with pytest.raises(RuntimeError, match="compile failed"):
        reflex._compile_app_worker(app_task, (), {})

    flush.assert_called_once_with()


def test_cloud_commands_registered():
    """The hosting CLI commands import, resolve, and dispatch only on demand."""
    probe = """
import json
import sys

import click
from click.testing import CliRunner
from reflex import reflex

deploy_command = reflex.cli.commands["deploy"]
cloud_command = reflex.cli.commands["cloud"]
imported_before = {
    "deploy": "reflex_cli.v2.deploy" in sys.modules,
    "cloud": "reflex_cli.v2.deployments" in sys.modules,
}
unresolved_before = {
    "deploy": deploy_command._resolved_command is None,
    "cloud": cloud_command._resolved_command is None,
}

runner = CliRunner()
deploy_result = runner.invoke(reflex.cli, ["deploy", "--help"])
cloud_result = runner.invoke(reflex.cli, ["cloud", "--help"])

from reflex_cli.v2.deploy import deploy

print(json.dumps({
    "cloud_is_click": isinstance(cloud_command._resolved_command, click.Command),
    "cloud_help_matches": cloud_command.get_short_help_str()
    == cloud_command._resolved_command.get_short_help_str(),
    "cloud_result": cloud_result.exit_code,
    "deploy_help_matches": deploy_command.help == deploy.help,
    "deploy_is_real": deploy_command._resolved_command is deploy,
    "deploy_result": deploy_result.exit_code,
    "imported_before": imported_before,
    "lazy_commands": [
        isinstance(deploy_command, reflex._LazyCommand),
        isinstance(cloud_command, reflex._LazyCommand),
    ],
    "unresolved_before": unresolved_before,
}))
"""
    outcome = _run_cli_probe(probe)

    assert outcome == {
        "cloud_help_matches": True,
        "cloud_is_click": True,
        "cloud_result": 0,
        "deploy_help_matches": True,
        "deploy_is_real": True,
        "deploy_result": 0,
        "imported_before": {"cloud": False, "deploy": False},
        "lazy_commands": [True, True],
        "unresolved_before": {"cloud": True, "deploy": True},
    }


def test_component_command_registered_lazily():
    """The component command preserves its help while loading on demand."""
    command = reflex.cli.commands["component"]

    assert isinstance(command, reflex._LazyCommand)
    result = click.testing.CliRunner().invoke(reflex.cli, ["component", "--help"])

    assert result.exit_code == 0
    resolved_command = command._resolved_command
    assert resolved_command is not None
    assert command.help == resolved_command.help
    assert "CLI for creating custom components." in result.output


def test_lazy_command_delegates_click_introspection():
    """Click integrations inspecting a registered command see its real metadata."""
    command = reflex._LazyCommand(
        "component",
        "reflex.custom_components.custom_components:custom_components_cli",
        help="CLI for creating custom components.",
    )
    context = click.Context(command, info_name="component")

    help_text = command.get_help(context)
    params = command.get_params(context)

    assert "Commands:" in help_text
    assert "build" in help_text
    assert command._resolved_command is not None
    assert params == command._resolved_command.get_params(context)


def test_lazy_command_delegates_direct_invoke(monkeypatch: pytest.MonkeyPatch):
    """Calling Click's public invoke method executes the resolved callback."""
    called = False

    @click.command()
    def implementation():
        nonlocal called
        called = True

    monkeypatch.setattr(
        reflex,
        "import_module",
        lambda name: type("Commands", (), {"implementation": implementation}),
    )
    command = reflex._LazyCommand(
        "implementation",
        "commands:implementation",
        help="Test command.",
    )

    command.invoke(click.Context(command))

    assert called
    assert command._resolved_command is implementation


def test_lazy_command_delegates_direct_metadata(monkeypatch: pytest.MonkeyPatch):
    """Direct reads of Click's command metadata resolve to the implementation."""

    @click.group()
    @click.option("--value")
    def implementation(value: str | None):
        pass

    monkeypatch.setattr(
        reflex,
        "import_module",
        lambda name: type("Commands", (), {"implementation": implementation}),
    )
    command = reflex._LazyCommand(
        "implementation",
        "commands:implementation",
        help="Test command.",
    )

    assert command.no_args_is_help is implementation.no_args_is_help
    assert command.params == implementation.params
    assert command.callback is implementation.callback
    assert command._resolved_command is implementation


def test_lazy_hosting_command_reports_missing_package(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """An unavailable lazy hosting command keeps the install guidance."""

    def missing_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr(reflex, "import_module", missing_import)
    command = reflex._LazyCommand(
        "deploy",
        "reflex_cli.v2.deploy:deploy",
        help="Deploy the app to the Reflex hosting service.",
        optional=True,
    )

    result = click.testing.CliRunner().invoke(
        command, ["--app-name", "demo", "--no-interactive"]
    )

    assert result.exit_code == 1
    assert "pip install reflex-hosting-cli" in caplog.text
    assert "No such option" not in result.output


def test_lazy_hosting_command_keeps_missing_package_help(
    monkeypatch: pytest.MonkeyPatch,
):
    """An unavailable hosting package retains its top-level help description."""
    monkeypatch.setattr(reflex, "find_spec", lambda name: None, raising=False)

    command = reflex._LazyCommand(
        "deploy",
        "reflex_cli.v2.deploy:deploy",
        help="Deploy the app to the Reflex hosting service.",
        optional=True,
    )

    assert command.help == "Requires the reflex-hosting-cli package."


def test_lazy_hosting_command_reports_incompatible_package(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """An outdated hosting module missing the command keeps the install guidance."""
    monkeypatch.setattr(reflex, "import_module", lambda name: object())
    command = reflex._LazyCommand(
        "deploy",
        "reflex_cli.v2.deploy:deploy",
        help="Deploy the app to the Reflex hosting service.",
        optional=True,
    )

    result = click.testing.CliRunner().invoke(command, ["--app-name", "demo"])

    assert result.exit_code == 1
    assert "pip install reflex-hosting-cli" in caplog.text
    assert not isinstance(result.exception, AttributeError)


def test_missing_command_reports_the_package(caplog: pytest.LogCaptureFixture):
    """Without the hosting CLI, the command says which package to install."""
    result = click.testing.CliRunner().invoke(reflex._missing_command("deploy"))

    assert result.exit_code == 1
    assert "is not installed" in caplog.text
    assert "pip install reflex-hosting-cli" in caplog.text


def test_missing_command_tolerates_flags(caplog: pytest.LogCaptureFixture):
    """The stand-in reports the missing package instead of a usage error.

    The real command's flags must not produce "No such option", which would hide
    the actual cause from the user.
    """
    result = click.testing.CliRunner().invoke(
        reflex._missing_command("deploy"), ["--app-name", "demo", "--no-interactive"]
    )

    assert result.exit_code == 1
    assert "pip install reflex-hosting-cli" in caplog.text
    assert "No such option" not in result.output


def test_init_records_version_check_after_frontend_setup(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """A new project's version-check timestamp survives web initialization."""
    events: list[str] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("reflex.utils.exec.output_system_info", lambda: None)
    monkeypatch.setattr(
        "reflex.utils.prerequisites.validate_app_name", lambda name: name
    )
    monkeypatch.setattr(
        "reflex.utils.prerequisites.initialize_reflex_user_directory", lambda: None
    )
    monkeypatch.setattr(
        "reflex.utils.prerequisites.ensure_reflex_installation_id", lambda: None
    )
    monkeypatch.setattr(
        "reflex.utils.prerequisites.initialize_frontend_dependencies",
        lambda: events.append("frontend"),
    )
    monkeypatch.setattr(
        "reflex.utils.prerequisites.check_latest_package_version",
        lambda package: events.append("version"),
    )
    monkeypatch.setattr(
        "reflex.utils.templates.initialize_app", lambda app_name, template: "blank"
    )
    monkeypatch.setattr(
        "reflex.utils.frontend_skeleton.initialize_gitignore", lambda: None
    )
    monkeypatch.setattr(
        "reflex.utils.frontend_skeleton.initialize_requirements_txt", lambda: False
    )

    reflex._init("demo")

    assert events == ["frontend", "version"]


posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX process groups and signals"
)

# Stubs the app and frontend setup so `_run_dev` launches stand-in processes.
# `PIDS` receives the pid of each launched frontend.
_DRIVER_TEMPLATE = """import signal, subprocess, sys, threading, time, types
from pathlib import Path
from reflex.utils import build, exec as exec_mod, processes, telemetry

PIDS = {pids!r}
telemetry.send = lambda *args, **kwargs: None
build.setup_frontend = lambda *args, **kwargs: None
exec_mod.get_web_dir = Path.cwd
exec_mod.get_package_json_and_hash = lambda *args: ({{}}, "unchanged")
exec_mod.frontend_env = lambda *args: {{}}
exec_mod.path_ops.get_node_bin_path = lambda: None
original_new_process = processes.new_process

def record_process(*args, **kwargs):
    p = original_new_process(*args, **kwargs)
    Path(PIDS).write_text(str(p.pid))
    return p

processes.new_process = record_process

def launch_frontend(code):
    exec_mod.run_frontend = lambda *args: exec_mod.run_process_and_launch_url(
        [sys.executable, "-c", code], True
    )

{body}

import reflex.reflex as rx
from reflex_base import constants
rx._compile_app = lambda: None
rx.get_config = lambda: types.SimpleNamespace(
    _set_persistent=lambda **kwargs: None,
    loglevel=types.SimpleNamespace(subprocess_level=lambda: None),
)
rx._run_dev(constants.RunningMode.{mode}, 3000, {backend_port}, "127.0.0.1")
"""

# A frontend that starts a worker and stops it on SIGTERM, like bun and node.
_FRONTEND_TREE = (
    "import signal,subprocess,sys,time\n"
    "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
    "signal.signal(signal.SIGTERM, lambda *a: (p.terminate(), sys.exit(0)))\n"
    "print(p.pid,flush=True);time.sleep(60)\n"
)


def _start_driver(tmp_path, body: str, mode: str = "FULLSTACK") -> subprocess.Popen:
    """Run `_run_dev` in a new session with no TTY.

    Args:
        tmp_path: The directory for the driver script and the pid file.
        body: Driver code that stubs the frontend and the backend.
        mode: The RunningMode member name.

    Returns:
        The launcher process.
    """
    driver = tmp_path / "driver.py"
    driver.write_text(
        _DRIVER_TEMPLATE.format(
            pids=str(tmp_path / "frontend.pid"),
            body=body,
            mode=mode,
            backend_port="None" if mode == "FRONTEND_ONLY" else "8000",
        )
    )
    return subprocess.Popen(
        [sys.executable, str(driver)],
        cwd=tmp_path,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_exit(launcher: subprocess.Popen) -> tuple[int, str]:
    """Wait for the launcher to exit.

    Args:
        launcher: The launcher process.

    Returns:
        The return code and the stderr output.
    """
    try:
        _, stderr = launcher.communicate(timeout=DEFAULT_TIMEOUT)
    except subprocess.TimeoutExpired:
        pytest.fail("the run did not exit")
    return launcher.returncode, stderr


def _read_pid(tmp_path) -> int:
    """Read the pid of the launched frontend.

    Args:
        tmp_path: The directory of the pid file.

    Returns:
        The frontend pid.
    """
    pid_file = tmp_path / "frontend.pid"
    assert pid_file.exists(), "frontend did not start"
    return int(pid_file.read_text())


def _wait_for_frontend_tree(launcher: subprocess.Popen, tmp_path) -> tuple[int, int]:
    """Wait for the frontend and its worker to start.

    Args:
        launcher: The launcher process.
        tmp_path: The directory of the pid file.

    Returns:
        The frontend pid and the worker pid.
    """
    pid_file = tmp_path / "frontend.pid"
    deadline = time.monotonic() + DEFAULT_TIMEOUT
    while time.monotonic() < deadline:
        if launcher.poll() is not None:
            pytest.fail(f"launcher exited early: {launcher.returncode}")
        if pid_file.exists() and (text := pid_file.read_text()):
            with contextlib.suppress(psutil.NoSuchProcess):
                if children := psutil.Process(int(text)).children():
                    return int(text), children[0].pid
        time.sleep(0.01)
    pytest.fail("frontend worker did not start")


def _assert_stopped(pid: int):
    """Assert that a process exits.

    Args:
        pid: The process id.
    """
    deadline = time.monotonic() + DEFAULT_TIMEOUT
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() in (
                psutil.STATUS_ZOMBIE,
                psutil.STATUS_DEAD,
            ):
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(0.05)
    pytest.fail(f"frontend process {pid} survived")


@contextlib.contextmanager
def _reaped(launcher: subprocess.Popen, pids: list[int]):
    """Kill the launcher group and the listed processes on exit.

    Args:
        launcher: The launcher process.
        pids: The processes to kill; filled in by the test.

    Yields:
        None.
    """
    try:
        yield
    finally:
        if launcher.poll() is None:
            os.killpg(launcher.pid, signal.SIGKILL)
            launcher.communicate(timeout=DEFAULT_TIMEOUT)
        for pid in pids:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


@posix_only
def test_frontend_preflight_failure_stops_frontend(tmp_path):
    """A failing frontend task exits the run and stops its process."""
    launcher = _start_driver(
        tmp_path,
        """
def frontend(*args):
    p = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    exec_mod.frontend_process = p
    Path(PIDS).write_text(str(p.pid))
    raise SystemExit(3)

exec_mod.run_frontend = frontend
exec_mod.run_backend = lambda *args: time.sleep(60)
""",
    )
    pids = []
    with _reaped(launcher, pids):
        returncode, stderr = _wait_exit(launcher)
        pids.append(_read_pid(tmp_path))
        assert returncode == 3, stderr[-1000:]
        _assert_stopped(pids[0])


@posix_only
def test_backend_return_stops_frontend(tmp_path):
    """The run exits after the backend returns."""
    launcher = _start_driver(
        tmp_path,
        """
launch_frontend("import time;time.sleep(60)")

def backend(*args):
    deadline = time.monotonic() + 5
    while not Path(PIDS).exists() and time.monotonic() < deadline:
        time.sleep(0.01)

exec_mod.run_backend = backend
""",
    )
    pids = []
    with _reaped(launcher, pids):
        returncode, stderr = _wait_exit(launcher)
        pids.append(_read_pid(tmp_path))
        assert returncode == 0, stderr[-1000:]
        _assert_stopped(pids[0])


@posix_only
def test_frontend_launched_after_stop_is_stopped(tmp_path):
    """A frontend that starts after the run stops the frontend is also stopped."""
    launcher = _start_driver(
        tmp_path,
        """
stopped = threading.Event()
original_stop_frontend = exec_mod.stop_frontend

def stop_frontend():
    original_stop_frontend()
    stopped.set()

exec_mod.stop_frontend = stop_frontend

def late_process(*args, **kwargs):
    assert stopped.wait(5)
    return record_process(*args, **kwargs)

processes.new_process = late_process
launch_frontend("import time;time.sleep(60)")
exec_mod.run_backend = lambda *args: None
""",
    )
    pids = []
    with _reaped(launcher, pids):
        returncode, stderr = _wait_exit(launcher)
        pids.append(_read_pid(tmp_path))
        assert returncode == 0, stderr[-1000:]
        _assert_stopped(pids[0])


@posix_only
@pytest.mark.parametrize("mode", ["FRONTEND_ONLY", "FULLSTACK"])
@pytest.mark.parametrize(
    ("sig", "returncodes"),
    [
        ("SIGTERM", (0,)),
        ("SIGINT", (0, -signal.SIGINT)),
        ("SIGKILL", (-signal.SIGKILL,)),
    ]
    if sys.platform != "win32"
    else [],
)
def test_signal_stops_frontend_tree(tmp_path, mode, sig, returncodes):
    """A signal to a run with no TTY stops the frontend and its worker."""
    launcher = _start_driver(
        tmp_path,
        f"""
launch_frontend({_FRONTEND_TREE!r})

def backend(*args):
    def stop(sig, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while True:
        time.sleep(1)

exec_mod.run_backend = backend
""",
        mode,
    )
    pids = []
    with _reaped(launcher, pids):
        pids.extend(_wait_for_frontend_tree(launcher, tmp_path))
        if sig == "SIGKILL":
            # A hard kill of the CLI job group must reach the frontend too.
            assert os.getpgid(pids[0]) == launcher.pid
            os.killpg(launcher.pid, signal.SIGKILL)
        else:
            os.kill(launcher.pid, getattr(signal, sig))
        returncode, stderr = _wait_exit(launcher)
        assert returncode in returncodes, stderr[-1000:]
        for pid in pids:
            _assert_stopped(pid)


@pytest.mark.parametrize(
    ("argv", "supervised", "expected"),
    [(["--json"], False, True), (["--json"], True, False), ([], False, False)],
)
def test_run_supervises_output_only_in_json_mode(
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    supervised: bool,
    expected: bool,
):
    """``reflex run --json`` runs itself again below an output supervisor.

    Args:
        mocker: The pytest-mock fixture.
        monkeypatch: The pytest monkeypatch fixture.
        argv: Extra ``reflex run`` arguments.
        supervised: Whether the process already runs under the supervisor.
        expected: Whether the supervisor is expected to start.
    """
    from reflex_base.environment import environment
    from reflex_base.utils import log

    # Registered so teardown restores the variables the CLI callbacks set.
    monkeypatch.setenv(log._MANAGED_ENV_VAR, "true")
    monkeypatch.setenv(environment.REFLEX_LOG_JSON.name, "false")
    monkeypatch.setenv(log._SUPERVISED_ENV_VAR, "1234" if supervised else "")
    monkeypatch.setattr(sys, "argv", ["reflex", "run", *argv])
    supervise = mocker.patch.object(log, "supervise_output", return_value=7)
    run = mocker.patch.object(reflex, "_run")
    mocker.patch("reflex.utils.prerequisites.check_running_mode")

    try:
        result = click.testing.CliRunner().invoke(reflex.cli, ["run", *argv])
    finally:
        log._reset()

    if expected:
        assert result.exit_code == 7
        supervise.assert_called_once_with([
            sys.executable,
            "-m",
            "reflex",
            "run",
            *argv,
        ])
        run.assert_not_called()
    else:
        assert result.exit_code == 0, result.output
        supervise.assert_not_called()
        run.assert_called_once()


@pytest.mark.parametrize(
    "args", [["init"], ["migrate"], ["makemigrations"], ["status"]]
)
def test_db_commands_without_db_extra_point_to_install(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, args: list[str]
):
    """Without the db extra, db commands print the install hint instead of a traceback."""
    monkeypatch.setattr(reflex, "find_spec", lambda name: None)

    result = click.testing.CliRunner().invoke(reflex.db_cli, args)

    assert result.exit_code == 1
    assert "pip install reflex[db]" in caplog.text
    assert not isinstance(result.exception, ImportError)


@pytest.mark.parametrize("missing", reflex._DB_PACKAGES)
def test_db_commands_with_partial_db_install_point_to_install(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, missing: str
):
    """A partial install missing any one db package still gets the install hint."""
    real_find_spec = reflex.find_spec
    monkeypatch.setattr(
        reflex,
        "find_spec",
        lambda name: None if name == missing else real_find_spec(name),
    )

    result = click.testing.CliRunner().invoke(reflex.db_cli, ["init"])

    assert result.exit_code == 1
    assert "pip install reflex[db]" in caplog.text
    assert not isinstance(result.exception, ImportError)
