import shlex
import subprocess
from pathlib import Path


ALLOWED_COMMANDS = {
    "pytest",
    "python",
    "python3",
}


class TestToolError(Exception):
    pass


def run_tests(
    workspace: Path,
    command: str = "pytest",
    timeout: int = 120,
) -> str:
    workspace = workspace.resolve()

    parts = shlex.split(command)

    if not parts:
        raise TestToolError(
            "El comando está vacío."
        )

    executable = parts[0]

    if executable not in ALLOWED_COMMANDS:
        raise TestToolError(
            f"Comando no permitido: {executable}"
        )

    try:
        result = subprocess.run(
            parts,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise TestToolError(
            "La ejecución excedió el tiempo permitido."
        ) from error

    output = (
        result.stdout.strip()
        + "\n"
        + result.stderr.strip()
    ).strip()

    if not output:
        output = "El comando terminó sin salida."

    if len(output) > 15_000:
        output = (
            output[:15_000]
            + "\n\n... salida truncada ..."
        )

    return (
        f"Exit code: {result.returncode}\n\n"
        f"{output}"
    )