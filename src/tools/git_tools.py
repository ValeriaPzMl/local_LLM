import subprocess
from pathlib import Path


class GitToolError(Exception):
    pass


def run_git_command(
    workspace: Path,
    arguments: list[str],
    timeout: int = 20,
) -> str:
    workspace = workspace.resolve()

    if not (workspace / ".git").exists():
        raise GitToolError(
            "El workspace no es un repositorio Git."
        )

    command = [
        "git",
        "-C",
        str(workspace),
        *arguments,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )

    output = (
        result.stdout.strip()
        or result.stderr.strip()
    )

    if result.returncode != 0:
        raise GitToolError(
            output or "Git devolvió un error."
        )

    return output or "Sin cambios."


def git_status(
    workspace: Path,
) -> str:
    return run_git_command(
        workspace,
        [
            "status",
            "--short",
            "--branch",
        ],
    )


def git_diff(
    workspace: Path,
    staged: bool = False,
    relative_path: str | None = None,
) -> str:
    arguments = [
        "diff",
        "--no-ext-diff",
    ]

    if staged:
        arguments.append("--cached")

    if relative_path:
        arguments.extend(
            [
                "--",
                relative_path,
            ]
        )

    output = run_git_command(
        workspace,
        arguments,
    )

    if len(output) > 8_000:
        output = (
            output[:8_000]
            + "\n\n... diff truncado ..."
        )

    return output