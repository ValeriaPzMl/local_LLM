from pathlib import Path


IGNORED_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

BLOCKED_FILES = {
    ".env",
    "id_rsa",
    "id_ed25519",
}


class WorkspaceError(Exception):
    pass


def resolve_safe_path(
    workspace: Path,
    relative_path: str,
) -> Path:
    workspace = workspace.resolve()

    target = (
        workspace / relative_path
    ).resolve()

    try:
        target.relative_to(workspace)
    except ValueError as error:
        raise WorkspaceError(
            "No se permite acceder fuera del workspace."
        ) from error

    if target.name in BLOCKED_FILES:
        raise WorkspaceError(
            f"El archivo `{target.name}` está protegido."
        )

    return target


def list_files(
    workspace: Path,
    relative_path: str = ".",
    max_depth: int = 4,
    max_items: int = 300,
) -> str:
    target = resolve_safe_path(
        workspace,
        relative_path,
    )

    if not target.exists():
        raise WorkspaceError(
            f"No existe la ruta: {relative_path}"
        )

    if not target.is_dir():
        raise WorkspaceError(
            f"La ruta no es una carpeta: {relative_path}"
        )

    lines = []
    item_count = 0

    def walk(
        directory: Path,
        depth: int,
    ) -> None:
        nonlocal item_count

        if depth > max_depth:
            return

        for child in sorted(
            directory.iterdir(),
            key=lambda item: (
                not item.is_dir(),
                item.name.lower(),
            ),
        ):
            if item_count >= max_items:
                return

            if child.name in IGNORED_NAMES:
                continue

            relative = child.relative_to(
                workspace
            )

            prefix = "  " * depth

            if child.is_dir():
                lines.append(
                    f"{prefix}📁 {relative}/"
                )
                item_count += 1

                walk(
                    child,
                    depth + 1,
                )
            else:
                lines.append(
                    f"{prefix}📄 {relative}"
                )
                item_count += 1

    walk(target, 0)

    if item_count >= max_items:
        lines.append(
            "... resultado truncado ..."
        )

    return (
        "\n".join(lines)
        if lines
        else "La carpeta está vacía."
    )


def read_file(
    workspace: Path,
    relative_path: str,
    max_characters: int = 12_000,
) -> str:
    target = resolve_safe_path(
        workspace,
        relative_path,
    )

    if not target.exists():
        raise WorkspaceError(
            f"No existe el archivo: {relative_path}"
        )

    if not target.is_file():
        raise WorkspaceError(
            f"La ruta no es un archivo: {relative_path}"
        )

    if target.stat().st_size > 2_000_000:
        raise WorkspaceError(
            "El archivo es demasiado grande para leerlo."
        )

    content = target.read_text(
        encoding="utf-8",
        errors="replace",
    )

    numbered_content = "\n".join(
        f"{line_number:4}: {line}"
        for line_number, line in enumerate(
            content.splitlines(),
            start=1,
        )
    )

    if len(numbered_content) > max_characters:
        numbered_content = (
            numbered_content[:max_characters]
            + "\n\n... contenido truncado ..."
        )

    return numbered_content

def search_code(
    workspace: Path,
    query: str,
    relative_path: str = ".",
    max_results: int = 50,
) -> str:
    if not query.strip():
        raise WorkspaceError(
            "La búsqueda no puede estar vacía."
        )

    target = resolve_safe_path(
        workspace,
        relative_path,
    )

    results = []
    query_lower = query.lower()

    paths = (
        [target]
        if target.is_file()
        else target.rglob("*")
    )

    for path in paths:
        if len(results) >= max_results:
            break

        if not path.is_file():
            continue

        if any(
            ignored in path.parts
            for ignored in IGNORED_NAMES
        ):
            continue

        if path.name in BLOCKED_FILES:
            continue

        if path.stat().st_size > 1_000_000:
            continue

        try:
            content = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue

        for line_number, line in enumerate(
            content.splitlines(),
            start=1,
        ):
            if query_lower in line.lower():
                relative = path.relative_to(
                    workspace
                )

                results.append(
                    f"{relative}:{line_number}: "
                    f"{line.strip()}"
                )

                if len(results) >= max_results:
                    break

    if not results:
        return (
            f"No se encontraron coincidencias "
            f"para `{query}`."
        )

    return "\n".join(results)
def write_file(
    workspace: Path,
    relative_path: str,
    content: str,
) -> str:
    target = resolve_safe_path(
        workspace,
        relative_path,
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.write_text(
        content,
        encoding="utf-8",
    )

    return (
        f"Archivo actualizado correctamente: "
        f"{relative_path}"
    )