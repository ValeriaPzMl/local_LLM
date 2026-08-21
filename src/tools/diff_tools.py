import difflib


def create_unified_diff(
    original_content: str,
    new_content: str,
    file_path: str,
) -> str:
    original_lines = original_content.splitlines(
        keepends=True
    )

    new_lines = new_content.splitlines(
        keepends=True
    )

    diff = difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )

    result = "".join(diff)

    if not result:
        return "No hay cambios."

    return result