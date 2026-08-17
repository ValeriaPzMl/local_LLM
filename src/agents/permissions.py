from enum import IntEnum


class PermissionLevel(IntEnum):
    READ_ONLY = 1
    CONFIRMATION_REQUIRED = 2
    BLOCKED = 3


TOOL_PERMISSIONS = {
    # Nivel 1: lectura segura
    "list_files": PermissionLevel.READ_ONLY,
    "read_file": PermissionLevel.READ_ONLY,
    "search_code": PermissionLevel.READ_ONLY,
    "git_status": PermissionLevel.READ_ONLY,
    "git_diff": PermissionLevel.READ_ONLY,

    # Nivel 2: cambios con confirmación
    "write_file": PermissionLevel.CONFIRMATION_REQUIRED,
    "apply_patch": PermissionLevel.CONFIRMATION_REQUIRED,
    "run_tests": PermissionLevel.CONFIRMATION_REQUIRED,
    "git_add": PermissionLevel.CONFIRMATION_REQUIRED,

    # Nivel 3: bloqueado
    "run_sudo": PermissionLevel.BLOCKED,
    "delete_workspace": PermissionLevel.BLOCKED,
    "git_push": PermissionLevel.BLOCKED,
    "read_secrets": PermissionLevel.BLOCKED,
}


def get_permission_level(
    tool_name: str,
) -> PermissionLevel:
    return TOOL_PERMISSIONS.get(
        tool_name,
        PermissionLevel.BLOCKED,
    )