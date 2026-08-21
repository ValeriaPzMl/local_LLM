from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agents.permissions import (
    PermissionLevel,
    get_permission_level,
)
from src.tools.file_tools import (
    list_files,
    read_file,
    search_code,
    write_file,
)
from src.tools.git_tools import (
    git_diff,
    git_status,
)


@dataclass
class ToolDefinition:
    name: str
    description: str
    function: Callable[..., Any]
    permission: PermissionLevel


class ToolRegistry:
    def __init__(
        self,
        workspace: Path,
    ):
        self.workspace = workspace.resolve()
        self._tools: dict[
            str,
            ToolDefinition,
        ] = {}

        self._register_default_tools()

    def _register_default_tools(
        self,
    ) -> None:
        self.register(
            name="list_files",
            description=(
                "Lista archivos y carpetas del workspace. "
                "Argumentos: relative_path opcional."
            ),
            function=list_files,
        )

        self.register(
            name="read_file",
            description=(
                "Lee un archivo de texto. "
                "Argumentos: relative_path."
            ),
            function=read_file,
        )

        self.register(
            name="search_code",
            description=(
                "Busca texto dentro de archivos. "
                "Argumentos: query y relative_path opcional."
            ),
            function=search_code,
        )

        self.register(
            name="git_status",
            description=(
                "Muestra el estado actual de Git."
            ),
            function=git_status,
        )

        self.register(
            name="git_diff",
            description=(
                "Muestra cambios de Git. "
                "Argumento opcional: staged."
            ),
            function=git_diff,
        )
        self.register(
            name="write_file",
            description=(
                "Escribe contenido en un archivo del workspace. "
                "Requiere aprobación explícita del usuario."
            ),
            function=write_file,
        )

    def register(
        self,
        name: str,
        description: str,
        function: Callable[..., Any],
    ) -> None:
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            function=function,
            permission=get_permission_level(
                name
            ),
        )

    def get(
        self,
        name: str,
    ) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(
        self,
    ) -> list[ToolDefinition]:
        return list(
            self._tools.values()
        )

    def execute(
        self,
        name: str,
        arguments: dict | None = None,
    ) -> str:
        tool = self.get(name)

        if tool is None:
            raise ValueError(
                f"Herramienta desconocida: {name}"
            )

        if (
            tool.permission
            == PermissionLevel.BLOCKED
        ):
            raise PermissionError(
                f"La herramienta `{name}` está bloqueada."
            )

        if (
            tool.permission
            == PermissionLevel.CONFIRMATION_REQUIRED
        ):
            raise PermissionError(
                f"La herramienta `{name}` requiere confirmación."
            )

        arguments = arguments or {}

        result = tool.function(
            self.workspace,
            **arguments,
        )

        return str(result)

    def describe_tools(
        self,
    ) -> str:
        return "\n".join(
            (
                f"- {tool.name}: "
                f"{tool.description} "
                f"[nivel {tool.permission.value}]"
            )
            for tool in self.list_tools()
        )