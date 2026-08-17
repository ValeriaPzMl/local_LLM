import json
from pathlib import Path

import aiohttp

from src.agents.permissions import PermissionLevel
from src.config import CODING_MODEL, OLLAMA_URL
from src.tools.tool_registry import ToolRegistry


CODING_SYSTEM_PROMPT = """
Eres ComputahMind Coding Agent, un agente especializado
en analizar proyectos de software.

Tienes acceso a herramientas de SOLO LECTURA.

Tu objetivo es inspeccionar el proyecto antes de responder.
No inventes archivos, funciones, clases ni resultados.

Reglas:

1. Antes de explicar cómo funciona una parte del proyecto,
   debes inspeccionar los archivos relevantes usando herramientas.
2. No afirmes haber leído un archivo si no utilizaste read_file.
3. No afirmes haber buscado código si no utilizaste search_code.
4. Puedes utilizar varias herramientas antes de responder.
5. No intentes modificar archivos.
6. No intentes ejecutar comandos arbitrarios.
7. No intentes acceder fuera del workspace.
8. No intentes leer secretos como .env o claves privadas.
9. Cuando tengas suficiente información, responde al usuario.
10. Responde siempre en español, aunque el código esté en inglés.
11. Encontrar el nombre o ubicación de un archivo NO significa
    haber inspeccionado su implementación.
12. Si una tarea pregunta cómo funciona código, debes usar read_file
    sobre los archivos relevantes antes de responder.
13. Nunca escribas código supuesto, simplificado o inventado como si
    fuera el contenido real de un archivo.
14. Si no has inspeccionado suficiente información, continúa usando
    herramientas en vez de responder.
15. Toda afirmación sobre la implementación debe estar respaldada
    por código que hayas leído mediante read_file.
16. Si afirmas que el proyecto usa una tecnología, base de datos,
    librería o framework, debes haber visto su importación, uso o
    configuración en el código leído.
17. Si no puedes verificar una afirmación en los archivos inspeccionados,
    debes decir "No pude verificarlo en el código leído".
18. Nunca sustituyas una tecnología real por otra basada en memoria
    previa del modelo.
19. En la respuesta final, incluye una sección "Evidencia" con referencias
    del tipo:
    - src/rag.py → RAGService.search()
    - src/rag.py → RAGService.build_context()
20. Si el usuario pregunta por el estado del repositorio,
    archivos modificados o cambios pendientes, usa git_status.
21. Si el usuario pregunta qué cambió en el código,
    utiliza git_diff.
22. Nunca respondas una pregunta sobre el estado actual de Git
    basándote en archivos leídos o conocimiento previo.
23. Para saber QUÉ archivos están modificados, usa git_status.
24. Para saber QUÉ CAMBIÓ dentro de los archivos, usa git_diff.
25. Una tarea de Git no requiere read_file salvo que el usuario pida
    analizar específicamente la implementación de un archivo.
26. Los resultados de git_status y git_diff son la fuente de verdad
    para preguntas sobre el estado actual del repositorio.
""".strip()


class CodingAgent:
    def __init__(
        self,
        workspace: Path,
        model: str = CODING_MODEL,
        base_url: str = OLLAMA_URL,
        max_steps: int = 10,
    ):
        self.workspace = workspace.resolve()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_steps = max_steps

        self.tools = ToolRegistry(
            workspace=self.workspace
        )

        self.timeout = aiohttp.ClientTimeout(
            total=300
        )

    def get_tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": (
                        "Lista archivos y carpetas del workspace. "
                        "Úsala para conocer la estructura del proyecto."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "relative_path": {
                                "type": "string",
                                "description": (
                                    "Ruta relativa dentro del workspace. "
                                    "Usa '.' para la raíz."
                                ),
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": (
                        "Lee el contenido de un archivo de texto "
                        "dentro del workspace."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "relative_path": {
                                "type": "string",
                                "description": (
                                    "Ruta relativa del archivo."
                                ),
                            },
                        },
                        "required": [
                            "relative_path"
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_code",
                    "description": (
                        "Busca texto dentro del código del proyecto."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "Texto, símbolo, función o clase "
                                    "que se desea buscar."
                                ),
                            },
                            "relative_path": {
                                "type": "string",
                                "description": (
                                    "Ruta relativa donde buscar. "
                                    "Usa '.' para todo el proyecto."
                                ),
                            },
                        },
                        "required": [
                            "query"
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_status",
                    "description": (
                        "Muestra el estado actual del repositorio Git."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_diff",
                    "description": (
                        "Muestra los cambios actuales del repositorio Git. "
                        "Puede limitarse a un archivo específico."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "staged": {
                                "type": "boolean",
                                "description": (
                                    "Si es true, muestra cambios staged."
                                ),
                            },
                            "relative_path": {
                                "type": "string",
                                "description": (
                                    "Archivo específico cuyo diff "
                                    "se desea consultar."
                                ),
                            },
                        },
                    },
                },
            },
        ]

    async def _chat(
        self,
        messages: list[dict],
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": self.get_tool_schemas(),
            "stream": False,
            "think": False,
        }

        async with aiohttp.ClientSession(
            timeout=self.timeout,
        ) as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                data = await response.json()

                if response.status != 200:
                    error_message = data.get(
                        "error",
                        "Error desconocido",
                    )

                    raise RuntimeError(
                        f"Ollama respondió con HTTP "
                        f"{response.status}: {error_message}"
                    )

        return data
    def _parse_json_tool_call(
        self,
        content: str,
    ) -> dict | None:
        """
        Algunos modelos devuelven la llamada a herramienta
        como JSON dentro de content en vez de usar tool_calls.
        """

        content = content.strip()

        if not content:
            return None

        if content.startswith("```"):
            lines = content.splitlines()

            if len(lines) >= 3:
                content = "\n".join(
                    lines[1:-1]
                ).strip()

                if content.startswith("json"):
                    content = content[4:].strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        tool_name = data.get("name")
        arguments = data.get(
            "arguments",
            {},
        )

        if not isinstance(tool_name, str):
            return None

        if not isinstance(arguments, dict):
            arguments = {}

        if self.tools.get(tool_name) is None:
            return None

        return {
            "function": {
                "name": tool_name,
                "arguments": arguments,
            }
        }

    def _requires_file_inspection(
        self,
        task: str,
    ) -> bool:
        """
        Determina si una tarea requiere leer código real antes
        de permitir una respuesta final.
        """

        task_lower = task.lower()

        inspection_keywords = {
            "cómo funciona",
            "como funciona",
            "implementación",
            "implementacion",
            "explica",
            "analiza",
            "revisa",
            "clase",
            "función",
            "funcion",
            "método",
            "metodo",
            "bug",
            "error",
            "código",
            "codigo",
        }

        return any(
            keyword in task_lower
            for keyword in inspection_keywords
        )
    def _required_tools_for_task(
        self,
        task: str,
    ) -> set[str]:
        task_lower = task.lower()

        required: set[str] = set()

        # Preguntas sobre qué archivos están modificados.
        status_phrases = {
            "estado de git",
            "git status",
            "cambios pendientes",
            "archivos modificados",
            "archivos con cambios",
            "cambios sin commit",
            "cambios sin guardar en commits",
        }

        # Preguntas sobre el contenido exacto de los cambios.
        diff_phrases = {
            "git diff",
            "qué cambió en el código",
            "que cambio en el codigo",
            "muéstrame el diff",
            "muestrame el diff",
            "muéstrame los cambios exactos",
            "muestrame los cambios exactos",
            "qué líneas cambiaron",
            "que lineas cambiaron",
        }

        if any(
            phrase in task_lower
            for phrase in status_phrases
        ):
            required.add("git_status")

        if any(
            phrase in task_lower
            for phrase in diff_phrases
        ):
            required.add("git_diff")

        return required
    async def run(
        self,
        task: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": CODING_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": task,
            },
        ]
        attempted_tools: list[str] = []
        successful_tools: list[str] = []

        for step in range(
            1,
            self.max_steps + 1,
        ):
            print(
                f"[CodingAgent] Paso "
                f"{step}/{self.max_steps}"
            )

            data = await self._chat(
                messages
            )

            assistant_message = data.get(
                "message",
                {},
            )

            messages.append(
                assistant_message
            )

            tool_calls = assistant_message.get(
                "tool_calls",
                [],
            )

            content = assistant_message.get(
                "content",
                "",
            ).strip()

            if not tool_calls and content:
                json_tool_call = self._parse_json_tool_call(
                    content
                )

                if json_tool_call is not None:
                    print(
                        "[CodingAgent] Detecté una llamada "
                        "JSON en content."
                    )

                    tool_calls = [
                        json_tool_call
                    ]

            if not tool_calls:
                if not content:
                    print(
                        "[CodingAgent] El modelo devolvió "
                        "una respuesta vacía."
                    )

                    print(
                        "[CodingAgent] Respuesta completa de Ollama:"
                    )

                    print(data)

                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Tu respuesta anterior quedó vacía. "
                                "Continúa con la tarea. "
                                "Si necesitas información adicional, "
                                "usa una herramienta. "
                                "Si ya tienes suficiente evidencia, "
                                "entrega la respuesta final."
                            ),
                        }
                    )

                    continue

                # 1. Primero comprobamos herramientas obligatorias.
                required_tools = self._required_tools_for_task(
                    task
                )
                print(
                    f"[CodingAgent] Herramientas requeridas: "
                    f"{required_tools}"
                )
                missing_tools = (
                    required_tools
                    - set(successful_tools)
                )

                if missing_tools:
                    missing_text = ", ".join(
                        sorted(missing_tools)
                    )

                    print(
                        "[CodingAgent] Respuesta rechazada: "
                        f"faltan herramientas: {missing_text}"
                    )

                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "No respondas todavía.\n"
                                "Para contestar esta tarea debes utilizar "
                                "la siguiente herramienta:\n"
                                f"{missing_text}\n\n"
                                "Haz la llamada a la herramienta ahora."
                            ),
                        }
                    )

                    continue

                # 2. Después verificamos si necesitaba leer código.
                required_tools = self._required_tools_for_task(
                    task
                )

                is_git_task = bool(
                    {"git_status", "git_diff"}
                    & required_tools
                )

                requires_inspection = (
                    self._requires_file_inspection(task)
                    and not is_git_task
                )

                has_read_file = (
                    "read_file" in successful_tools
                )

                if requires_inspection and not has_read_file:
                    print(
                        "[CodingAgent] Respuesta rechazada: "
                        "todavía no leyó ningún archivo."
                    )

                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "No respondas todavía. "
                                "Debes inspeccionar primero el código real. "
                                "Usa read_file sobre el archivo relevante."
                            ),
                        }
                    )

                    continue

                return self._clean_final_response(
                    content
                )

            for tool_call in tool_calls:
                function_data = tool_call.get(
                    "function",
                    {},
                )

                tool_name = function_data.get(
                    "name",
                    "",
                )
                attempted_tools.append(tool_name)

                arguments = function_data.get(
                    "arguments",
                    {},
                )

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(
                            arguments
                        )
                    except json.JSONDecodeError:
                        arguments = {}

                print(
                    f"[CodingAgent] Tool: "
                    f"{tool_name}"
                )

                print(
                    f"[CodingAgent] Args: "
                    f"{arguments}"
                )

                tool = self.tools.get(
                    tool_name
                )

                if tool is None:
                    result = (
                        f"Error: herramienta desconocida "
                        f"`{tool_name}`."
                    )

                elif (
                    tool.permission
                    != PermissionLevel.READ_ONLY
                ):
                    result = (
                        f"Error: la herramienta "
                        f"`{tool_name}` no está permitida "
                        "en modo solo lectura."
                    )

                else:
                    try:
                        result = self.tools.execute(
                            name=tool_name,
                            arguments=arguments,
                        )

                        successful_tools.append(
                            tool_name
                        )

                    except Exception as error:
                        result = (
                            f"Error ejecutando "
                            f"`{tool_name}`: {error}"
                        )

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": str(result),
                    }
                )
                if tool_name == "git_status":
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "El resultado anterior proviene de git_status. "
                                "Úsalo como fuente de verdad para determinar "
                                "qué archivos tienen cambios pendientes. "
                                "No inspecciones archivos individuales a menos "
                                "que el usuario pregunte qué cambió dentro de ellos."
                            ),
                        }
                    )

                elif tool_name == "git_diff":
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "El resultado anterior proviene de git_diff. "
                                "Úsalo como fuente de verdad para explicar "
                                "los cambios concretos del repositorio. "
                                "No inventes cambios que no aparezcan ahí."
                            ),
                        }
                    )

        return (
            "El agente alcanzó el límite de pasos "
            "sin completar la tarea."
        )
    def _clean_final_response(
        self,
        content: str,
    ) -> str:
        content = content.strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return content

        if (
            isinstance(data, dict)
            and isinstance(data.get("response"), str)
        ):
            return data["response"].strip()

        return content