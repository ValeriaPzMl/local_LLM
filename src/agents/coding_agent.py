import json
from pathlib import Path

import aiohttp

from src.agents.permissions import PermissionLevel
from src.config import CODING_MODEL, OLLAMA_URL
from src.tools.tool_registry import ToolRegistry
from src.agents.change_request import (
    ChangeRequest,
    create_change_request,
)
from src.tools.diff_tools import create_unified_diff
from src.tools.file_tools import resolve_safe_path
from src.agents.change_store import ChangeStore
from src.agents.run_request import (
    RunRequest,
    create_run_request,
)
from src.agents.run_store import RunStore

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
27. Cuando el usuario solicite modificar código, primero inspecciona
    los archivos relevantes.
28. Nunca uses write_file directamente.
29. Para modificar código debes usar propose_change.
30. propose_change NO modifica el archivo; únicamente crea una
    propuesta pendiente para que el usuario la revise.
31. El contenido de new_content debe representar el archivo completo
    después de aplicar la modificación.
32. Después de crear una propuesta, explica brevemente qué cambiaría
    y espera aprobación.
33. Crear una propuesta NO requiere aprobación del usuario.
34. Si el usuario pide modificar código, debes crear directamente
    una propuesta con propose_change.
35. No preguntes "¿quieres que proponga el cambio?". Proponer es seguro.
36. La aprobación solamente es necesaria para APLICAR una propuesta.
37. Después de usar propose_change, informa el ID de la propuesta
    y espera aprobación.
38. No ejecutes pruebas ni comandos directamente.
39. Cuando necesites verificar un cambio, usa propose_run.
40. propose_run solo crea una solicitud pendiente.
41. La ejecución real requiere aprobación explícita del usuario.
42. Después de modificar código, recomienda una verificación
    adecuada cuando sea útil.
43. Para cambios pequeños o medianos en archivos existentes,
    prefiere propose_patch sobre propose_change.
44. Antes de usar propose_patch debes haber leído el código
    relevante mediante read_file o read_file_lines.
45. old_text debe coincidir exactamente con el contenido real
    del archivo.
46. No inventes old_text.
47. Usa propose_change solamente cuando necesites reemplazar
    prácticamente todo el archivo o crear una reestructuración grande.
48. Cuando conozcas la zona concreta del archivo, prefiere
    read_file_lines para reducir el contexto utilizado.
49. Si el usuario pide modificar solamente una función, método,
    bloque o sección concreta de un archivo existente, debes usar
    propose_patch y NO propose_change.
50. propose_change se reserva para archivos nuevos, reestructuraciones
    grandes o reemplazos casi completos del archivo.
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

        if not self.workspace.exists():
            raise ValueError(
                f"El workspace no existe: {self.workspace}"
            )

        if not self.workspace.is_dir():
            raise ValueError(
                f"El workspace debe ser una carpeta: {self.workspace}"
            )

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_steps = max_steps

        self.tools = ToolRegistry(
            workspace=self.workspace
        )
        self.change_store = ChangeStore(
            "data/memory.db"
        )
        self.run_store = RunStore(
            "data/memory.db"
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
            {
                "type": "function",
                "function": {
                    "name": "propose_change",
                    "description": (
                        "Propone una modificación completa para un archivo. "
                        "NO aplica el cambio. El usuario deberá aprobarlo."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": (
                                    "Ruta relativa del archivo a modificar."
                                ),
                            },
                            "description": {
                                "type": "string",
                                "description": (
                                    "Descripción breve del cambio."
                                ),
                            },
                            "new_content": {
                                "type": "string",
                                "description": (
                                    "Contenido completo que debería tener "
                                    "el archivo después del cambio."
                                ),
                            },
                        },
                        "required": [
                            "file_path",
                            "description",
                            "new_content",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "propose_run",
                    "description": (
                        "Propone ejecutar una prueba o verificación. "
                        "No ejecuta nada sin aprobación."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                            },
                            "description": {
                                "type": "string",
                            },
                        },
                        "required": [
                            "command",
                            "description",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file_lines",
                    "description": (
                        "Lee solamente un rango de líneas de un archivo. "
                        "Prefiere esta herramienta sobre read_file cuando "
                        "ya sabes dónde se encuentra el código relevante."
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
                            "start_line": {
                                "type": "integer",
                                "description": (
                                    "Primera línea a leer, empezando en 1."
                                ),
                            },
                            "end_line": {
                                "type": "integer",
                                "description": (
                                    "Última línea a leer."
                                ),
                            },
                        },
                        "required": [
                            "relative_path",
                            "start_line",
                            "end_line",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "propose_patch",
                    "description": (
                        "Propone reemplazar una sección concreta de un archivo. "
                        "No modifica el archivo. Prefiere esta herramienta sobre "
                        "propose_change para cambios pequeños o medianos."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                            },
                            "old_text": {
                                "type": "string",
                                "description": (
                                    "Texto exacto actualmente existente "
                                    "que debe reemplazarse."
                                ),
                            },
                            "new_text": {
                                "type": "string",
                                "description": (
                                    "Texto que reemplazará old_text."
                                ),
                            },
                            "description": {
                                "type": "string",
                            },
                        },
                        "required": [
                            "file_path",
                            "old_text",
                            "new_text",
                            "description",
                        ],
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
            "options": {
                "num_ctx": 4096,
                "num_predict": 800,
                "temperature": 0.1,
            },
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

        virtual_tools = {
            "propose_change",
            "propose_patch",
            "propose_run",
        }

        if (
            self.tools.get(tool_name) is None
            and tool_name not in virtual_tools
        ):
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
    def _requires_change_proposal(
        self,
        task: str,
    ) -> bool:
        task_lower = task.lower()

        change_keywords = {
            "modifica",
            "modificar",
            "cambia",
            "cambiar",
            "agrega",
            "agregar",
            "añade",
            "añadir",
            "corrige",
            "corregir",
            "arregla",
            "arreglar",
            "implementa",
            "implementar",
            "refactoriza",
            "refactorizar",
            "elimina",
            "eliminar",
            "crea",
            "crear",
        }

        return any(
            keyword in task_lower
            for keyword in change_keywords
        )
    def _prefers_patch(
        self,
        task: str,
    ) -> bool:
        task_lower = task.lower()

        patch_keywords = {
            "únicamente",
            "unicamente",
            "solo",
            "solamente",
            "función",
            "funcion",
            "método",
            "metodo",
            "bloque",
            "sección",
            "seccion",
        }

        return any(
            keyword in task_lower
            for keyword in patch_keywords
        )
    def create_pending_change(
        self,
        file_path: str,
        original_content: str,
        new_content: str,
        description: str,
    ) -> ChangeRequest:
        change = create_change_request(
            workspace=self.workspace,
            file_path=file_path,
            original_content=original_content,
            new_content=new_content,
            description=description,
        )

        self.change_store.save(
            change
        )

        return change
    def get_pending_change(
        self,
        change_id: str,
    ) -> ChangeRequest | None:
        return self.change_store.get(
            change_id
        )
    def list_pending_changes(
        self,
    ) -> list[ChangeRequest]:
        return self.change_store.list_pending(
            workspace=self.workspace
        )
    def get_change_diff(
        self,
        change_id: str,
    ) -> str:
        change = self.get_pending_change(
            change_id
        )

        if change is None:
            raise ValueError(
                f"No existe la propuesta {change_id}."
            )

        return create_unified_diff(
            original_content=change.original_content,
            new_content=change.new_content,
            file_path=change.file_path,
        )
    def approve_change(
        self,
        change_id: str,
    ) -> str:
        change = self.get_pending_change(
            change_id
        )

        if change is None:
            raise ValueError(
                f"No existe la propuesta {change_id}."
            )

        if change.rejected:
            raise ValueError(
                "La propuesta fue rechazada."
            )

        if change.applied:
            return (
                f"La propuesta {change_id} "
                "ya fue aplicada."
            )

        tool = self.tools.get(
            "write_file"
        )

        if tool is None:
            raise RuntimeError(
                "write_file no está registrado."
            )

        # No usamos self.tools.execute() porque ese método
        # bloquearía correctamente las herramientas Nivel 2.
        # Aquí Python la ejecuta después de aprobación explícita.
        result = tool.function(
            self.workspace,
            relative_path=change.file_path,
            content=change.new_content,
        )

        change.approved = True
        change.applied = True
        self.change_store.update_status(
            change
        )

        return str(result)
    def reject_change(
        self,
        change_id: str,
    ) -> str:
        change = self.get_pending_change(
            change_id
        )

        if change is None:
            raise ValueError(
                f"No existe la propuesta {change_id}."
            )

        if change.applied:
            raise ValueError(
                "No puedes rechazar una propuesta "
                "que ya fue aplicada."
            )

        change.rejected = True
        self.change_store.update_status(
            change
        )

        return (
            f"Propuesta {change_id} rechazada."
        )
    def create_pending_run(
        self,
        command: str,
        description: str,
    ) -> RunRequest:
        run = create_run_request(
            command=command,
            description=description,
        )

        self.run_store.save(
            self.workspace,
            run,
        )

        return run


    def approve_run(
        self,
        run_id: str,
    ) -> str:
        run = self.run_store.get(
            run_id
        )

        if run is None:
            raise ValueError(
                f"No existe la ejecución {run_id}."
            )

        if run.rejected:
            raise ValueError(
                "La ejecución fue rechazada."
            )

        if run.executed:
            return run.result or "Ya fue ejecutada."

        tool = self.tools.get(
            "run_tests"
        )

        if tool is None:
            raise RuntimeError(
                "run_tests no está registrado."
            )

        result = tool.function(
            self.workspace,
            command=run.command,
        )

        run.approved = True
        run.executed = True
        run.result = result

        self.run_store.save(
            self.workspace,
            run,
        )

        return result
    def reject_run(
        self,
        run_id: str,
    ) -> str:
        run = self.run_store.get(
            run_id
        )

        if run is None:
            raise ValueError(
                f"No existe la ejecución {run_id}."
            )

        if run.executed:
            raise ValueError(
                "No puedes rechazar una ejecución "
                "que ya fue realizada."
            )

        run.rejected = True

        self.run_store.save(
            self.workspace,
            run,
        )

        return (
            f"Ejecución {run_id} rechazada."
        )
    def list_pending_runs(
        self,
    ) -> list[RunRequest]:
        return self.run_store.list_pending(
            workspace=self.workspace
        )
    async def propose_verification(
        self,
        change_id: str,
    ) -> str:
        change = self.get_pending_change(
            change_id
        )

        if change is None:
            raise ValueError(
                f"No existe la propuesta {change_id}."
            )

        if not change.applied:
            raise ValueError(
                "El cambio todavía no ha sido aplicado."
            )

        diff = self.get_change_diff(
            change_id
        )

        task = f"""
    Acaba de aplicarse un cambio de código.

    ARCHIVO MODIFICADO:
    {change.file_path}

    DESCRIPCIÓN:
    {change.description}

    DIFF APLICADO:
    {diff}

    Debes proponer una verificación adecuada para comprobar que
    el cambio no introdujo errores.

    Reglas:

    - Usa propose_run.
    - No modifiques ningún archivo.
    - No ejecutes comandos directamente.
    - Prefiere una verificación pequeña y específica.
    - Para Python puedes usar, según corresponda:
    python -m compileall <archivo>
    pytest
    - Si existe una prueba específica relevante, prefiérela sobre
    ejecutar todo el proyecto.
    """.strip()

        return await self.run(
            task=task,
            force_change_proposal=False,
            force_run_proposal=True,
        )
    async def review_run_result(
        self,
        run_id: str,
    ) -> str:
        run = self.run_store.get(
            run_id
        )

        if run is None:
            raise ValueError(
                f"No existe la ejecución {run_id}."
            )

        if not run.executed:
            raise ValueError(
                "La ejecución todavía no ha sido realizada."
            )

        if not run.result:
            raise ValueError(
                "La ejecución no tiene ningún resultado guardado."
            )

        result = run.result.strip()

        # Si todo salió bien, no gastamos otra llamada al LLM.
        if result.startswith("Exit code: 0"):
            return (
                "✅ La verificación terminó correctamente.\n\n"
                f"Comando: `{run.command}`\n"
                "No se detectaron errores."
            )

        task = f"""
    Una verificación del proyecto falló.

    COMANDO EJECUTADO:
    {run.command}

    RESULTADO:
    {result}

    Analiza este error.

    Debes:
    1. Identificar la causa probable.
    2. Inspeccionar los archivos reales del proyecto usando herramientas.
    3. No inventar implementaciones.
    4. Si la solución requiere modificar código, crea directamente
    una propuesta usando propose_change.
    5. No apliques ningún cambio.
    6. Explica brevemente el problema y cualquier propuesta creada.
    """.strip()

        return await self.run(
            task,
            force_change_proposal=False,
        )
    async def run(
        self,
        task: str,
        force_change_proposal: bool | None = None,
        force_run_proposal: bool = False,
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
        created_change_ids: list[str] = []

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
                    raise RuntimeError(
                        "Ollama devolvió una respuesta vacía. "
                        "Probablemente el modelo se quedó sin memoria."
                    )


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
                    or "read_file_lines" in successful_tools
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
                if force_change_proposal is None:
                    requires_change = (
                        self._requires_change_proposal(task)
                    )
                else:
                    requires_change = force_change_proposal
                    
                prefers_patch = self._prefers_patch(
                    task
                )

                has_patch = (
                    "propose_patch" in successful_tools
                )

                has_full_change = (
                    "propose_change" in successful_tools
                )

                has_proposed_change = (
                    "propose_change" in successful_tools
                    or "propose_patch" in successful_tools
                )

                if requires_change and not has_proposed_change:
                    print(
                        "[CodingAgent] Respuesta rechazada: "
                        "la tarea requiere una propuesta real."
                    )

                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "No puedes terminar todavía. "
                                "El usuario pidió modificar código y aún "
                                "no has creado una propuesta real.\n\n"
                                "Debes usar propose_change ahora. "
                                "No preguntes si puede proponer el cambio: "
                                "crea la propuesta sin aplicarla. "
                                "La aprobación ocurrirá después."
                            ),
                        }
                    )

                    continue
                has_proposed_run = (
                    "propose_run" in successful_tools
                )

                if force_run_proposal and not has_proposed_run:
                    print(
                        "[CodingAgent] Respuesta rechazada: "
                        "falta una propuesta de verificación."
                    )

                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "No puedes terminar todavía. "
                                "Debes crear una propuesta de verificación "
                                "usando propose_run.\n\n"
                                "No ejecutes nada. "
                                "Solo crea la propuesta de ejecución."
                            ),
                        }
                    )

                    continue
                return self._clean_final_response(
                    content
                )
            executed_tool_calls = set()
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
                tool_key = (
                    tool_name,
                    json.dumps(
                        arguments,
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                )

                if tool_key in executed_tool_calls:
                    print(
                        f"[CodingAgent] Tool duplicada ignorada: "
                        f"{tool_name}"
                    )
                    continue

                executed_tool_calls.add(tool_key)

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(
                            arguments
                        )
                    except json.JSONDecodeError:
                        arguments = {}

                print(
                    f"[CodingAgent] Tool: {tool_name}"
                )

                print(
                    f"[CodingAgent] Args: {arguments}"
                )
                if tool_name == "propose_change":
                    # Si la tarea claramente pide tocar solo una parte,
                    # rechazamos el cambio completo.
                    if self._prefers_patch(task):
                        result = (
                            "Esta tarea requiere una modificación localizada. "
                            "No uses propose_change. "
                            "Debes usar propose_patch con old_text exacto "
                            "y new_text."
                        )

                        messages.append(
                            {
                                "role": "tool",
                                "tool_name": tool_name,
                                "content": result,
                            }
                        )

                        continue

                    try:
                        file_path = arguments[
                            "file_path"
                        ]

                        description = arguments[
                            "description"
                        ]

                        new_content = arguments[
                            "new_content"
                        ]

                        target = resolve_safe_path(
                            self.workspace,
                            file_path,
                        )

                        if not target.exists():
                            raise ValueError(
                                f"No existe el archivo {file_path}."
                            )

                        original_content = target.read_text(
                            encoding="utf-8",
                            errors="replace",
                        )

                        change = self.create_pending_change(
                            file_path=file_path,
                            original_content=original_content,
                            new_content=new_content,
                            description=description,
                        )

                        created_change_ids.append(
                            change.id
                        )

                        successful_tools.append(
                            "propose_change"
                        )

                        diff = self.get_change_diff(
                            change.id
                        )

                        result = (
                            "PROPUESTA CREADA\n"
                            f"ID: {change.id}\n"
                            f"Archivo: {file_path}\n"
                            f"Descripción: {description}\n\n"
                            f"DIFF:\n{diff}"
                        )

                    except Exception as error:
                        result = (
                            "Error creando propuesta: "
                            f"{error}"
                        )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": tool_name,
                            "content": result,
                        }
                    )

                    continue
                if tool_name == "propose_patch":
                    try:
                        file_path = arguments[
                            "file_path"
                        ]

                        old_text = arguments[
                            "old_text"
                        ]

                        new_text = arguments[
                            "new_text"
                        ]

                        description = arguments[
                            "description"
                        ]

                        target = resolve_safe_path(
                            self.workspace,
                            file_path,
                        )

                        if not target.exists():
                            raise ValueError(
                                f"No existe el archivo {file_path}."
                            )

                        original_content = target.read_text(
                            encoding="utf-8",
                            errors="replace",
                        )

                        occurrence_count = (
                            original_content.count(
                                old_text
                            )
                        )

                        if occurrence_count == 0:
                            raise ValueError(
                                "El texto original propuesto no existe "
                                "exactamente en el archivo."
                            )

                        if occurrence_count > 1:
                            raise ValueError(
                                "El texto original aparece varias veces. "
                                "La propuesta es ambigua; lee más contexto "
                                "y vuelve a intentarlo."
                            )

                        new_content = original_content.replace(
                            old_text,
                            new_text,
                            1,
                        )

                        change = self.create_pending_change(
                            file_path=file_path,
                            original_content=original_content,
                            new_content=new_content,
                            description=description,
                        )

                        created_change_ids.append(
                            change.id
                        )


                        successful_tools.append(
                            "propose_patch"
                        )

                        diff = self.get_change_diff(
                            change.id
                        )

                        result = (
                            "PATCH PROPUESTO\n"
                            f"ID: {change.id}\n"
                            f"Archivo: {file_path}\n"
                            f"Descripción: {description}\n\n"
                            f"DIFF:\n{diff}"
                        )

                    except Exception as error:
                        result = (
                            "Error creando patch: "
                            f"{error}"
                        )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": tool_name,
                            "content": result,
                        }
                    )

                    continue
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
                if tool_name == "propose_run":
                    try:
                        command = arguments[
                            "command"
                        ]

                        description = arguments[
                            "description"
                        ]

                        run = self.create_pending_run(
                            command=command,
                            description=description,
                        )

                        successful_tools.append(
                            "propose_run"
                        )

                        result = (
                            "EJECUCIÓN PROPUESTA\n"
                            f"ID: {run.id}\n"
                            f"Comando: {run.command}\n"
                            f"Descripción: {run.description}"
                        )

                    except Exception as error:
                        result = (
                            "Error creando ejecución: "
                            f"{error}"
                        )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_name": tool_name,
                            "content": result,
                        }
                    )

                    continue
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
    