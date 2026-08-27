# ComputahMind

A Discord bot with **local RAG** (Retrieval-Augmented Generation) and a
built-in **coding agent**, powered by [Ollama](https://ollama.com) so every
model (chat, embeddings and vision) runs locally, with no dependency on
external APIs.

Each Discord channel acts as an independent "project": it has its own
conversation memory, its own permanent facts, and its own collection of
indexed documents.

## Features

- **Per-channel memory**: conversation history and "permanent facts" (project
  name, tech stack, technical decisions, etc.) automatically extracted from
  messages and stored in SQLite.
- **Document & image RAG**: upload `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`,
  `.json`, or text/code files and they get indexed in ChromaDB to answer
  questions while citing sources. Images are described using a vision model
  and indexed as well.
- **Hybrid search**: the RAG combines semantic similarity (embeddings) with
  lexical keyword matching, plus a diversity factor so it doesn't always pull
  from the same source.
- **Coding Agent**: an agent that can explore a codebase on disk
  (`list_files`, `read_file`, `search_code`, `git_status`, `git_diff`),
  propose changes (`propose_change` / `propose_patch`) and propose
  verification commands (`propose_run`), all subject to **explicit approval**
  before anything is applied or executed.
- **Standalone CLI**: the Coding Agent can also be used outside of Discord,
  directly from the terminal.

## Architecture

```
src/
├── bot.py              # Discord client and "!" command handling
├── main.py             # Bot entry point
├── cli.py              # Coding Agent terminal entry point
├── config.py           # Environment variable loading (.env)
├── ollama_client.py    # HTTP client for /api/generate, /api/chat and vision
├── embeddings.py        # Embeddings client (Ollama /api/embed)
├── rag.py               # Indexing, hybrid search and context-based answers
├── memory.py            # Conversation memory and permanent facts (SQLite)
├── loaders.py            # Text extraction from PDF/DOCX/PPTX/XLSX/CSV/JSON/text
├── utils.py              # Text cleanup and chunking
├── agents/
│   ├── coding_agent.py   # Agent loop (tool calling over Ollama)
│   ├── permissions.py    # Permission levels per tool
│   ├── change_store.py   # Persistence of change proposals (diff/patch)
│   ├── change_request.py # Model of a change proposal
│   ├── run_store.py      # Persistence of run requests
│   └── run_request.py    # Model of a run request
└── tools/
    ├── file_tools.py     # list_files, read_file, read_file_lines, search_code, write_file
    ├── git_tools.py      # git_status, git_diff
    ├── diff_tools.py     # Unified diff generation
    ├── test_tools.py     # run_tests
    └── tool_registry.py  # Tool registry + permission enforcement
```

Persistent data (git-ignored, created automatically):

- `data/memory.db` — SQLite: messages, facts, per-channel workspace, pending proposals and runs.
- `data/chroma/` — ChromaDB: vectors for documents and images indexed per channel.
- `data/uploads/` — Attachments downloaded from Discord.
- `data/agent_runs/` and `data/workspaces/` — Coding Agent artifacts.

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) running locally, with the models you plan to
  use already pulled, for example:
  ```bash
  ollama pull qwen3.5:4b        # chat model
  ollama pull qwen3.5:9b        # coding agent model
  ollama pull qwen3-vl:4b       # vision model
  ollama pull nomic-embed-text  # embeddings
  ```
- A Discord bot with the **Message Content Intent** enabled.

## Installation

```bash
git clone git@github.com:ValeriaPzMl/local_LLM.git
cd local_LLM  # discord-rag-bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in the bot token:

```bash
cp .env.example .env
```

Available variables (all have defaults except `DISCORD_TOKEN`):

| Variable          | Description                                 | Default                     |
|-------------------|----------------------------------------------|------------------------------|
| `DISCORD_TOKEN`   | Discord bot token (**required**)            | —                            |
| `OLLAMA_URL`      | Ollama server URL                            | `http://localhost:11434`    |
| `OLLAMA_MODEL`    | Chat/generation model                        | `qwen3.5:4b`                |
| `EMBEDDING_MODEL` | Embeddings model                             | `nomic-embed-text`          |
| `VISION_MODEL`    | Vision model (image analysis)                | `qwen3-vl:4b`                |
| `CODING_MODEL`    | Model used by the Coding Agent               | `qwen3.5:9b`                |

## Usage

### Discord bot

```bash
python -m src.main
```

Once connected, in any channel you can:

- Type normally to chat (RAG kicks in automatically if the channel has
  indexed documents).
- Attach a document or image to index it (optionally with a question in the
  same message).

**General commands**

| Command | Description |
|---|---|
| `!docs` | Lists documents and images indexed in the channel |
| `!fuentes` | Shows the sources used in the last RAG-based answer |
| `!borrar-doc <id>` | Deletes an indexed document by its ID |
| `!estado` | Overall status: models, RAG, memory and services |
| `!formatos` | Supported document and image formats |
| `!memoria` | Number of messages stored in the history |
| `!hechos` | Lists the project's permanent facts |
| `!olvidar` | Clears the channel's conversation history |
| `!olvidar-hechos` | Clears the channel's permanent facts |

**Coding Agent commands** (require `!ruta` to be set first)

| Command | Description |
|---|---|
| `!ruta <path>` | Sets the project folder associated with the channel |
| `!proyecto` | Shows the configured workspace |
| `!codigo <task>` | Runs the agent on the workspace to inspect or modify code |
| `!propuestas` | Lists pending change proposals |
| `!diff <id>` | Shows the diff for a proposal |
| `!aprobar <id>` | Applies a change proposal and suggests a verification |
| `!rechazar <id>` | Discards a change proposal |
| `!ejecuciones` | Lists pending run requests (commands) |
| `!aprobar-run <id>` | Executes an approved request and analyzes the result |
| `!rechazar-run <id>` | Discards a run request |

### Coding Agent from the terminal

```bash
python -m src.cli --project /path/to/project
```

Opens an interactive prompt (`>>`) where you can give the agent tasks in
natural language; type `salir` (or `exit`/`quit`) to stop.

## The Coding Agent and its permissions

The agent never modifies or executes anything directly. Its tools are
classified into three levels (`src/agents/permissions.py`):

1. **Read-only** (`list_files`, `read_file`, `read_file_lines`,
   `search_code`, `git_status`, `git_diff`): run freely.
2. **Requires confirmation** (`write_file`, `apply_patch`, `run_tests`,
   `git_add`): the agent can only *propose* (`propose_change`,
   `propose_patch`, `propose_run`); the change stays pending until it's
   explicitly approved from Discord or the CLI.
3. **Blocked** (`run_sudo`, `delete_workspace`, `git_push`,
   `read_secrets`): never executed.

This flow prevents the agent from writing files, running commands, or
touching secrets (`.env`) without human supervision.

## Supported formats

- **Documents for RAG**: `.txt`, `.md`, `.py`, `.js`, `.ts`, `.html`, `.css`,
  `.java`, `.c`, `.cpp`, `.h`, `.hpp`, `.rs`, `.go`, `.sql`, `.yaml`, `.yml`,
  `.toml`, `.ini`, `.log`, `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`, `.json`.
- **Images**: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`.
