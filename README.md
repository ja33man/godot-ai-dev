# GodotAI Dev

```
  ██████╗  ██████╗ ██████╗  ██████╗ ████████╗     █████╗ ██╗
 ██╔════╝ ██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝    ██╔══██╗██║
 ██║  ███╗██║   ██║██║  ██║██║   ██║   ██║       ███████║██║
 ██║   ██║██║   ██║██║  ██║██║   ██║   ██║       ██╔══██║██║
 ╚██████╔╝╚██████╔╝██████╔╝╚██████╔╝   ██║       ██║  ██║██║
  ╚═════╝  ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝       ╚═╝  ╚═╝╚═╝
```

**Local, free, codebase-aware AI assistant for Godot 4 game development.**  
No cloud. No API keys. No subscription. Powered by [Ollama](https://ollama.com).

---

## How it works

```
Your PC / Mac                           Remote server (Linux + GPU)
─────────────────────────────────       ──────────────────────────
  Godot (your game)                       Ollama  ← AI models live here
  godot_ai.py  ──── LAN ──────────→      192.168.2.49:11434
  your .gd files
  ChromaDB index (local)
  git repo (local)
```

- `godot_ai.py` runs **on your PC/Mac**, next to your Godot project
- Ollama runs on the **remote server** (or localhost) and does the GPU work
- Your code never leaves your LAN

---

## Requirements

| What | Where |
|------|-------|
| Python 3.10+ | Your PC / Mac |
| Godot 4 project | Your PC / Mac |
| git (optional but recommended) | Your PC / Mac |
| Ollama | Linux server with GPU, or localhost |

### Models to pull on the Ollama server

```bash
ollama pull nomic-embed-text      # ~274 MB — indexes your codebase
ollama pull qwen2.5-coder:14b     # ~9 GB   — writes and fixes GDScript
ollama pull qwen3.5:2b            # ~2.7 GB — fast chat answers
```

---

## Installation

```bash
git clone https://github.com/ja33man/godot-ai-dev.git
cd godot-ai-dev
```

**Windows:**
```cmd
pip install -r requirements.txt
```

**Mac / Linux:**
```bash
pip3 install -r requirements.txt
```

Dependencies: `ollama`, `chromadb`, `rich`, `watchdog`, `prompt_toolkit`

---

## Running

### Windows
```cmd
python godot_ai.py C:\Users\you\Documents\MyGame --host http://192.168.2.49:11434
```

### Mac / Linux
```bash
python3 godot_ai.py ~/Documents/MyGame --host http://192.168.2.49:11434
```

### No arguments — interactive prompt
```
python godot_ai.py
📁 Godot project path: C:\Users\you\Documents\MyGame
🖥  Ollama host [http://localhost:11434]: http://192.168.2.49:11434
```

### Other flags
```bash
python godot_ai.py ./mygame --host http://... --no-git
python godot_ai.py ./mygame --host http://... --model qwen2.5-coder:32b
```

### Make sure Ollama accepts remote connections
```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

---

## Commands

### Core

| Command | What it does |
|---------|-------------|
| `/fix` | Paste a Godot error → AI diagnoses → patches your files |
| `/feature` | Describe new functionality → AI generates GDScript → writes files |
| `/chat` | Ask anything about Godot or your project (no file writes) |
| `/retry` | Re-run the last `/fix`, `/feature`, or `/chat` with a stricter prompt |
| `/index` | Re-scan and re-index the whole project |
| `/tree` | Show the full project file tree |
| `/files` | List all indexed files |
| `/models` | List Ollama models, switch which model is used for code/chat |
| `/clear` | Reset conversation history |
| `/help` | Show the command panel |
| `/quit` | Exit |

### Godot live integration

| Command | What it does |
|---------|-------------|
| `/run` | Launch Godot from here — output streams to this terminal, errors trigger auto-fix offer |
| `/run C:/path/godot.exe` | Same but specify the Godot executable path (remembered for next time) |
| `/watch` | Watch Godot's log file when Godot is open separately |
| `/last` | Show the last captured error batch and offer to fix it |

### Git safety net

| Command | What it does |
|---------|-------------|
| `/status` | `git status` |
| `/diff` | Changes since session started |
| `/log` | Commits made this session |
| `/history` | Full table of AI patches: files changed, SHA, truncation flags |
| `/branches` | List all `godot-ai/*` branches with dates |
| `/checkout <branch>` | Switch to a godot-ai branch |
| `/save <name>` | Create a named git tag at the current state |
| `/undo` | Revert last patch — call multiple times for more levels |

---

## Fixing an error

**Option 1 — Paste manually:**
```
godot-ai> /fix
SCRIPT ERROR: Parse Error: Unexpected identifier "import" in class body.
   at: GDScript::reload (res://Autoloads/game_manager.gd:27)
END
```

**Option 2 — Launch Godot from the tool:**
```
godot-ai> /run C:\Users\you\Desktop\Godot_v4.6-stable_win64.exe
```
Errors are highlighted in red. Press **Enter** to trigger the fix offer.

**Option 3 — Watch Godot's log file:**
```
godot-ai> /watch
```
Run your game normally in the Godot editor. Errors auto-appear within ~1 second.

---

## Truncation protection

Three layers guard against the AI writing corrupt partial files:

- **Full file injection** — before any `/fix` or `/feature`, the complete content of every affected file is injected into the prompt
- **Pattern detection** — scans for placeholder comments like `# ... rest of the code ...` before writing
- **Size ratio check** — flags output less than 50% of the original file size as suspected silent truncation

If a file is flagged, writing defaults to **Skip**. Use `/retry` to re-run with an even stricter prompt.

---

## Git safety net

On startup the tool automatically:
1. Runs `git init` if the project has no repo
2. Creates a timestamped session branch: `godot-ai/2026-05-09_14-30-00`
3. Commits every accepted patch to this branch

```bash
git checkout main                          # abandon all AI changes
git diff main godot-ai/2026-05-09_...     # see everything the AI changed
```

---

## Troubleshooting

**`Connection refused`**  
Make sure Ollama is running with `OLLAMA_HOST=0.0.0.0 ollama serve` and port 11434 is open.

**`model not found`**  
SSH into the server and run `ollama pull qwen2.5-coder:14b`.

**Index seems stale**  
Run `/index`. To rebuild from zero delete `~/.godot-ai-dev/db` and restart.

**`pip install` fails on Mac ("externally managed environment")**
```bash
pip3 install --break-system-packages -r requirements.txt
# OR
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

**The AI keeps truncating files**  
Use `/retry` — it re-runs with a stricter prompt that includes the full file. Or switch models with `/models`.

---

## License

MIT
