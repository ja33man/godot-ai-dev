# 🎮 GodotAI Dev

A local, free, codebase-aware AI assistant for **Godot 4** game development.  
**No cloud. No API keys. No subscription.**

---

## How it works

```
Your PC / Mac (Windows or Mac)          Remote server (Linux + GPU)
──────────────────────────────          ──────────────────────────
  Godot (your game)                       Ollama  ← AI models live here
  godot_ai.py  ──── LAN ────────────→    192.168.2.49:11434
  your .gd files
  ChromaDB index (local)
  git repo (local)
```

- `godot_ai.py` runs **on your PC/Mac**, next to your Godot project
- Ollama runs on the **remote server** and does the GPU work
- Your code never leaves your LAN

---

## Requirements

| What | Where |
|------|-------|
| Python 3.10+ | Your PC / Mac |
| Godot 4 project | Your PC / Mac |
| git (optional but recommended) | Your PC / Mac |
| Ollama | Remote Linux server with GPU |

### Models to pull on the server

```bash
ollama pull nomic-embed-text      # ~274 MB — indexes your codebase
ollama pull qwen2.5-coder:14b     # ~9 GB   — writes and fixes GDScript
ollama pull qwen3.5:2b            # ~2.7 GB — fast chat answers
```

---

## Installation (on your PC / Mac)

```bash
git clone https://github.com/yourname/godot-ai-dev.git
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

Dependencies:

| Package | Purpose |
|---------|---------|
| `ollama` | Talks to the Ollama server over the network |
| `chromadb` | Local vector database for codebase search |
| `rich` | Terminal UI (colors, tables, panels, progress bars) |
| `watchdog` | Watches your project folder, re-indexes on save |
| `prompt_toolkit` | Reliable multiline paste input |

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

### What IP to use

| Setup | `--host` value |
|-------|---------------|
| Ollama on same machine | `http://localhost:11434` |
| Ollama on LAN server | `http://192.168.x.x:11434` |
| Ollama on VPN | `http://<vpn-ip>:11434` |

> Find your server's LAN IP: `ip addr show` (Linux) → look for `192.168.x.x`

### No arguments — it will ask

```
python godot_ai.py
📁 Godot project path: C:\Users\you\Documents\MyGame
🖥  Ollama host [http://localhost:11434]: http://192.168.2.49:11434
```

### Other flags

```bash
python godot_ai.py ./mygame --host http://... --no-git   # disable git safety net
python godot_ai.py ./mygame --host http://... --model qwen2.5-coder:32b
```

---

## Make sure Ollama accepts remote connections

On your server, start Ollama like this:
```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```
Without `0.0.0.0`, Ollama only accepts connections from `localhost` and your PC will be refused.

---

## First run

The tool scans every `.gd`, `.tscn`, `.tres`, `.gdshader` file, splits them into chunks, and embeds them with `nomic-embed-text`. The index is saved at `~/.godot-ai-dev/db` and loads instantly on future runs.

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
| `/run C:/path/godot.exe` | Same but specify the Godot executable path |
| `/watch` | Watch Godot's log file when Godot is open separately — errors auto-appear here |
| `/last` | Show the last captured error batch and offer to fix it |

### Git safety net

| Command | What it does |
|---------|-------------|
| `/status` | `git status` |
| `/diff` | Changes since session started |
| `/log` | Commits made this session |
| `/history` | Full table of AI patches: files changed, SHA, truncation flags |
| `/branches` | List all `godot-ai/*` branches with dates |
| `/checkout <branch>` | Switch to a godot-ai branch (guards against dirty tree) |
| `/save <name>` | Create a named git tag at the current state |
| `/undo` | Revert last patch — call multiple times for more levels |

---

## Fixing an error

**Option 1 — Paste manually:**
```
godot-ai> /fix
(paste the Godot error/log — type END on its own line when done)

SCRIPT ERROR: Parse Error: Unexpected identifier "import" in class body.
   at: GDScript::reload (res://Autoloads/game_manager.gd:27)
END
```

**Option 2 — Launch Godot from the tool:**
```
godot-ai> /run C:\Users\you\Desktop\Godot_v4.6-stable_win64.exe
```
Errors appear highlighted in red. When the burst of errors ends, a notification prints. Press **Enter** to trigger the fix offer.

**Option 3 — Watch Godot's log file:**
```
godot-ai> /watch
```
Run your game normally in the Godot editor. Errors auto-appear here within ~1 second.

---

## Adding a feature

```
godot-ai> /feature
(describe the feature — type END on its own line when done)

Add a double-jump. Second jump = 70% of first jump force.
Play "jump2" animation if it exists.
END
```

The AI reads your existing player code first, writes a plan, then generates complete GDScript.

---

## Asking questions

```
godot-ai> /chat
(ask your question — type END on its own line when done)

What is the difference between CharacterBody2D and RigidBody2D?
When should I use each one?
END
```

Short questions can be inline:
```
godot-ai> /chat What does @onready do?
```

---

## Truncation protection (how files stay safe)

This is the main safety system against the AI writing corrupt partial files.

**Layer 1 — Full file injection**: Before any `/fix` or `/feature`, the **complete current content** of every affected file is injected into the prompt. The model cannot say "I don't know the rest" — it's right there.

**Layer 2 — Pattern detection**: Before writing, the tool scans for placeholder comments like `# ... rest of the code ...`, `# existing code here`, `# [unchanged]` etc. If found, it warns and defaults to **Skip**.

**Layer 3 — Size ratio check**: If the AI output is less than 50% of the original file's byte count, it is flagged as suspected silent truncation — even if there are no visible markers.

If a file is flagged, you see:
```
⚠  TRUNCATION DETECTED in player.gd:
    # ... rest of the _physics_process function ...
  The AI output appears incomplete. Writing this would corrupt it.
  Skip player.gd (recommended)? [Y/n]:
```

Skipping is the default. You can then use `/retry` which re-asks with an even more forceful prompt.

---

## Git safety net

On startup, the tool automatically:
1. Runs `git init` if the project has no repo
2. Creates a timestamped session branch: `godot-ai/2026-05-09_14-30-00`
3. Commits every accepted patch to this branch

This means you can always:
```bash
git checkout main                          # abandon all AI changes
git diff main godot-ai/2026-05-09_14-30-00  # see everything the AI changed
```

`/undo` reverts one patch at a time. `/history` shows a table of every patch this session with timestamps, files changed, and whether any truncation warning was accepted.

---

## Model routing

The tool auto-selects the right model:

| Trigger | Model used |
|---------|-----------|
| `/fix`, `/feature` | Always the code model (`qwen2.5-coder:14b`) |
| `/chat` with code keywords (`error`, `func`, `node`, `signal`…) | Code model |
| `/chat` with general questions | Chat model (`qwen3.5:2b` — fast) |

Use `/models` to see all available models and switch them interactively.

Every response shows which model answered: `🤖 qwen2.5-coder:14b`

---

## Model swap delay

Your GPU has ~11GB VRAM. Both `qwen2.5-coder:14b` (~9GB) and `qwen3.5:2b` (~2.7GB) can't be in VRAM at the same time. When switching, Ollama unloads one and loads the other — expect a **5–15 second delay** on the first call after a switch. A spinner shows during this time. Subsequent calls to the same model are fast.

---

## Troubleshooting

**`Connection refused` or `Failed to connect`**
- Is Ollama running on the server? (`OLLAMA_HOST=0.0.0.0 ollama serve`)
- Is the IP correct? (`ping 192.168.2.49`)
- Is port 11434 open? (`sudo ufw allow 11434` on the server)

**`model not found`**
```bash
# SSH into the server and run:
ollama pull qwen2.5-coder:14b
```

**`/last` says "No errors captured" but I see errors in the terminal**
The errors were printed but not yet queued. Press bare **Enter** at the prompt — this drains the error queue. Or use `/watch` which catches errors from the log file even without `/run`.

**Index seems stale**
Run `/index`. To rebuild from zero: delete `~/.godot-ai-dev/db` (Mac/Linux) or `C:\Users\you\.godot-ai-dev\db` (Windows) and restart.

**`python` not found on Mac**
Use `python3` instead.

**`pip install` fails on Mac ("externally managed environment")**
```bash
pip3 install --break-system-packages -r requirements.txt
# OR use a virtual environment:
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

**The AI keeps truncating files even with the warnings**
1. Use `/retry` — it re-runs with an even stricter prompt that includes the full file
2. Try a different model: `/models` → switch code model to one that follows instructions better
3. Use `/chat` to ask the AI to rewrite just one specific function, then manually merge
