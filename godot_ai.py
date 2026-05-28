#!/usr/bin/env python3
"""
GodotAI Dev
────────────────────────────────────────────────────────────
Codebase-aware AI assistant for Godot 4 game development.
Runs locally against your own Ollama server — no cloud, no API keys.

Usage:
  python godot_ai.py [project_path] [--host URL] [--no-git]
  OLLAMA_HOST=http://192.168.2.49:11434 python godot_ai.py ~/games/mygame

Interrupt cheatsheet:
  Ctrl+C during AI response  → stop generation, keep partial, return to prompt
  Ctrl+C at godot-ai> prompt → exit the tool
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─── Dependency check ────────────────────────────────────────────────────────

_missing: list[str] = []
for _pkg, _imp in [
    ("chromadb",       "chromadb"),
    ("ollama",         "ollama"),
    ("rich",           "rich"),
    ("watchdog",       "watchdog"),
    ("prompt_toolkit", "prompt_toolkit"),
]:
    try:
        __import__(_imp)
    except ImportError:
        _missing.append(_pkg)

if _missing:
    print(f"Missing packages: {', '.join(_missing)}")
    print(f"Run: pip install {' '.join(_missing)}")
    sys.exit(1)

import chromadb
import ollama as _ollama_lib
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document as _PTDoc


# ─── @ file mention completer ─────────────────────────────────────────────────

class AtFileCompleter(Completer):
    """
    Triggers on '@' — fuzzy-matches against indexed project files.
    Works like VS Code's @file mention: type @play → Player/player.gd etc.
    """
    def __init__(self) -> None:
        self.files: list[str] = []

    def refresh(self, files: list[str]) -> None:
        self.files = [f.replace("\\", "/") for f in files]

    def get_completions(self, document: _PTDoc, complete_event):
        text = document.text_before_cursor
        at   = text.rfind("@")
        if at == -1:
            return
        query = text[at + 1:].lower()
        for f in self.files:
            name = f.split("/")[-1].lower()          # match on filename
            full = f.lower()
            if query in name or query in full:
                yield Completion(
                    f,
                    start_position=-len(text[at + 1:]),
                    display=f"@{f}",
                    display_meta=f.split("/")[-1],
                )


def _parse_at_mentions(text: str) -> tuple[str, list[str]]:
    """
    Extract @path/to/file.gd mentions from user input.
    Returns (cleaned_text, [list_of_file_paths]).
    Paths keep the @ stripped.
    """
    found: list[str] = []
    def _repl(m: re.Match) -> str:
        found.append(m.group(1))
        return m.group(1)    # leave the path in the text for context
    clean = re.sub(r'@([\w./\\\-]+)', _repl, text)
    return clean, found

console = Console()

# ─── Config ──────────────────────────────────────────────────────────────────

class Config:
    OLLAMA_HOST: str   = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    EMBED_MODEL: str   = "nomic-embed-text"
    CODE_MODEL:  str   = "qwen2.5-coder:14b"
    CHAT_MODEL:  str   = "qwen3.5:2b"
    CHROMA_PATH: str   = os.path.expanduser("~/.godot-ai-dev/db")
    MAX_CHUNKS:  int   = 6     # kept low — project tree + full-file injection
                               # already consume significant context
    # Minimum cosine similarity to include a chunk in context (0–1).
    # Chunks below this are semantically too distant to help.
    MIN_SCORE:   float = 0.30
    # !! IMPORTANT — set this to match your model's actual context window.
    # 8192 is far too small once you add a project tree + full file content.
    # Most modern models (qwen2.5-coder:14b, deepseek-coder-v2) support 32768.
    # If you use a tiny model (2b), lower this to 16384.
    CTX_WINDOW:  int   = 32768
    # Token budget reserved for the AI's OUTPUT. The remainder is available
    # for the system prompt + history + injected context.
    CTX_OUTPUT_RESERVE: int = 4096
    TEMPERATURE: float = 0.1
    HISTORY_LEN: int   = 6    # fewer pairs — context is now used by file injection

    # Text files: embedded + semantically searched
    GODOT_TEXT_EXTS: frozenset[str] = frozenset({
        ".gd", ".gdshader", ".gdshaderinc",
        ".tscn", ".tres", ".godot", ".cfg",
        ".json", ".txt", ".csv",
    })

    # Binary assets: appear in project tree only, never embedded
    ASSET_EXTS: frozenset[str] = frozenset({
        ".png", ".jpg", ".jpeg", ".webp", ".svg",
        ".wav", ".ogg", ".mp3",
        ".glb", ".dae", ".fbx", ".obj",
        ".ttf", ".otf", ".import",
    })

    SKIP_DIRS: frozenset[str] = frozenset({
        ".godot", ".git", "addons", "__pycache__", "node_modules",
    })

    CODE_SIGNALS: frozenset[str] = frozenset({
        "func", "class", "script", "shader", "error", "crash",
        "null", "fix", "bug", "node", "scene", "signal", "export",
        "ready", "process", "physics", "collision", "animation",
        "tween", "await", "gdscript", "rewrite", "refactor",
        "implement", "create", "modify", "edit", "add", "dash",
        "jump", "inventory", "player", "enemy", "spawn", "ui",
        "hud", "save", "load", "input", "camera", "network",
        "multiplayer", "rpc", "peer", "host", "client", "server",
    })


# ─── Persistent settings ─────────────────────────────────────────────────────

_SETTINGS_PATH = Path(os.path.expanduser("~/.godot-ai-dev/settings.json"))


def _load_settings() -> dict:
    """Load persistent user settings (godot exe path, etc.)."""
    try:
        return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_settings(data: dict) -> None:
    """Persist settings to disk. Silently ignores write errors."""
    try:
        _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SETTINGS_PATH.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


# ─── System prompts ──────────────────────────────────────────────────────────

SYSTEM_GODOT = """\
You are an expert Godot 4 software engineer.

════════════════════════════════════════════════════════════════
  FILE EDIT FORMATS  —  use EXACTLY one of these three:
════════════════════════════════════════════════════════════════

① SEARCH/REPLACE  ← USE THIS for any .gd / .tscn / .tres change
  Works for 1-line fixes up to ~30 lines. Models use this format
  well because it comes from git conflict notation.

  path/to/file.gd
  <<<<<<< SEARCH
  exact original lines — copy verbatim from the file shown below
  =======
  replacement lines
  >>>>>>> REPLACE

  Rules:
  • The SEARCH block must be an exact substring of the file.
  • Include enough context lines (2–3) so the match is unambiguous.
  • Multiple SEARCH/REPLACE blocks are allowed for the same file.
  • If more than ~30% of the file changes, use EDIT_FILE instead.

② SET_CONFIG  ← USE THIS for .godot / .cfg / .tres key-value changes
  Python handles the edit directly — no text matching needed.
  Handles version bumps, window size, display settings, autoloads, etc.

  SET_CONFIG: project.godot
  SECTION: application
  KEY: config/version
  VALUE: "2-beta"

  SET_CONFIG: project.godot
  SECTION: display
  KEY: window/size/viewport_width
  VALUE: 1920

  Multiple SET_CONFIG blocks are allowed per response.
  VALUE must be the raw value as it appears in the file (no quotes
  needed around numbers; strings need their own double-quotes).

③ EDIT_FILE / NEW_FILE  ← for new files or large rewrites only
  Include 100% of the file — no ellipsis, no placeholders.
  ⚠  Partial output CORRUPTS the project. Zero exceptions.

  EDIT_FILE: path/to/file.gd
  ```gdscript
  # complete file here — every line, no shortcuts
  ```

════════════════════════════════════════════════════════════════
  DECISION GUIDE:
    Changing a setting in project.godot?   → SET_CONFIG
    Fixing one function in a .gd file?     → SEARCH/REPLACE
    Adding 10 lines to a scene file?       → SEARCH/REPLACE
    Creating a new script from scratch?    → NEW_FILE
    Rewriting >30% of a large script?      → EDIT_FILE
════════════════════════════════════════════════════════════════

Godot 4 knowledge:
- GDScript 2.0: typed vars (@export, @onready), signals, await, @rpc
- Nodes: CharacterBody2D/3D, RigidBody2D/3D, Area2D/3D, Control
- Multiplayer: ENetMultiplayerPeer, MultiplayerAPI, @rpc
- Animation: AnimationPlayer, AnimationTree, Tween
- Lifecycle: _ready, _process(delta), _physics_process(delta)

Output rules:
1. Paths are relative to the project root — never use res:// prefix.
2. Use Godot 4 API only — never Godot 3 (no yield, no connect(name,obj,method)).
3. Typed GDScript everywhere: var speed: float = 200.0
4. Only use file paths that appear in the PROJECT TREE below.
5. End every response with:
   ## Setup Steps
   List every manual action in the Godot editor (node names, property
   values, autoload registration, firewall rules, etc.).
"""

SYSTEM_FIX = SYSTEM_GODOT + """
Diagnosis:
1. Root Cause — one sentence.
2. Which lines / keys are wrong.
3. Fix using the appropriate format above.
"""

SYSTEM_FEATURE = SYSTEM_GODOT + """
Implementation:
1. 2–4 bullet plan.
2. Use SEARCH/REPLACE for additions to existing files.
3. Use SET_CONFIG for any project.godot changes.
4. Use NEW_FILE for new scripts.
5. End with ## Setup Steps covering every editor and OS action.
"""


# ─── UTF-8 helpers ───────────────────────────────────────────────────────────

def read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_utf8(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")


# ─── Conversation logger ──────────────────────────────────────────────────────

class ConversationLogger:
    """
    Writes every session to a human-readable log file.

    Location: ~/.godot-ai-dev/logs/<project_name>/<YYYY-MM-DD_HH-MM-SS>.log

    Format is plain text with timestamps so you can grep through it later.
    Nothing is ever truncated in the log — it is the permanent record.
    """

    _DIVIDER   = "─" * 72
    _SEPARATOR = "═" * 72

    def __init__(self, project_path: str) -> None:
        project_name = Path(project_path).name
        log_dir = Path(os.path.expanduser("~/.godot-ai-dev/logs")) / project_name
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = log_dir / f"{stamp}.log"
        self._write_header(project_path)

    def _write_header(self, project_path: str) -> None:
        self._append(
            f"{self._SEPARATOR}\n"
            f"GodotAI Dev — Session Log\n"
            f"Project : {project_path}\n"
            f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Models  : code={Config.CODE_MODEL}  chat={Config.CHAT_MODEL}\n"
            f"Embeds  : {Config.EMBED_MODEL}\n"
            f"{self._SEPARATOR}\n"
        )

    def _append(self, text: str) -> None:
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(text)
        except Exception:
            pass   # logging must never crash the tool

    def _ts(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    # ── public log methods ────────────────────────────────────────────────────

    def log_user(self, cmd_type: str, content: str) -> None:
        """Log a user command (fix/feature/chat/etc.) and its input."""
        self._append(
            f"\n[{self._ts()}] USER  /{cmd_type}\n"
            f"{self._DIVIDER}\n"
            f"{content}\n"
        )

    def log_ai(self, model: str, response: str, interrupted: bool = False) -> None:
        """Log the full AI response."""
        flag = "  (interrupted)" if interrupted else ""
        self._append(
            f"\n[{self._ts()}] AI  {model}{flag}\n"
            f"{self._DIVIDER}\n"
            f"{response}\n"
        )

    def log_files(self, applied: list[str], skipped: list[str],
                  git_sha: str = "") -> None:
        """Log which files were written and which were skipped."""
        lines = [f"\n[{self._ts()}] FILES\n{self._DIVIDER}"]
        for f in applied:
            lines.append(f"  APPLIED  {f}")
        for f in skipped:
            lines.append(f"  SKIPPED  {f}")
        if git_sha:
            lines.append(f"  git SHA  {git_sha}")
        self._append("\n".join(lines) + "\n")

    def log_godot_errors(self, errors: str) -> None:
        """Log errors captured from Godot /run or /watch."""
        self._append(
            f"\n[{self._ts()}] GODOT ERRORS\n"
            f"{self._DIVIDER}\n"
            f"{errors}\n"
        )

    def log_compression(self, n_compressed: int, summary: str) -> None:
        """Log when history is compressed to save context space."""
        self._append(
            f"\n[{self._ts()}] HISTORY COMPRESSED ({n_compressed} messages → summary)\n"
            f"{self._DIVIDER}\n"
            f"{summary}\n"
        )

    def log_event(self, label: str, detail: str = "") -> None:
        """Log a miscellaneous event (git ops, /undo, /clear, etc.)."""
        self._append(
            f"\n[{self._ts()}] {label}"
            + (f"\n{self._DIVIDER}\n{detail}" if detail else "")
            + "\n"
        )

    @staticmethod
    def list_logs(project_path: str, n: int = 10) -> list[Path]:
        """Return the N most recent log files for this project."""
        project_name = Path(project_path).name
        log_dir = Path(os.path.expanduser("~/.godot-ai-dev/logs")) / project_name
        if not log_dir.exists():
            return []
        files = sorted(log_dir.glob("*.log"), reverse=True)
        return files[:n]


# ─── Git manager ─────────────────────────────────────────────────────────────

class GitManager:
    """
    Lightweight git safety net.

    On startup:
      - If no repo, runs git init + initial commit of the current project state
      - Creates a timestamped session branch: godot-ai/YYYY-MM-DD_HH-MM-SS
      - Every applied patch is committed to this branch

    Users can always:
      git checkout main                  # abandon all AI changes
      git diff main godot-ai/<stamp>     # review everything the AI did
    """

    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root    = root
        self.enabled = enabled
        self.branch  = ""
        # Multi-level undo: stack of HEAD SHAs captured before each patch
        self._patch_stack: list[str] = []
        # Session patch history for /history command
        self.patch_history: list[dict] = []

        if not enabled:
            return
        if not self._git_available():
            console.print(
                "[yellow]⚠  git not found in PATH — safety net disabled.[/yellow]"
            )
            self.enabled = False
            return

        self._ensure_repo()
        self.branch = self._create_session_branch()

    @staticmethod
    def _git_available() -> bool:
        try:
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def _run(self, *args: str, check: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            capture_output=True, text=True, encoding="utf-8", check=check
        )

    def _ensure_repo(self) -> None:
        result = self._run("rev-parse", "--is-inside-work-tree")
        if result.returncode != 0:
            console.print("[cyan]git init[/cyan] — initialising repository…")
            self._run("init", check=True)
            gi = self.root / ".gitignore"
            if not gi.exists():
                write_utf8(gi, ".godot/\n*.import\n")
            self._run("add", ".")
            self._run("commit", "-m", "Initial commit (godot-ai)", "--allow-empty")
        else:
            # Make sure there's at least one commit
            if self._run("log", "--oneline", "-1").returncode != 0:
                self._run("add", ".")
                self._run("commit", "-m", "Initial commit (godot-ai)", "--allow-empty")

    def _create_session_branch(self) -> str:
        stamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name   = f"godot-ai/{stamp}"
        result = self._run("checkout", "-b", name)
        if result.returncode == 0:
            console.print(f"[green]✓ git branch:[/green] [cyan]{name}[/cyan]")
        else:
            console.print(f"[yellow]git: {result.stderr.strip()}[/yellow]")
        return name

    # ── public ───────────────────────────────────────────────────────────────

    def snapshot_before_patch(self) -> Optional[str]:
        """Push current HEAD SHA onto the undo stack before applying a patch."""
        if not self.enabled:
            return None
        result = self._run("rev-parse", "HEAD")
        sha = result.stdout.strip() if result.returncode == 0 else None
        if sha:
            self._patch_stack.append(sha)
        return sha

    def commit_patch(self, message: str, op_type: str = "patch",
                     files: Optional[list[str]] = None,
                     had_truncation: bool = False) -> bool:
        if not self.enabled:
            return False
        self._run("add", ".")
        result = self._run("commit", "-m", message, "--allow-empty")
        if result.returncode == 0:
            sha = self._run("rev-parse", "--short", "HEAD").stdout.strip()
            console.print(f"[dim]  git: committed {sha}[/dim]")
            self.patch_history.append({
                "sha":            sha,
                "timestamp":      datetime.now().strftime("%H:%M:%S"),
                "op_type":        op_type,
                "files":          list(files or []),
                "had_truncation": had_truncation,
                "message":        message,
            })
            return True
        return False

    def undo_last_patch(self) -> bool:
        if not self.enabled:
            console.print("[yellow]git safety net is disabled.[/yellow]")
            return False
        if not self._patch_stack:
            console.print("[yellow]Nothing to undo — patch stack is empty.[/yellow]")
            return False
        sha = self._patch_stack.pop()
        result = self._run("reset", "--hard", sha)
        if result.returncode == 0:
            console.print(f"[green]✓ Reverted to {sha[:8]}[/green]")
            if self.patch_history:
                self.patch_history.pop()
            remaining = len(self._patch_stack)
            if remaining:
                console.print(f"[dim]  {remaining} more undo level(s) available.[/dim]")
            return True
        console.print(f"[red]git reset failed:[/red] {result.stderr.strip()}")
        # Put the SHA back since the reset failed
        self._patch_stack.append(sha)
        return False

    def undo_levels_available(self) -> int:
        return len(self._patch_stack)

    def list_branches(self) -> list[dict]:
        """Return all godot-ai/* branches with their latest commit date."""
        result = self._run(
            "branch", "--list", "godot-ai/*",
            "--format=%(refname:short)|%(committerdate:format:%Y-%m-%d %H:%M)"
        )
        branches: list[dict] = []
        for line in result.stdout.strip().splitlines():
            if "|" in line:
                name, date = line.split("|", 1)
            else:
                name, date = line.strip(), ""
            if name.strip():
                branches.append({
                    "name":    name.strip(),
                    "date":    date.strip(),
                    "current": name.strip() == self.branch,
                })
        return branches

    def checkout_branch(self, name: str,
                        session: Optional[object] = None) -> bool:
        """Checkout a branch. If the tree is dirty, offer to commit or stash first."""
        status = self._run("status", "--porcelain")
        if status.stdout.strip():
            console.print(
                f"[yellow]⚠  Uncommitted changes:[/yellow]\n"
                f"[dim]{status.stdout.strip()[:300]}[/dim]\n"
            )
            # Offer options
            prompt_fn = (session.prompt if session else input)
            try:
                choice = prompt_fn(
                    "  Handle changes: [commit / stash / cancel]: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "cancel"

            if choice == "commit":
                try:
                    msg = prompt_fn(
                        "  Commit message (Enter = auto): "
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    msg = ""
                if not msg:
                    msg = f"[checkpoint] before checkout {name}"
                self._run("add", ".")
                self._run("commit", "-m", msg)
                console.print("[green]✓ Changes committed.[/green]")
            elif choice == "stash":
                self._run("stash", "push", "-m", f"godot-ai stash before {name}")
                console.print(
                    "[green]✓ Changes stashed.[/green] "
                    "[dim](restore later with: git stash pop)[/dim]"
                )
            else:
                console.print("[dim]Checkout cancelled.[/dim]")
                return False

        result = self._run("checkout", name)
        if result.returncode == 0:
            self.branch = name
            self._patch_stack.clear()
            console.print(f"[green]✓ Switched to:[/green] [cyan]{name}[/cyan]")
            return True
        console.print(f"[red]git checkout failed:[/red] {result.stderr.strip()}")
        return False

    def patch_history_table(self) -> "Table":
        """Rich table of all patches applied this session."""
        t = Table(title=f"Session Patch History — {self.branch}", border_style="cyan")
        t.add_column("Time",  style="dim",    width=8)
        t.add_column("SHA",   style="cyan",   width=8)
        t.add_column("Op",    style="yellow", width=9)
        t.add_column("Files", style="white")
        t.add_column("⚠",    style="red",    width=3)
        for p in self.patch_history:
            trunc_flag = "✗" if p["had_truncation"] else ""
            t.add_row(
                p["timestamp"],
                p["sha"],
                p["op_type"],
                ", ".join(p["files"])[:70],
                trunc_flag,
            )
        if not self.patch_history:
            t.add_row("[dim]—[/dim]", "[dim]—[/dim]", "[dim]—[/dim]",
                      "[dim]no patches applied yet[/dim]", "")
        return t

    def diff(self) -> str:
        if not self.enabled:
            return "(git disabled)"
        result = self._run("diff", "HEAD~1", "HEAD", "--stat")
        return result.stdout or "(nothing committed yet in this session)"

    def status(self) -> str:
        if not self.enabled:
            return "(git disabled)"
        return self._run("status", "--short").stdout or "(clean)"

    def session_log(self) -> str:
        if not self.enabled:
            return "(git disabled)"
        result = self._run("log", "--oneline", "-20")
        return result.stdout.strip() or "(no commits yet)"

    def create_checkpoint(self, name: str) -> bool:
        """Create a named git tag: godot-ai/checkpoint-<name>."""
        if not self.enabled:
            console.print("[yellow]git safety net is disabled.[/yellow]")
            return False
        tag = f"godot-ai/checkpoint-{name}"
        result = self._run("tag", tag)
        if result.returncode == 0:
            sha = self._run("rev-parse", "--short", "HEAD").stdout.strip()
            console.print(
                f"[green]✓ Checkpoint created:[/green] [cyan]{tag}[/cyan] "
                f"[dim]@ {sha}[/dim]"
            )
            return True
        console.print(f"[red]git tag failed:[/red] {result.stderr.strip()}")
        return False


# ─── Indexer ─────────────────────────────────────────────────────────────────

class CodebaseIndexer:
    def __init__(self, project_path: str, ollama_host: str) -> None:
        self.root        = Path(project_path).resolve()
        self.ollama_host = ollama_host

        os.makedirs(Config.CHROMA_PATH, exist_ok=True)
        self.chroma = chromadb.PersistentClient(path=Config.CHROMA_PATH)
        self.col    = self.chroma.get_or_create_collection(
            "godot_code", metadata={"hnsw:space": "cosine"}
        )
        self.client = _ollama_lib.Client(host=ollama_host)

        self.project_tree: str  = ""
        self._reindex_lock      = threading.Lock()
        self._pending: list[Path] = []

    # ── embed ────────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        return self.client.embeddings(
            model=Config.EMBED_MODEL, prompt=text[:4000]
        )["embedding"]

    # ── project tree ─────────────────────────────────────────────────────────

    def build_project_tree(self) -> str:
        """
        Full asset-aware manifest of the project.
        Injected into every prompt so the AI knows ALL files (code + art + audio).
        """
        lines: list[str] = []
        all_exts = Config.GODOT_TEXT_EXTS | Config.ASSET_EXTS

        for path in sorted(self.root.rglob("*")):
            if any(skip in path.parts for skip in Config.SKIP_DIRS):
                continue
            rel    = path.relative_to(self.root)
            indent = "  " * (len(rel.parts) - 1)
            if path.is_dir():
                lines.append(f"{indent}📁 {path.name}/")
            elif path.suffix in all_exts or path.suffix == "":
                tag = "📄" if path.suffix in Config.GODOT_TEXT_EXTS else "🖼"
                lines.append(f"{indent}{tag} {path.name}")

        return "\n".join(lines)

    # ── chunking ─────────────────────────────────────────────────────────────

    def _chunk_gd(self, content: str, rel: str) -> list[dict]:
        chunks: list[dict] = []
        buf: list[str]     = []
        start = 0
        for i, line in enumerate(content.splitlines()):
            boundary = (
                line.startswith("func ")
                or line.startswith("static func ")
                or line.startswith("class ")
            )
            if boundary and buf and i > 0:
                text = "\n".join(buf)
                if text.strip():
                    chunks.append({"text": f"# {rel}\n{text}", "file": rel, "ln": start})
                buf   = [line]
                start = i
            else:
                buf.append(line)
        if buf:
            text = "\n".join(buf)
            if text.strip():
                chunks.append({"text": f"# {rel}\n{text}", "file": rel, "ln": start})
        return chunks

    def _chunk_tscn(self, content: str, rel: str) -> list[dict]:
        """Extract node/connection structure — gives AI scene hierarchy."""
        node_lines = [
            l for l in content.splitlines()
            if l.startswith("[node ") or l.startswith("[connection ")
            or l.startswith("[gd_scene ") or l.startswith("script =")
        ]
        if node_lines:
            summary = f"# SCENE STRUCTURE: {rel}\n" + "\n".join(node_lines[:100])
            return [{"text": summary, "file": rel, "ln": 0}]
        return self._chunk_generic(content, rel)

    def _chunk_generic(self, content: str, rel: str,
                        size: int = 2000, overlap: int = 200) -> list[dict]:
        chunks: list[dict] = []
        for i in range(0, len(content), size - overlap):
            chunk = content[i: i + size]
            if chunk.strip():
                chunks.append({"text": f"# {rel}\n{chunk}", "file": rel, "ln": i})
        return chunks

    def _chunks_for(self, path: Path) -> list[dict]:
        try:
            content = read_utf8(path)
        except Exception:
            return []
        rel = str(path.relative_to(self.root))
        if path.suffix == ".gd":
            return self._chunk_gd(content, rel)
        if path.suffix == ".tscn":
            return self._chunk_tscn(content, rel)
        return self._chunk_generic(content, rel)

    # ── index ────────────────────────────────────────────────────────────────

    def index_file(self, path: Path) -> int:
        if path.suffix not in Config.GODOT_TEXT_EXTS:
            return 0
        if any(skip in path.parts for skip in Config.SKIP_DIRS):
            return 0
        rel = str(path.relative_to(self.root))
        try:
            old = self.col.get(where={"file": rel})
            if old["ids"]:
                self.col.delete(ids=old["ids"])
        except Exception:
            pass
        chunks = self._chunks_for(path)
        added  = 0
        for i, c in enumerate(chunks):
            uid = hashlib.md5(f"{rel}:{i}:{c['text'][:80]}".encode()).hexdigest()
            try:
                emb = self._embed(c["text"])
                self.col.add(
                    ids=[uid], embeddings=[emb],
                    documents=[c["text"]],
                    metadatas=[{"file": c["file"], "ln": c["ln"]}],
                )
                added += 1
            except Exception:
                pass
        return added

    def index_project(self) -> int:
        files = [
            f for f in self.root.rglob("*")
            if f.is_file()
            and f.suffix in Config.GODOT_TEXT_EXTS
            and not any(skip in f.parts for skip in Config.SKIP_DIRS)
        ]
        total = 0
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as prog:
            task = prog.add_task("Indexing…", total=len(files))
            for f in files:
                n = self.index_file(f)
                total += n
                prog.update(task, advance=1,
                            description=f"[cyan]{f.name}[/cyan] ({n} chunks)")
        self.project_tree = self.build_project_tree()
        return total

    # ── search ───────────────────────────────────────────────────────────────

    def search(self, query: str, n: int = Config.MAX_CHUNKS) -> list[dict]:
        """
        Semantic search with score filtering.
        Chunks below MIN_SCORE are dropped to keep the context window clean.
        """
        if self.col.count() == 0:
            return []
        emb = self._embed(query)
        n   = min(n, self.col.count())
        res = self.col.query(query_embeddings=[emb], n_results=n)

        results = []
        for i, doc in enumerate(res["documents"][0]):
            score = 1.0 - res["distances"][0][i]
            if score >= Config.MIN_SCORE:
                results.append({
                    "text":  doc,
                    "file":  res["metadatas"][0][i]["file"],
                    "score": score,
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ── helpers ──────────────────────────────────────────────────────────────

    def read_file(self, rel: str) -> Optional[str]:
        p = self.root / rel
        return read_utf8(p) if p.exists() else None

    def list_indexed_files(self) -> list[str]:
        data = self.col.get()
        return sorted(set(m["file"] for m in data["metadatas"]))

    # ── deferred re-index queue (watcher-safe) ───────────────────────────────

    def queue_reindex(self, path: Path) -> None:
        with self._reindex_lock:
            if path not in self._pending:
                self._pending.append(path)

    def drain_reindex_queue(self) -> list[str]:
        """
        Process queued re-indexes. Called from the main REPL loop between
        prompts so it never races with Rich streaming output.
        """
        with self._reindex_lock:
            pending = list(self._pending)
            self._pending.clear()
        done = []
        for path in pending:
            try:
                self.index_file(path)
                done.append(path.name)
            except Exception:
                pass
        return done


# ─── Ollama model listing ────────────────────────────────────────────────────

def fetch_available_models(client: _ollama_lib.Client) -> list[str]:
    try:
        data = client.list()
        return [m["model"] for m in data.get("models", [])]
    except Exception:
        return []


# ─── Agent ───────────────────────────────────────────────────────────────────

_FILE_REF_RE = re.compile(r'[\w/.-]+\.(gd|tscn|tres|cfg|json|gdshader)')


class GodotAgent:
    def __init__(self, indexer: CodebaseIndexer,
                 logger: Optional["ConversationLogger"] = None) -> None:
        self.indexer = indexer
        self.client  = _ollama_lib.Client(host=indexer.ollama_host)
        self.history: list[dict] = []
        self.logger  = logger
        self._code_model: Optional[str] = None
        self._chat_model: Optional[str] = None

    @property
    def code_model(self) -> str:
        return self._code_model or Config.CODE_MODEL

    @property
    def chat_model(self) -> str:
        return self._chat_model or Config.CHAT_MODEL

    def _route_model(self, text: str) -> str:
        words = set(re.findall(r'\w+', text.lower()))
        return self.code_model if words & Config.CODE_SIGNALS else self.chat_model

    # ── history compression ───────────────────────────────────────────────────

    def _compress_history(self) -> None:
        """
        Summarise the oldest half of the conversation history using the fast
        chat model, then replace those messages with a single compact memory
        block. This frees context space without losing what was accomplished.

        Called automatically at 60% context utilisation by _stream().
        The summary is written to the log file so nothing is permanently lost.
        """
        if len(self.history) < 6:
            return   # not enough history to be worth compressing

        cutoff = len(self.history) // 2
        old_messages = self.history[:cutoff]
        self.history = self.history[cutoff:]

        # Build a compact conversation transcript for the summariser
        transcript = "\n".join(
            f"{m['role'].upper()}: {m['content'][:600]}"
            for m in old_messages
        )
        summary_prompt = (
            "Summarise this Godot game-dev AI session in 5–8 bullet points.\n"
            "Focus on: what errors were fixed, what features were implemented,\n"
            "which files were changed, and any important decisions made.\n"
            "Be specific (file names, function names, error types).\n\n"
            f"{transcript}"
        )

        summary = "(summary unavailable)"
        try:
            result = self.client.chat(
                model=self.chat_model,
                messages=[{"role": "user", "content": summary_prompt}],
                options={"temperature": 0.1, "num_ctx": 4096},
            )
            summary = result["message"]["content"]
        except Exception as exc:
            summary = f"(summarisation failed: {exc})"

        # Prepend as a system-level memory block
        self.history.insert(0, {
            "role":    "system",
            "content": f"COMPRESSED SESSION MEMORY (earlier messages summarised):\n{summary}",
        })

        console.print(
            f"[dim]🗜  History compressed: {cutoff} messages → 1 summary block "
            f"({len(self.history)} messages remain)[/dim]"
        )
        if self.logger:
            self.logger.log_compression(cutoff, summary)

    # ── context builder ──────────────────────────────────────────────────────

    def _context(self, query: str, extra_files: Optional[list[str]] = None) -> str:
        """
        Context injected into every prompt:
          1. Full project tree (code + assets)
          2. Semantically relevant chunks (scored, filtered)
          3. Full content of explicitly referenced files
        """
        parts: list[str] = []

        # 1. Project tree — always
        if self.indexer.project_tree:
            parts.append(f"# PROJECT TREE\n{self.indexer.project_tree}")

        # 2. Semantic chunks
        chunks = self.indexer.search(query)
        seen_files: list[str] = []
        for c in chunks:
            parts.append(c["text"])
            if c["file"] not in seen_files:
                seen_files.append(c["file"])

        if chunks:
            score_str = ", ".join(f"{c['file']}({c['score']:.2f})" for c in chunks[:4])
            console.print(f"[dim]  RAG: {score_str}[/dim]")
        else:
            console.print("[dim]  RAG: no relevant chunks found (index may need /index)[/dim]")

        # 3. Full file content for explicitly referenced files
        for rel in (extra_files or []):
            content = self.indexer.read_file(rel)
            if content:
                parts.append(f"# FULL FILE: {rel}\n{content}")

        return "\n\n---\n\n".join(parts)

    # ── stream ───────────────────────────────────────────────────────────────

    def _stream(self, user_prompt: str, system: str = SYSTEM_GODOT,
                model: Optional[str] = None) -> str:
        chosen   = model or self.code_model
        messages = [{"role": "system", "content": system}]
        messages += self.history[-(Config.HISTORY_LEN * 2):]
        messages.append({"role": "user", "content": user_prompt})

        # ── Context usage check ───────────────────────────────────────────────
        total_chars = sum(len(m["content"]) for m in messages)
        est_tokens  = total_chars // 4
        # Available budget = context window minus the output reserve
        budget      = Config.CTX_WINDOW - Config.CTX_OUTPUT_RESERVE
        ctx_pct     = est_tokens / budget

        if ctx_pct >= 0.85:
            console.print(
                f"[bold red]⚠  Context nearly full:[/bold red] "
                f"~{est_tokens:,} / {budget:,} tokens ({ctx_pct:.0%}). "
                "The model will likely truncate output. Use /clear to reset, "
                "or /chat to ask a question without file injection."
            )
        elif ctx_pct >= 0.55:
            # Compress before sending — rebuild messages after
            console.print(
                f"[yellow]Context at {ctx_pct:.0%} — compressing history…[/yellow]"
            )
            self._compress_history()
            # Rebuild with compressed history
            messages = [{"role": "system", "content": system}]
            messages += self.history[-(Config.HISTORY_LEN * 2):]
            messages.append({"role": "user", "content": user_prompt})
            new_tokens = sum(len(m["content"]) for m in messages) // 4
            console.print(
                f"[dim]  Context reduced: ~{est_tokens:,} → ~{new_tokens:,} tokens "
                f"(budget: {budget:,})[/dim]"
            )

        console.print(f"\n[dim]🤖 {chosen}  (Ctrl+C to interrupt)[/dim]")

        full        = ""
        interrupted = False

        try:
            stream = self.client.chat(
                model=chosen, messages=messages, stream=True,
                options={"temperature": Config.TEMPERATURE,
                         "num_ctx":    Config.CTX_WINDOW},
            )
            first_chunk = None
            with console.status(f"[dim]Loading {chosen}…[/dim]", spinner="dots"):
                try:
                    first_chunk = next(iter(stream))
                except StopIteration:
                    pass
                except KeyboardInterrupt:
                    interrupted = True

            if first_chunk and not interrupted:
                delta = first_chunk["message"]["content"]
                full += delta
                console.print(delta, end="", markup=False)

            if not interrupted:
                try:
                    for chunk in stream:
                        delta = chunk["message"]["content"]
                        full += delta
                        console.print(delta, end="", markup=False)
                except KeyboardInterrupt:
                    interrupted = True

        except KeyboardInterrupt:
            interrupted = True
        except Exception as exc:
            console.print(f"\n[red]Ollama error: {exc}[/red]")
            console.print("[dim]Check Ollama is running and the model is pulled.[/dim]")
            return ""

        console.print()
        if interrupted:
            console.print("[yellow]⚡ Generation interrupted.[/yellow]")

        if full:
            self.history.append({"role": "user",      "content": user_prompt})
            self.history.append({"role": "assistant",  "content": full})
            if self.logger:
                self.logger.log_ai(chosen, full, interrupted=interrupted)

        return full

    # ── error parsing ────────────────────────────────────────────────────────

    @staticmethod
    def _error_files(error: str) -> list[str]:
        return list(dict.fromkeys(
            rel for rel, _ in re.findall(r'res://([\w/._-]+\.gd):(\d+)', error)
        ))[:4]

    @staticmethod
    def _error_snippets(error: str, indexer: CodebaseIndexer) -> str:
        out = []
        for rel, ln_str in re.findall(r'res://([\w/._-]+\.gd):(\d+)', error)[:3]:
            content = indexer.read_file(rel)
            if not content:
                continue
            ln = int(ln_str)
            lines = content.splitlines()
            lo, hi = max(0, ln - 10), min(len(lines), ln + 10)
            body = "\n".join(
                f"{'→ ' if lo + idx + 1 == ln else '  '}{lo + idx + 1}: {line}"
                for idx, line in enumerate(lines[lo:hi])
            )
            out.append(f"# ERROR LOCATION {rel}:{ln}\n{body}")
        return "\n\n".join(out)

    # ── full-file injection ──────────────────────────────────────────────────

    def _inject_full_files(self, file_refs: list[str], label: str = "CURRENT FILE",
                           token_budget: int = 8000) -> str:
        """
        Read the current on-disk content of each referenced file and format it
        so the model has the COMPLETE content in front of it.

        token_budget: approximate maximum tokens to spend across ALL injected files.
        This prevents the injected context from blowing out the context window.
        Files are prioritised in order — earlier entries get more budget.

        This is the primary defence against truncation: the model cannot claim
        it doesn't know what the rest of the file contains when it's right there.
        """
        parts: list[str] = []
        chars_remaining = token_budget * 4   # rough chars → tokens conversion

        for rel in file_refs[:6]:
            if chars_remaining <= 400:
                console.print(
                    f"[dim]  (file injection budget exhausted — skipping {rel})[/dim]"
                )
                break
            content = self.indexer.read_file(rel)
            if not content:
                continue
            max_chars = min(len(content), chars_remaining - 200)
            trimmed   = len(content) > max_chars
            body      = content[:max_chars]
            tail      = (
                "\n# ... (file trimmed for context budget — STILL write the COMPLETE "
                "file in your output, using the full original as a reference)"
                if trimmed else ""
            )
            header_chars = len(rel) + 120
            parts.append(
                f"## {label}: {rel}\n"
                f"⚠️  YOU MUST OUTPUT THIS ENTIRE FILE with your changes applied.\n"
                f"```gdscript\n{body}{tail}\n```"
            )
            chars_remaining -= len(body) + header_chars

        return "\n\n".join(parts)

    # ── public API ───────────────────────────────────────────────────────────

    # Keywords that suggest a project.godot config change rather than a code error
    _CONFIG_SIGNALS = frozenset({
        "window", "resolution", "size", "4k", "fullscreen", "version",
        "startup", "launch", "display", "screen", "project settings",
        "viewport", "stretch", "vsync", "fps", "frame",
    })

    def _is_config_request(self, text: str) -> bool:
        words = set(re.findall(r'\w+', text.lower()))
        return bool(words & self._CONFIG_SIGNALS)

    # ── Step 1: Triage ────────────────────────────────────────────────────────

    def triage(self, query: str, hint_files: list[str] | None = None) -> list[str]:
        """
        Use the fast chat model to decide which files are relevant.
        Returns a list of relative file paths (max 6).

        This is cheap (~2s) and dramatically improves the quality of the
        main analysis step by filtering noise before loading full files.
        """
        rag_hits = self.indexer.search(query, n=12)
        rag_names = list(dict.fromkeys(c["file"] for c in rag_hits))

        # Hint files (from @mentions or error stack traces) go first
        candidates = list(dict.fromkeys((hint_files or []) + rag_names))[:20]

        tree = getattr(self.indexer, "project_tree", "") or ""
        tree_snippet = "\n".join(tree.splitlines()[:80])

        prompt = (
            f"Task:\n{query}\n\n"
            f"Candidate files (from search):\n" + "\n".join(candidates) + "\n\n"
            f"Project tree (partial):\n{tree_snippet}\n\n"
            "List the 2–5 files that MUST be read to complete this task.\n"
            "Output ONLY a JSON array of relative paths. Example:\n"
            '["Autoloads/game_manager.gd", "src/network_manager.gd"]\n'
            "If unsure, include the most likely files. No explanation."
        )
        try:
            result = self.client.chat(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_ctx": 4096},
            )
            raw = result["message"]["content"].strip()
            # Extract JSON array even if model adds surrounding text
            m = re.search(r'\[.*?\]', raw, re.DOTALL)
            if m:
                paths = json.loads(m.group(0))
                # Deduplicate, keep order, limit
                seen: set[str] = set()
                out: list[str] = []
                for p in paths:
                    p = p.strip().lstrip("/").replace("\\", "/")
                    if p and p not in seen:
                        seen.add(p)
                        out.append(p)
                return out[:6]
        except Exception:
            pass
        # Fallback: just use the top RAG hits
        return rag_names[:5]

    # ── Step 2a: Dependency tracer ────────────────────────────────────────────

    def trace_deps(self, file_path: str, depth: int = 1) -> list[str]:
        """
        Follow `extends`, `preload()`, and `load()` references one level deep.
        Returns additional relative paths to load for context.
        """
        if depth <= 0:
            return []
        full = self.indexer.read_file(file_path)
        if not full:
            return []
        deps: list[str] = []
        for pattern in (
            r'^extends\s+"([^"]+)"',
            r'(?:preload|load)\("([^"]+\.gd)"',
        ):
            for m in re.finditer(pattern, full, re.MULTILINE):
                ref = m.group(1)
                # Convert res:// to relative
                ref = re.sub(r'^res://', '', ref)
                if ref and ref not in deps:
                    deps.append(ref)
        return deps[:4]   # cap at 4 extra files

    # ── Step 2b: Load files in full ───────────────────────────────────────────

    def load_files(self, paths: list[str],
                   follow_deps: bool = True) -> dict[str, str]:
        """
        Read files from disk in full. Optionally follows extends/preload
        one level deep so the model understands class hierarchies.
        Returns {relative_path: content}.
        """
        result: dict[str, str] = {}
        work_queue = list(paths)
        seen:  set[str] = set()

        while work_queue:
            rel = work_queue.pop(0)
            if rel in seen:
                continue
            seen.add(rel)
            content = self.indexer.read_file(rel)
            if content:
                result[rel] = content
                if follow_deps and rel.endswith(".gd") and len(result) < 10:
                    for dep in self.trace_deps(rel, depth=1):
                        if dep not in seen:
                            work_queue.append(dep)

        return result

    # ── Step 4: Verify SEARCH blocks ─────────────────────────────────────────

    def verify_edits(self, edits: list[dict],
                     files: dict[str, str]) -> tuple[list[dict], list[str]]:
        """
        Check every SEARCH/REPLACE block against the actual file content
        BEFORE showing anything to the user.

        Returns (verified_edits, warnings).
        Edits that don't match are replaced with a warning variant so the
        caller can decide whether to retry.
        """
        verified: list[dict] = []
        warnings: list[str]  = []

        for edit in edits:
            if edit["action"] != "SEARCH_REPLACE":
                verified.append(edit)
                continue

            content = files.get(edit["file"], "")
            if not content:
                # Try reading from disk
                path = self.indexer.root / edit["file"]   # type: ignore[attr-defined]
                if path.exists():
                    content = read_utf8(path)

            if not content:
                warnings.append(f"{edit['file']}: file not found")
                verified.append(edit)  # let Patcher handle the error
                continue

            try:
                matched = _sr_apply(content, edit["search"], edit["replace"])
            except Exception:
                matched = None

            if matched is None:
                warnings.append(
                    f"{edit['file']}: SEARCH text not found — "
                    "model will be asked to retry with exact lines"
                )
                # Tag the edit so Patcher knows it's unverified
                edit = {**edit, "_unverified": True}

            verified.append(edit)

        return verified, warnings

    def fix_error(self, error: str, force_files: Optional[list[str]] = None) -> str:
        hint_files = force_files or self._error_files(error)

        # Step 1 — Triage
        console.print("[dim]Step 1/3  Triaging relevant files…[/dim]")
        triaged    = self.triage(error, hint_files=hint_files)
        all_files  = list(dict.fromkeys(hint_files + triaged))
        console.print(f"[dim]  → {', '.join(all_files) or '(none found)'}[/dim]")

        # Step 2 — Load in full + follow deps
        console.print("[dim]Step 2/3  Loading files…[/dim]")
        loaded     = self.load_files(all_files)
        file_blocks = self._format_loaded(loaded)
        snippets   = self._error_snippets(error, self.indexer)

        # Step 3 — Analyze
        console.print("[dim]Step 3/3  Generating fix…[/dim]")
        prompt = (
            f"## ERROR / REQUEST\n```\n{error}\n```\n\n"
            f"{snippets}\n\n"
            f"{file_blocks}\n\n"
            "## TASK\n"
            "1. Root Cause — one sentence.\n"
            "2. Use SEARCH/REPLACE for .gd / .tscn — copy SEARCH lines VERBATIM\n"
            "   from the FILE blocks above (exact spacing, exact characters).\n"
            "3. Use SET_CONFIG for .godot / .cfg changes.\n"
            "4. Use NEW_FILE for new scripts (100% content).\n"
            "5. ## Setup Steps — every manual editor action."
        )
        return self._stream(prompt, system=SYSTEM_FIX, model=self.code_model)

    def add_feature(self, description: str,
                    force_files: Optional[list[str]] = None) -> str:
        hint_files = force_files or []

        # Step 1 — Triage
        console.print("[dim]Step 1/3  Triaging relevant files…[/dim]")
        triaged    = self.triage(description, hint_files=hint_files)
        all_files  = list(dict.fromkeys(hint_files + triaged))
        console.print(f"[dim]  → {', '.join(all_files) or '(none found)'}[/dim]")

        # Step 2 — Load
        console.print("[dim]Step 2/3  Loading files…[/dim]")
        loaded      = self.load_files(all_files)
        file_blocks = self._format_loaded(loaded)

        # Step 3 — Implement
        console.print("[dim]Step 3/3  Generating implementation…[/dim]")
        prompt = (
            f"## FEATURE REQUEST\n{description}\n\n"
            f"{file_blocks}\n\n"
            "## TASK\n"
            "1. 2–4 bullet plan.\n"
            "2. Use SEARCH/REPLACE for .gd / .tscn — copy SEARCH text VERBATIM\n"
            "   from the FILE blocks above.\n"
            "3. Use SET_CONFIG for project.godot / .cfg changes.\n"
            "4. Use NEW_FILE for new scripts (100% content).\n"
            "5. ## Setup Steps — nodes, autoloads, signals, editor actions."
        )
        return self._stream(prompt, system=SYSTEM_FEATURE, model=self.code_model)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _format_loaded(self, loaded: dict[str, str]) -> str:
        """Format loaded files as clearly labelled blocks for the prompt."""
        if not loaded:
            return ""
        MAX_CHARS = 12_000
        parts: list[str] = []
        total = 0
        for path, content in loaded.items():
            remaining = MAX_CHARS - total
            if remaining <= 0:
                parts.append(f"## FILE: {path}\n(omitted — context limit)")
                break
            snippet = content[:remaining]
            trimmed = len(content) > remaining
            parts.append(
                f"## FILE: {path}\n"
                "Copy lines verbatim into SEARCH blocks.\n"
                f"```\n{snippet}"
                + ("\n# ... (truncated)" if trimmed else "")
                + "\n```"
            )
            total += len(snippet)
        return "\n\n".join(parts)

    def chat(self, cmd: str) -> str:
        """
        Explicit /chat: conversational answers about Godot or the project.
        Does NOT write files. If the answer happens to include code blocks
        the patcher will catch them and ask confirmation as normal.
        """
        model = self._route_model(cmd)
        ctx   = self._context(cmd)
        prompt = f"{cmd}\n\nProject context:\n{ctx}" if ctx else cmd
        return self._stream(prompt, model=model)


# ─── Patcher ─────────────────────────────────────────────────────────────────
#
# Three patch modes — each optimised for reliability:
#
#  ① SEARCH/REPLACE (aider format)  — .gd, .tscn, .tres, any text file
#     Model outputs the git-conflict-style markers. Models know this format
#     from training data. Python finds the text and replaces it.
#
#  ② SET_CONFIG                     — .godot, .cfg only
#     Model outputs section/key/value. Python edits the INI-like file
#     directly. Zero text matching — 100% reliable.
#
#  ③ EDIT_FILE / NEW_FILE           — new files or large rewrites
#     Full file write with truncation guard.

# ① SEARCH/REPLACE regex
# Tolerates models that vary marker length (5–12 chars), wrap blocks in
# code fences, add markdown prefixes, or put trailing text on marker lines.
_SEARCH_REPLACE_RE = re.compile(
    r'(?:^|\n)'                                  # start or after newline
    r'[ \t]*(?:[*#>`]+[ \t]*)*'                # optional markdown prefix
    r'(?:(?:SEARCH[/\\-]?REPLACE)[ \t]*:?[ \t]*)?'  # optional "SEARCH/REPLACE:" header
    r'([^\n<>=`*#\\]+\.(?:gd|tscn|tres|godot|cfg|gdshader|gdshaderinc|json|txt))'
    r'[^\n]*\n'                                 # filepath line (trailing text ignored)
    r'(?:[ \t]*```[^\n]*\n)?'                  # optional opening code fence
    r'[ \t]*<{5,12}[ \t]+SEARCH[^\n]*\n'     # <<<<<<< SEARCH  (5–12 <'s)
    r'(.*?)'                                       # SEARCH block content
    r'\n[ \t]*={5,12}[^\n]*\n'               # ======= divider
    r'(.*?)'                                       # REPLACE block content
    r'\n[ \t]*>{5,12}[ \t]+REPLACE'            # >>>>>>> REPLACE
    r'[^\n]*(?:\n[ \t]*```)??',               # optional trailing + closing fence
    re.MULTILINE | re.DOTALL,
)

# ② SET_CONFIG regex
_SET_CONFIG_RE = re.compile(
    r'SET_CONFIG:\s*([^\n]+)\n'
    r'SECTION:\s*([^\n]+)\n'
    r'KEY:\s*([^\n]+)\n'
    r'VALUE:\s*([^\n]+)',
    re.IGNORECASE,
)

# ③ EDIT_FILE / NEW_FILE regex
_BLOCK_RE = re.compile(
    r'(EDIT_FILE|NEW_FILE):\s*([^\n`]+?)\s*\n```[a-zA-Z0-9_+\-. ]*\n(.*?)```',
    re.DOTALL,
)

_SETUP_RE = re.compile(
    r'##\s*Setup Steps\s*\n(.*?)(?=\n##|\Z)',
    re.DOTALL | re.IGNORECASE,
)

_TRUNCATION_RE = re.compile(
    r'#[^\n]*'
    r'(?:'
    r'\.{2,}[^\n]{0,60}(?:rest|exist|unchanged|method|function|code|continu|previous|same|keep|omit)|'
    r'(?:rest|exist)\w*[^\n]{0,30}(?:code|file|script|content|implement)|'
    r'\[\s*\.{2,}[^\n]{0,40}\]|'
    r'\(\s*\.{2,}[^\n]{0,40}\)'
    r')',
    re.IGNORECASE,
)


def _set_godot_config(content: str, section: str, key: str, value: str) -> tuple[str, bool]:
    """
    Set key=value inside [section] in a Godot INI-style config file.
    Returns (new_content, changed).  Pure Python — no text matching needed.
    """
    section_re = re.compile(
        rf'(\[{re.escape(section)}\][^\[]*)',
        re.DOTALL,
    )
    m = section_re.search(content)
    if not m:
        # Section not found — append it
        addition = f"\n[{section}]\n{key}={value}\n"
        return content + addition, True

    sec_text = m.group(1)
    key_re   = re.compile(rf'^({re.escape(key)})=.*$', re.MULTILINE)

    if key_re.search(sec_text):
        new_sec = key_re.sub(rf'\g<1>={value}', sec_text)
    else:
        # Key not in section — insert before the closing blank line
        new_sec = sec_text.rstrip() + f'\n{key}={value}\n'

    new_content = content[: m.start()] + new_sec + content[m.end() :]
    return new_content, new_content != content


def _sr_apply(original: str, search: str, replace: str) -> Optional[str]:
    """
    Apply one SEARCH/REPLACE block with progressive whitespace relaxation.
    Returns the new content or None if the search text was not found.
    """
    # 1. Exact match
    if search in original:
        return original.replace(search, replace, 1)
    # 2. Strip trailing whitespace on every line
    def strip_trailing(s: str) -> str:
        return "\n".join(l.rstrip() for l in s.splitlines())
    s2, o2 = strip_trailing(search), strip_trailing(original)
    if s2 in o2:
        return o2.replace(s2, strip_trailing(replace), 1)
    # 3. Collapse all internal whitespace runs to single space
    def norm_ws(s: str) -> str:
        return re.sub(r'[ \t]+', ' ', s.strip())
    s3, o3 = norm_ws(search), norm_ws(original)
    if s3 in o3:
        return o3.replace(s3, norm_ws(replace), 1)
    return None


class GodotValidator:
    """
    Runs Godot headless after patches to catch parse/compile errors
    before the developer reopens the editor.
    """
    def __init__(self, project_path: str) -> None:
        self.project_path = project_path
        self.exe          = self._find_godot()

    @staticmethod
    def _find_godot() -> str:
        names = ["godot4", "godot"]
        if platform.system() == "Windows":
            names = ["godot4.exe", "godot.exe"] + names
        elif platform.system() == "Darwin":
            names = [
                "/Applications/Godot.app/Contents/MacOS/Godot",
                "/Applications/Godot_v4.app/Contents/MacOS/Godot",
            ] + names
        for n in names:
            try:
                subprocess.run([n, "--version"], capture_output=True,
                               check=True, timeout=5)
                return n
            except Exception:
                pass
        return ""

    def validate(self) -> dict:
        """Return {"ok": bool|None, "errors": [str], "note": str}."""
        if not self.exe:
            return {"ok": None, "errors": [], "note": "Godot not found in PATH"}
        try:
            r = subprocess.run(
                [self.exe, "--headless", "--path", self.project_path, "--quit-after", "3"],
                capture_output=True, text=True, timeout=30,
            )
            combined = r.stdout + r.stderr
            errors = [l for l in combined.splitlines()
                      if re.match(r'^(SCRIPT ERROR|ERROR):', l)]
            return {"ok": len(errors) == 0, "errors": errors, "note": ""}
        except subprocess.TimeoutExpired:
            return {"ok": None, "errors": [], "note": "Validation timed out (30s)"}
        except Exception as exc:
            return {"ok": None, "errors": [], "note": str(exc)}


class Patcher:
    def __init__(self, root: str, indexer: "CodebaseIndexer",
                 git: Optional["GitManager"] = None,
                 logger: Optional["ConversationLogger"] = None,
                 validator: Optional[GodotValidator] = None,
                 runner: Optional["GodotRunner"] = None) -> None:
        self.root      = Path(root).resolve()
        self.indexer   = indexer
        self.git       = git
        self.logger    = logger
        self.validator = validator
        # Updated by the REPL when /run starts — lets apply() mute runner output
        self.runner: Optional["GodotRunner"] = runner

    def parse(self, response: str) -> list[dict]:
        edits: list[dict] = []

        # ① SEARCH/REPLACE blocks
        for filepath, search, replace in _SEARCH_REPLACE_RE.findall(response):
            filepath = self._clean_path(filepath.strip())
            if not filepath:
                continue
            edits.append({
                "action":  "SEARCH_REPLACE",
                "file":    filepath,
                "search":  search,
                "replace": replace,
                "code":    f"SEARCH:\n{search}\nREPLACE:\n{replace}",
            })

        # ② SET_CONFIG blocks
        for filepath, section, key, value in _SET_CONFIG_RE.findall(response):
            filepath = self._clean_path(filepath.strip())
            if not filepath:
                continue
            edits.append({
                "action":  "SET_CONFIG",
                "file":    filepath,
                "section": section.strip(),
                "key":     key.strip(),
                "value":   value.strip(),
                "code":    f"[{section.strip()}]\n{key.strip()}={value.strip()}",
            })

        # ③ EDIT_FILE / NEW_FILE blocks
        for action, filepath, code in _BLOCK_RE.findall(response):
            filepath = self._clean_path(filepath)
            if not filepath:
                continue
            edits.append({
                "action": action.strip(),
                "file":   filepath,
                "code":   code.strip(),
            })

        return edits

    @staticmethod
    def _clean_path(p: str) -> str:
        # Strip res:// / user:// scheme prefixes
        p = re.sub(r'^(?:res|user)://', '', p)
        # Strip markdown bold/italic/code/heading chars
        p = re.sub(r'[\*`"\'#]+', '', p)
        # Strip leading "path/to/" segments models sometimes invent
        p = re.sub(r'^(?:path[/\\\\]to[/\\\\])+', '', p, flags=re.IGNORECASE)
        # Strip leading "3. " or "### 3. " heading prefixes
        p = re.sub(r'^\d+\.\s*', '', p)
        return p.strip().lstrip("/")

    @staticmethod
    def extract_setup_steps(response: str) -> str:
        m = _SETUP_RE.search(response)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _find_truncation(code: str, original_path: Optional[Path] = None) -> list[str]:
        hits = [l.strip() for l in code.splitlines() if _TRUNCATION_RE.search(l)]
        if original_path and original_path.exists():
            orig = original_path.stat().st_size
            new  = len(code.encode())
            if orig > 200 and new < orig * 0.5:
                hits.append(
                    f"[SIZE] Output {new}B vs original {orig}B "
                    f"({new/orig:.0%}) — likely incomplete"
                )
        return hits

    def apply(self, edits: list[dict], response: str = "",
              op_type: str = "fix") -> None:
        if not edits:
            return

        n = len(edits)
        console.print(Rule(f"[yellow]{n} change(s) proposed[/yellow]"))

        applied: list[str] = []
        skipped: list[str] = []
        any_truncation = False

        # Mute Godot runner background output while we wait for user input
        if self.runner:
            self.runner.pause_output.set()

        for edit in edits:
            path   = self.root / edit["file"]
            action = edit["action"]

            # ── ① SEARCH/REPLACE ─────────────────────────────────────────────
            if action == "SEARCH_REPLACE":
                console.print(f"\n  [magenta]PATCH[/magenta]  [white]{edit['file']}[/white]")
                console.print(
                    f"  [dim]SEARCH:[/dim] [red]{edit['search'].strip()[:100]}…[/red]"
                )
                console.print(
                    f"  [dim]REPLACE:[/dim] [green]{edit['replace'].strip()[:100]}…[/green]"
                )
                if not path.exists():
                    console.print(f"  [red]File not found: {edit['file']}[/red]")
                    skipped.append(edit["file"])
                    continue
                original    = read_utf8(path)
                new_content = _sr_apply(original, edit["search"], edit["replace"])
                if new_content is None:
                    console.print(
                        f"  [red]SEARCH text not found in {edit['file']}.[/red]\n"
                        "  [dim]Tip: run [bold]/retry[/bold] — the file content is now\n"
                        "  injected so the model will use exact lines next time.[/dim]"
                    )
                    skipped.append(edit["file"])
                    continue
                if not Confirm.ask(f"  Apply patch to [cyan]{edit['file']}[/cyan]?",
                                   default=True):
                    skipped.append(edit["file"])
                    continue
                write_utf8(path, new_content)
                applied.append(edit["file"])
                console.print("  [green]✓ Patched[/green]")
                self.indexer.queue_reindex(path)
                continue

            # ── ② SET_CONFIG ─────────────────────────────────────────────────
            if action == "SET_CONFIG":
                console.print(
                    f"\n  [blue]CONFIG[/blue]  [white]{edit['file']}[/white]  "
                    f"[dim][{edit['section']}] {edit['key']} = {edit['value']}[/dim]"
                )
                if not path.exists():
                    console.print(f"  [red]File not found: {edit['file']}[/red]")
                    skipped.append(edit["file"])
                    continue
                original = read_utf8(path)
                new_content, changed = _set_godot_config(
                    original, edit["section"], edit["key"], edit["value"]
                )
                if not changed:
                    console.print(
                        f"  [yellow]Value already set — skipping.[/yellow]"
                    )
                    skipped.append(edit["file"])
                    continue
                if not Confirm.ask(f"  Apply to [cyan]{edit['file']}[/cyan]?",
                                   default=True):
                    skipped.append(edit["file"])
                    continue
                write_utf8(path, new_content)
                applied.append(edit["file"])
                console.print("  [green]✓ Config updated[/green]")
                self.indexer.queue_reindex(path)
                continue

            # ── ③ EDIT_FILE / NEW_FILE ────────────────────────────────────────
            label = "[cyan]EDIT[/cyan]" if action == "EDIT_FILE" else "[green]NEW [/green]"
            code  = edit["code"]
            console.print(f"\n  {label}  [white]{edit['file']}[/white]")

            orig_path    = path if path.exists() else None
            trunc_lines  = self._find_truncation(code, orig_path)
            write_direct = False

            if trunc_lines:
                console.print(f"  [bold red]⚠  TRUNCATION in {edit['file']}:[/bold red]")
                for tl in trunc_lines[:4]:
                    console.print(f"  [red]    {tl}[/red]")
                try:
                    choice = Prompt.ask("  Action",
                                        choices=["skip", "apply", "retry"],
                                        default="skip")
                except KeyboardInterrupt:
                    choice = "skip"
                if choice == "skip":
                    skipped.append(edit["file"])
                    continue
                elif choice == "retry":
                    console.print("  [dim]Use /retry to re-run with stricter prompt.[/dim]")
                    skipped.append(edit["file"])
                    continue
                else:
                    console.print("  [red]Writing despite truncation warning.[/red]")
                    any_truncation = True
                    write_direct   = True

            preview = code[:800] + (" …" if len(code) > 800 else "")
            console.print(Syntax(preview, "gdscript", theme="monokai", line_numbers=True))

            if not write_direct:
                if not Confirm.ask(f"  Apply [cyan]{edit['file']}[/cyan]?", default=True):
                    skipped.append(edit["file"])
                    continue

            path.parent.mkdir(parents=True, exist_ok=True)
            write_utf8(path, code + "\n")
            applied.append(edit["file"])
            console.print("  [green]✓ Written[/green]")
            self.indexer.queue_reindex(path)

        # Resume runner output now that all user prompts are done
        if self.runner:
            self.runner.pause_output.clear()
            # Show deferred notification if errors were queued while muted
            if not self.runner.error_queue.empty():
                console.print(
                    "\n[bold red on dark_red]  ⚠  Godot errors captured  [/bold red on dark_red] "
                    "[red]press Enter or type /last to fix[/red]\n"
                )

        # ── summary bar ───────────────────────────────────────────────────────
        console.print(Rule())
        if applied:
            console.print(f"  [green]Applied:[/green] {', '.join(applied)}")
        if skipped:
            console.print(f"  [dim]Skipped: {', '.join(skipped)}[/dim]")

        # ── validation ────────────────────────────────────────────────────────
        if applied and self.validator and self.validator.exe:
            with console.status("[dim]Validating with Godot headless…[/dim]"):
                result = self.validator.validate()
            if result["ok"] is True:
                console.print("  [green]✓ Godot validation passed[/green]")
            elif result["ok"] is False:
                console.print(
                    f"  [red]⚠  Godot reports {len(result['errors'])} error(s):[/red]"
                )
                for e in result["errors"][:6]:
                    console.print(f"  [red]    {e}[/red]")
                console.print(
                    "  [dim]These errors are now in /last — type /fix to address them.[/dim]"
                )
            # note: ok=None means validator not available, already shown at startup

        # ── git commit ────────────────────────────────────────────────────────
        git_sha = ""
        if applied and self.git and self.git.enabled:
            files_str  = ", ".join(applied[:3])
            if len(applied) > 3:
                files_str += f" +{len(applied)-3} more"
            trunc_note = " [TRUNCATION-WARNED]" if any_truncation else ""
            self.git.commit_patch(
                f"[{op_type}] {files_str}{trunc_note}",
                op_type=op_type, files=applied, had_truncation=any_truncation,
            )
            sha_r  = self.git._run("rev-parse", "--short", "HEAD")
            git_sha = sha_r.stdout.strip()

        if self.logger and (applied or skipped):
            self.logger.log_files(applied, skipped, git_sha=git_sha)



# Targeted find-and-replace format — no truncation risk because the model
# only outputs the lines that actually change, not the whole file.
#
# REPLACE_IN: project.godot
# FIND:
# ```
# config/version="1b"
# ```
# WITH:
# ```
# config/version="2beta"
# ```
#
# The regex is intentionally permissive about:
#  - leading whitespace / markdown headers (### 3. REPLACE_IN)
#  - blank lines between the keyword and the code fence
#  - fence language tags (```gdscript, ```ini, ``` etc.)
#  - trailing whitespace on each block boundary
_REPLACE_RE = re.compile(
    r'REPLACE_IN:\s*([^\n`*#]+?)\s*\n'           # file path (strip markdown)
    r'\s*FIND:\s*\n\s*```[^\n]*\n(.*?)```'        # FIND block
    r'\s*\n\s*WITH:\s*\n\s*```[^\n]*\n(.*?)```',  # WITH block
    re.DOTALL,
)

_SETUP_RE = re.compile(
    r'##\s*Setup Steps\s*\n(.*?)(?=\n##|\Z)',
    re.DOTALL | re.IGNORECASE,
)

# Matches lines that look like truncation placeholders, e.g.:
#   # ... rest of the code ...
#   # ... existing code ...
#   # ... (unchanged) ...
#   # [rest of file unchanged]
#   # (rest of code same as before)
_TRUNCATION_RE = re.compile(
    r'#[^\n]*'
    r'(?:'
    r'\.{2,}[^\n]{0,60}(?:rest|exist|unchanged|method|function|code|continu|previous|same|keep|omit)|'
    r'(?:rest|exist)\w*[^\n]{0,30}(?:code|file|script|content|implement)|'
    r'\[\s*\.{2,}[^\n]{0,40}\]|'
    r'\(\s*\.{2,}[^\n]{0,40}\)'
    r')',
    re.IGNORECASE,
)
# ─── File watcher ─────────────────────────────────────────────────────────────

class GodotWatcher(FileSystemEventHandler):
    """
    Queues changed files instead of re-indexing immediately.
    The REPL drains the queue between prompts so streaming output is never
    interrupted by background work.
    """
    def __init__(self, indexer: CodebaseIndexer) -> None:
        self.indexer   = indexer
        self._cooldown: dict[str, float] = {}

    def on_modified(self, event) -> None:   # type: ignore[override]
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix not in Config.GODOT_TEXT_EXTS:
            return
        now = time.time()
        key = str(path)
        if now - self._cooldown.get(key, 0) < 2.0:
            return
        self._cooldown[key] = now
        self.indexer.queue_reindex(path)


# ─── Multiline input ─────────────────────────────────────────────────────────

def collect_multiline(label: str, session: "PromptSession") -> str:
    """
    Multiline paste input using prompt_toolkit throughout — no raw input() calls.
    Terminate with: a line containing only END, or Ctrl+D (Mac/Linux), Ctrl+Z+Enter (Windows).

    Why prompt_toolkit for everything:
    The REPL uses PromptSession which takes over stdin. Mixing with raw input()
    causes pasted lines after the first to be read as separate REPL commands.
    Using session.prompt() here keeps stdin control consistent.
    """
    console.print(
        f"\n[dim]({label})[/dim]\n"
        "[dim]Paste your text below. Type [bold]END[/bold] on its own line to finish.\n"
        "Ctrl+C cancels.[/dim]\n"
    )
    lines: list[str] = []
    while True:
        try:
            line = session.prompt("  … ")
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print("[dim]Cancelled.[/dim]")
            return ""
        if line.rstrip("\r\n") == "END":
            break
        lines.append(line)
    return "\n".join(lines)


# ─── Godot log integration ───────────────────────────────────────────────────

def _godot_log_path(project_path: str) -> Optional[Path]:
    """
    Locate Godot's on-disk log file for this project.
    Godot writes the same output you see in the Output panel to a log file.

    Paths by OS:
      Windows : %APPDATA%/Godot/app_userdata/<project>/logs/godot.log
      macOS   : ~/Library/Application Support/Godot/app_userdata/<project>/logs/godot.log
      Linux   : ~/.local/share/godot/app_userdata/<project>/logs/godot.log
    """
    # Read project name from project.godot
    godot_proj = Path(project_path) / "project.godot"
    project_name = Path(project_path).name          # fallback = folder name
    if godot_proj.exists():
        text = godot_proj.read_text(errors="replace")
        m = re.search(r'config/name="([^"]+)"', text)
        if m:
            project_name = m.group(1)

    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA", "")) / "Godot" / "app_userdata"
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "Godot" / "app_userdata"
    else:
        base = Path.home() / ".local" / "share" / "godot" / "app_userdata"

    candidate = base / project_name / "logs" / "godot.log"
    return candidate if candidate.exists() else None


_GODOT_ERROR_RE = re.compile(r'^(SCRIPT ERROR|ERROR|USER ERROR|WARNING):')


class GodotLogWatcher:
    """
    Tails Godot's log file for new ERROR lines.
    Runs in a background thread; results are queued for the REPL to drain.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path    = log_path
        self.error_queue: queue.Queue[str] = queue.Queue()
        self._stop       = threading.Event()
        self._thread     = threading.Thread(target=self._tail, daemon=True)
        self._pos        = log_path.stat().st_size   # start at end of existing content

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _tail(self) -> None:
        while not self._stop.is_set():
            try:
                size = self.log_path.stat().st_size
                if size > self._pos:
                    with self.log_path.open(encoding="utf-8", errors="replace") as f:
                        f.seek(self._pos)
                        new_text = f.read()
                    self._pos = size
                    errors = self._extract_errors(new_text)
                    if errors:
                        self.error_queue.put(errors)
                elif size < self._pos:
                    # Log was rotated / Godot restarted
                    self._pos = 0
            except Exception:
                pass
            time.sleep(0.8)

    @staticmethod
    def _extract_errors(text: str) -> str:
        """Collect ERROR/SCRIPT ERROR lines and their stack traces."""
        lines      = text.splitlines()
        collecting = False
        block: list[str] = []
        for line in lines:
            if _GODOT_ERROR_RE.match(line):
                collecting = True
            if collecting:
                block.append(line)
        return "\n".join(block)

    def drain(self) -> Optional[str]:
        """Return all queued error batches merged, or None if none."""
        batches: list[str] = []
        while True:
            try:
                batches.append(self.error_queue.get_nowait())
            except queue.Empty:
                break
        return "\n".join(batches) if batches else None


class GodotRunner:
    """
    Launches Godot as a subprocess, captures stdout+stderr in real time,
    and queues error blocks for the REPL to process.
    """

    def __init__(self, godot_exe: str, project_path: str) -> None:
        self.godot_exe    = godot_exe
        self.project_path = project_path
        self.error_queue: queue.Queue[str] = queue.Queue()
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        # Set by Patcher.apply() while waiting for user input so the background
        # thread does not print over the Rich/prompt_toolkit confirmation prompts.
        self.pause_output: threading.Event = threading.Event()

    def start(self) -> bool:
        try:
            self._proc = subprocess.Popen(
                [self.godot_exe, "--path", self.project_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
            return True
        except FileNotFoundError:
            return False

    def _reader(self) -> None:
        """
        Reads Godot subprocess stdout line by line.

        Flush strategy:
          - A separate mini-thread pushes raw lines into an inner queue.
          - The outer loop pulls with a 0.8s timeout.
          - If the timeout fires while an error block is in progress, we flush
            immediately — this handles the common case where Godot opens its
            window and stdout goes silent after the startup error burst.
        """
        assert self._proc and self._proc.stdout

        # Inner queue so we can implement a read timeout on Windows
        # (select() does not work on Windows subprocess pipes)
        line_q: queue.Queue[Optional[str]] = queue.Queue()

        def _pipe_reader() -> None:
            try:
                for raw in self._proc.stdout:          # type: ignore[union-attr]
                    line_q.put(raw)
            finally:
                line_q.put(None)                       # sentinel: process closed

        threading.Thread(target=_pipe_reader, daemon=True).start()

        error_block: list[str] = []
        FLUSH_TIMEOUT = 0.8   # seconds of silence → flush pending block

        while True:
            try:
                raw_line = line_q.get(timeout=FLUSH_TIMEOUT)
            except queue.Empty:
                # Stdout went quiet — flush whatever we have
                if error_block:
                    self._flush_block(error_block)
                    error_block = []
                continue

            if raw_line is None:
                break                                  # process exited

            line = raw_line.rstrip()

            # ── colour-coded echo (suppressed while patcher waits for input) ─
            if not self.pause_output.is_set():
                if _GODOT_ERROR_RE.match(line):
                    console.print(f"[red]{line}[/red]")
                elif re.match(r'^\s+(at:|GDScript backtrace)', line):
                    console.print(f"[yellow]{line}[/yellow]")
                else:
                    console.print(f"[dim]{line}[/dim]")

            # ── block collection ──────────────────────────────────────────
            is_error = bool(_GODOT_ERROR_RE.match(line))
            is_trace = bool(re.match(r'^\s+(at:|GDScript backtrace)', line))

            if is_error:
                error_block.append(line)
            elif is_trace and error_block:
                error_block.append(line)
            elif error_block:
                # Non-error line after a block: end the block
                self._flush_block(error_block)
                error_block = []

        # Final flush when the process exits
        if error_block:
            self._flush_block(error_block)

    def _flush_block(self, block: list[str]) -> None:
        """Queue an error batch. Print notification only when not mid-prompt."""
        text = "\n".join(block)
        self.error_queue.put(text)
        if not self.pause_output.is_set():
            console.print(
                "\n[bold red on dark_red]  ⚠  Godot errors captured  [/bold red on dark_red] "
                "[red]press Enter or type /last to fix[/red]\n"
            )

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def drain(self) -> Optional[str]:
        batches: list[str] = []
        while True:
            try:
                batches.append(self.error_queue.get_nowait())
            except queue.Empty:
                break
        return "\n".join(batches) if batches else None


# ─── UI helpers ──────────────────────────────────────────────────────────────

BANNER = r"""\
[bold cyan]
  ██████╗  ██████╗ ██████╗  ██████╗ ████████╗     █████╗ ██╗
 ██╔════╝ ██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝    ██╔══██╗██║
 ██║  ███╗██║   ██║██║  ██║██║   ██║   ██║       ███████║██║
 ██║   ██║██║   ██║██║  ██║██║   ██║   ██║       ██╔══██║██║
 ╚██████╔╝╚██████╔╝██████╔╝╚██████╔╝   ██║       ██║  ██║██║
  ╚═════╝  ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝       ╚═╝  ╚═╝╚═╝
[/bold cyan]

[green]
                            .....              .....                                      
                        ...........            ...........                                 
                        .............         ............                                 
                        ..................................                                 
                        ..................................                                 
                        ..................................                                 
        ...        ........................................        ...                   
        .......   .............................................  ........                 
    ....................................................................                
    .......................................................................              
    ........................................................................              
    ......................................................................               
    ....................................................................                
        ..................................................................                 
        ...........***##***..........................***##***...........                  
        .........**#**...**#*......................*#**...**#**.........                  
        ........*##*.      .**........****........**.       *##*........                  
        ........*##.        .#*.......####........#.        .##*........                  
        ........*##.        **........####........**        .##*........                  
        .........*#*..    .**.........####.........**.    ..*#*.........                  
        ...........*********..........####..........**********..........                  
        ...............................**...............................                  
        ................................................................                  
        ******....................................................******.                 
        ******######*............**************............*######******.                 
        ..........*##...........*##***********#*...........##*..........                  
        ...........##...........*#*..........*#*..........*##*..........                  
        ...........##***********##*...........##**********###...........                  
        ...........**************............**************...........                   
        ............................................................                    
            ........................................................                      
            ....................................................                        
                ............................................                            
                        .................................                                  
                                    ......
[/green]

[dim]  GodotAI Dev · codebase-aware AI for Godot 4 · powered by Ollama[/dim]
"""

def help_panel() -> None:
    console.print(Panel(
        "[bold green]Core commands[/bold green]\n"
        "[green]/fix[/green]        Paste a Godot error → diagnose & patch files\n"
        "[green]/feature[/green]    Describe a feature → generate & write GDScript\n"
        "[green]/chat[/green]       Ask any question about Godot or your project\n"
        "[green]/retry[/green]      Re-run last /fix, /feature, or /chat (stricter prompt)\n"
        "[green]/index[/green]      Re-scan and re-index the whole project\n"
        "[green]/tree[/green]       Show full project file tree\n"
        "[green]/files[/green]      List indexed files\n"
        "[green]/models[/green]     List Ollama models + switch code/chat model\n"
        "[green]/clear[/green]      Clear conversation history\n\n"

        "[bold magenta]Godot live integration[/bold magenta]\n"
        "[magenta]/run[/magenta]    Launch Godot directly from here.\n"
        "         All output streams here. Errors → notification → press Enter to fix.\n"
        "         Usage: /run   or   /run C:/path/to/godot.exe\n\n"
        "[magenta]/watch[/magenta]  Watch Godot's log file (Godot open separately).\n"
        "         Run your game → errors auto-appear. Toggle with /watch again.\n\n"
        "[magenta]/last[/magenta]   Show last captured error batch and offer to fix.\n\n"

        "[bold cyan]Git safety net[/bold cyan]\n"
        "[cyan]/status[/cyan]           git status\n"
        "[cyan]/diff[/cyan]             Changes since session started\n"
        "[cyan]/log[/cyan]              Session commits\n"
        "[cyan]/history[/cyan]          Full table: patches, SHAs, truncation flags\n"
        "[cyan]/branches[/cyan]         List all godot-ai/* branches\n"
        "[cyan]/checkout <branch>[/cyan] Switch to a godot-ai branch\n"
        "[cyan]/save [name][/cyan]      Create a named git checkpoint tag\n"
        "[cyan]/undo[/cyan]             Revert last patch (call repeatedly for more)\n\n"

        "[green]/help[/green]       Show this panel\n"
        "[green]/quit[/green]       Exit\n\n"

        "[dim]━━━ Truncation protection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  • Current file content injected into every /fix and /feature prompt.\n"
        "  • Pattern check: flags '# ... rest of code ...' style placeholders.\n"
        "  • Size check: flags output < 50% of original — silent truncation.\n"
        "  • Both checks warn and default to SKIP before any write happens.\n\n"
        "Ctrl+C during AI = interrupt (keeps partial). Ctrl+C at prompt = exit.[/dim]",
        title="[bold]GodotAI Dev[/bold]",
        border_style="cyan",
    ))


# ─── REPL ────────────────────────────────────────────────────────────────────

def repl(project_path: str, ollama_host: str, git_enabled: bool = True) -> None:
    Config.OLLAMA_HOST = ollama_host

    console.print(BANNER)
    console.print(f"[cyan]Project   :[/cyan] {project_path}")
    console.print(f"[cyan]Ollama    :[/cyan] {ollama_host}")
    console.print(f"[cyan]Code model:[/cyan] {Config.CODE_MODEL}")
    console.print(f"[cyan]Chat model:[/cyan] {Config.CHAT_MODEL}")
    console.print(f"[cyan]Embeds    :[/cyan] {Config.EMBED_MODEL}\n")

    git     = GitManager(Path(project_path), enabled=git_enabled)
    logger  = ConversationLogger(project_path)
    indexer = CodebaseIndexer(project_path, ollama_host)
    agent   = GodotAgent(indexer, logger=logger)
    patcher   = Patcher(project_path, indexer, git=git, logger=logger,
                        validator=GodotValidator(project_path))
    if patcher.validator.exe:
        console.print(f"[dim]✓ Godot validator: {patcher.validator.exe}[/dim]")
    else:
        console.print("[dim]Godot not found in PATH — post-patch validation disabled.[/dim]")
    at_completer = AtFileCompleter()
    session      = PromptSession(completer=at_completer)

    console.print(f"[dim]📝 Session log: {logger.path}[/dim]")
    logger.log_event("SESSION START",
                     f"Project: {project_path}\nOllama: {ollama_host}\n"
                     f"Code model: {agent.code_model}\nChat model: {agent.chat_model}")

    # ── Godot integration state ──────────────────────────────────────────────
    log_watcher:  Optional[GodotLogWatcher] = None
    godot_runner: Optional[GodotRunner]     = None
    last_errors:  str = ""
    autowatch:    bool = False          # True = auto-offer /fix when errors found
    # /retry state — stores the last runnable prompt
    last_prompt:  str = ""
    last_op:      str = ""             # "fix" | "feature" | "chat"

    # Try to locate the Godot log file immediately
    log_path = _godot_log_path(project_path)
    if log_path:
        console.print(f"[dim]📋 Godot log found: {log_path}[/dim]")
        console.print("[dim]   Use /watch to start monitoring it for errors.[/dim]")
    else:
        console.print("[dim]📋 Godot log not found yet (start Godot once to create it).[/dim]")

    # ── initial index ────────────────────────────────────────────────────────
    count = indexer.col.count()
    if count == 0:
        console.print("[yellow]No index found — indexing project now…[/yellow]")
        n = indexer.index_project()
        at_completer.refresh(indexer.list_indexed_files())
        console.print(f"[green]✓ Indexed {n} chunks[/green]\n")
    else:
        console.print(f"[green]✓ Loaded existing index ({count} chunks)[/green]")
        indexer.project_tree = indexer.build_project_tree()
        if Confirm.ask("  Re-index project now?", default=False):
            n = indexer.index_project()
            at_completer.refresh(indexer.list_indexed_files())
            console.print(f"[green]✓ Re-indexed: {n} chunks[/green]")
        else:
            at_completer.refresh(indexer.list_indexed_files())
        console.print()

    # ── file watcher (re-indexing) ───────────────────────────────────────────
    observer = Observer()
    observer.schedule(GodotWatcher(indexer), project_path, recursive=True)
    observer.start()
    console.print("[dim]👁  Watching for file changes…[/dim]\n")

    help_panel()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _offer_fix(errors: str) -> None:
        nonlocal last_errors
        last_errors = errors
        logger.log_godot_errors(errors)
        console.print(Panel(
            errors[:2000] + ("…" if len(errors) > 2000 else ""),
            title="[red]⚠  Godot Errors Detected[/red]",
            border_style="red",
        ))
        try:
            if session.prompt("  Fix these errors? [y/N] ").strip().lower() == "y":
                response = agent.fix_error(errors)
                edits = patcher.parse(response)
                if edits:
                    patcher.apply(edits, response=response, op_type="fix")
        except KeyboardInterrupt:
            pass

    # ── main loop ────────────────────────────────────────────────────────────
    try:
        while True:
            # Drain file re-index queue
            done = indexer.drain_reindex_queue()
            if done:
                console.print(f"[dim]🔄 Re-indexed: {', '.join(done)}[/dim]")

            try:
                raw: str = session.prompt("\ngodot-ai> ")
            except (EOFError, KeyboardInterrupt):
                break

            cmd = raw.strip()
            low = cmd.lower()

            # ── always drain error queues (also triggers on bare Enter) ────
            if log_watcher:
                errors = log_watcher.drain()
                if errors:
                    _offer_fix(errors)

            if godot_runner:
                if not godot_runner.is_running():
                    # Drain final errors before marking as stopped
                    errors = godot_runner.drain()
                    if errors:
                        _offer_fix(errors)
                    console.print("[dim]Godot process exited.[/dim]")
                    godot_runner   = None
                    patcher.runner = None
                else:
                    errors = godot_runner.drain()
                    if errors:
                        _offer_fix(errors)

            if not cmd:
                continue

            # ── commands ────────────────────────────────────────────────────
            if low in ("/quit", "/exit", "quit", "exit"):
                break

            elif low == "/help":
                help_panel()

            elif low == "/clear":
                agent.history.clear()
                console.print("[green]✓ Conversation history cleared[/green]")

            elif low == "/tree":
                if indexer.project_tree:
                    console.print(Panel(
                        indexer.project_tree,
                        title="Project Tree", border_style="cyan"
                    ))
                else:
                    console.print("[yellow]Run /index first.[/yellow]")

            elif low == "/index":
                n = indexer.index_project()
                at_completer.refresh(indexer.list_indexed_files())
                console.print(f"[green]✓ Indexed {n} chunks[/green]")

            elif low == "/files":
                files = indexer.list_indexed_files()
                t = Table(title="Indexed Files", border_style="cyan")
                t.add_column("Path", style="white")
                for f in files:
                    t.add_row(f)
                console.print(t)

            elif low == "/logs":
                console.print(f"[cyan]Current log:[/cyan] {logger.path}")
                past = ConversationLogger.list_logs(project_path)
                if past:
                    t = Table(title="Recent Session Logs", border_style="cyan")
                    t.add_column("File", style="white")
                    t.add_column("Size", style="dim", justify="right")
                    for p in past:
                        size = f"{p.stat().st_size // 1024}KB"
                        current = " ← current" if p == logger.path else ""
                        t.add_row(str(p.name) + current, size)
                    console.print(t)
                    console.print(f"[dim]Log directory: {logger.path.parent}[/dim]")

            elif low in ("/model", "/models"):
                _handle_models(agent)

            # ── Godot integration ────────────────────────────────────────────

            elif low == "/last":
                if last_errors:
                    console.print(Panel(
                        last_errors,
                        title="[red]Last captured Godot errors[/red]",
                        border_style="red",
                    ))
                    if session.prompt("Fix these? [y/N] ").strip().lower() == "y":
                        response = agent.fix_error(last_errors)
                        edits = patcher.parse(response)
                        if edits:
                            patcher.apply(edits, response=response, op_type="fix")
                else:
                    console.print("[dim]No errors captured yet. Run /watch or /run first.[/dim]")

            elif low == "/watch":
                nonlocal_log_path = _godot_log_path(project_path)
                if not nonlocal_log_path:
                    console.print(
                        "[yellow]Godot log not found. Start your game in Godot at least once "
                        "to create the log file, then try /watch again.[/yellow]"
                    )
                elif log_watcher:
                    log_watcher.stop()
                    log_watcher = None
                    console.print("[dim]Log watching stopped.[/dim]")
                else:
                    log_watcher = GodotLogWatcher(nonlocal_log_path)
                    log_watcher.start()
                    console.print(Panel(
                        f"[green]✓ Watching:[/green] {nonlocal_log_path}\n\n"
                        "[white]What happens next:[/white]\n"
                        "  • Run your game inside Godot normally\n"
                        "  • When errors appear in Godot's Output panel, they auto-appear here\n"
                        "  • A [bold red]⚠ notification[/bold red] prints when errors are detected\n"
                        "  • Press [bold]Enter[/bold] or type [bold]/last[/bold] to fix them\n\n"
                        "[dim]Type /watch again to stop watching.[/dim]",
                        title="Log Watcher Active",
                        border_style="magenta",
                    ))

            elif low == "/run" or low.startswith("/run "):
                # Find godot executable
                godot_exe = cmd[5:].strip() if low.startswith("/run ") else ""
                if not godot_exe:
                    # Check remembered path first
                    settings    = _load_settings()
                    remembered  = settings.get("godot_exe", "")
                    if remembered and Path(remembered).exists():
                        console.print(f"[dim]Using remembered Godot path: {remembered}[/dim]")
                        godot_exe = remembered
                if not godot_exe:
                    # Try common locations
                    candidates = ["godot", "godot4", "Godot_v4"]
                    if platform.system() == "Windows":
                        candidates = ["godot.exe", "godot4.exe"] + candidates
                    elif platform.system() == "Darwin":
                        candidates = [
                            "/Applications/Godot.app/Contents/MacOS/Godot",
                            "/Applications/Godot_v4.app/Contents/MacOS/Godot",
                        ] + candidates
                    for c in candidates:
                        try:
                            subprocess.run([c, "--version"], capture_output=True, check=True)
                            godot_exe = c
                            break
                        except (FileNotFoundError, subprocess.CalledProcessError):
                            pass
                    if not godot_exe:
                        godot_exe = session.prompt(
                            "Godot executable path (e.g. /usr/bin/godot or C:/Godot/godot.exe): "
                        ).strip()

                if godot_runner and godot_runner.is_running():
                    godot_runner.stop()
                    console.print("[dim]Stopped previous Godot instance.[/dim]")

                godot_runner = GodotRunner(godot_exe, project_path)
                if godot_runner.start():
                    # Remember this path for next time
                    s = _load_settings()
                    s["godot_exe"] = godot_exe
                    _save_settings(s)
                    patcher.runner = godot_runner   # mute output during prompts
                    console.print(Panel(
                        f"[green]✓ Godot launched:[/green] {godot_exe}\n\n"
                        "[white]What happens next:[/white]\n"
                        "  • Godot output streams below in dim text\n"
                        "  • [red]ERROR[/red] and [yellow]SCRIPT ERROR[/yellow] lines are highlighted\n"
                        "  • When errors appear, a [bold red]⚠ notification[/bold red] prints\n"
                        "  • Press [bold]Enter[/bold] (empty input) to trigger the fix offer\n"
                        "  • Or type [bold]/last[/bold] at any time to see the last errors",
                        title="Godot Running",
                        border_style="magenta",
                    ))
                else:
                    console.print(
                        f"[red]Could not launch:[/red] {godot_exe}\n"
                        "[dim]Try: /run /full/path/to/godot[/dim]"
                    )
                    godot_runner = None

            # ── git ─────────────────────────────────────────────────────────
            elif low == "/status":
                console.print(Panel(
                    git.status(),
                    title=f"git status  [{git.branch}]",
                    border_style="cyan"
                ))

            elif low == "/diff":
                console.print(Panel(
                    git.diff(),
                    title="Changes this session",
                    border_style="cyan"
                ))

            elif low == "/log":
                console.print(Panel(
                    git.session_log(),
                    title=f"git log  [{git.branch}]",
                    border_style="cyan"
                ))

            elif low == "/history":
                if git.enabled:
                    console.print(git.patch_history_table())
                else:
                    console.print("[yellow]git safety net is disabled.[/yellow]")

            elif low == "/branches":
                if git.enabled:
                    branches = git.list_branches()
                    t = Table(title="godot-ai/* Branches", border_style="cyan")
                    t.add_column("Branch", style="white")
                    t.add_column("Last Commit", style="dim")
                    t.add_column("", style="green", width=3)
                    for b in branches:
                        t.add_row(
                            b["name"],
                            b["date"],
                            "◀" if b["current"] else "",
                        )
                    if not branches:
                        console.print("[dim]No godot-ai/* branches found yet.[/dim]")
                    else:
                        console.print(t)
                else:
                    console.print("[yellow]git safety net is disabled.[/yellow]")

            elif low.startswith("/checkout"):
                branch_name = cmd[9:].strip()
                if not branch_name:
                    console.print("[yellow]Usage: /checkout godot-ai/YYYY-MM-DD_HH-MM-SS[/yellow]")
                elif not git.enabled:
                    console.print("[yellow]git safety net is disabled.[/yellow]")
                else:
                    switched = git.checkout_branch(branch_name, session=session)
                    if switched:
                        console.print(
                            "[dim]Branch changed — re-indexing codebase "
                            "(files may differ from previous branch)…[/dim]"
                        )
                        n = indexer.index_project()
                        at_completer.refresh(indexer.list_indexed_files())
                        console.print(f"[green]✓ Re-indexed {n} chunks from {branch_name}[/green]")
                        logger.log_event("CHECKOUT + REINDEX",
                                         f"Switched to {branch_name}, indexed {n} chunks")

            elif low == "/undo":
                levels = git.undo_levels_available() if git.enabled else 0
                if levels == 0 and git.enabled:
                    console.print("[yellow]Nothing to undo — patch stack is empty.[/yellow]")
                elif Confirm.ask(
                    f"Revert last patch? ({levels} level(s) available — this is a hard reset)",
                    default=False
                ):
                    git.undo_last_patch()

            elif low == "/save" or low.startswith("/save "):
                checkpoint_name = cmd[5:].strip() if low.startswith("/save ") else ""
                if not checkpoint_name:
                    checkpoint_name = session.prompt("Checkpoint name: ").strip()
                if not checkpoint_name:
                    console.print("[dim]Cancelled.[/dim]")
                else:
                    # Sanitise: replace spaces with dashes, remove unsafe chars
                    safe = re.sub(r'[^\w\-]', '-', checkpoint_name).strip('-')
                    git.create_checkpoint(safe)

            elif low == "/retry":
                if not last_prompt:
                    console.print("[dim]No previous prompt to retry.[/dim]")
                else:
                    console.print(
                        f"[dim]Retrying last [{last_op}] prompt…[/dim]\n"
                        f"[dim]{last_prompt[:120]}{'…' if len(last_prompt) > 120 else ''}[/dim]"
                    )
                    if last_op == "fix":
                        console.print(Panel("[yellow]Analyzing error…[/yellow]", border_style="yellow"))
                        response = agent.fix_error(last_prompt)
                        edits = patcher.parse(response)
                        if edits:
                            patcher.apply(edits, response=response, op_type="fix")
                        else:
                            console.print("[dim](No file edits detected.)[/dim]")
                    elif last_op == "feature":
                        console.print(Panel("[green]Planning feature…[/green]", border_style="green"))
                        response = agent.add_feature(last_prompt)
                        edits = patcher.parse(response)
                        if edits:
                            patcher.apply(edits, response=response, op_type="feature")
                        else:
                            console.print("[dim](No file edits detected.)[/dim]")
                    else:
                        response = agent.chat(last_prompt)
                        edits = patcher.parse(response)
                        if edits:
                            console.print(f"\n[yellow]{len(edits)} file change(s) detected.[/yellow]")
                            patcher.apply(edits, response=response, op_type="chat")

            # ── /fix ────────────────────────────────────────────────────────
            elif low == "/fix" or low.startswith("/fix "):
                error = cmd[4:].strip() if low.startswith("/fix ") else ""
                if not error:
                    error = collect_multiline("paste the Godot error / Output log", session)
                if not error.strip():
                    console.print("[dim]Nothing to fix.[/dim]")
                    continue

                # ── Smart routing: detect config/feature requests vs real errors ──
                # A real Godot error contains "ERROR:", "SCRIPT ERROR:", a res:// path,
                # or a stack trace marker.
                _REAL_ERROR_RE = re.compile(
                    r'(SCRIPT ERROR|ERROR|USER ERROR|WARNING):|res://[\w/._-]+\.gd:\d+|'
                    r'\bat:\s+[\w._]+|GDScript backtrace',
                    re.IGNORECASE
                )
                if not _REAL_ERROR_RE.search(error):
                    console.print(
                        "[yellow]ℹ  This doesn't look like a Godot error log.[/yellow]\n"
                        "[dim]  /fix is designed for error messages from the Output panel.\n"
                        "  For config changes and new features, use [bold]/feature[/bold].\n"
                        "  Routing to /feature automatically…[/dim]"
                    )
                    last_prompt = error
                    last_op     = "feature"
                    logger.log_user("feature (auto-routed from /fix)", error)
                    console.print(Panel("[green]Planning change…[/green]", border_style="green"))
                    response = agent.add_feature(error)
                    edits    = patcher.parse(response)
                    if edits:
                        patcher.apply(edits, response=response, op_type="feature")
                    else:
                        console.print("[dim](No file edits detected.)[/dim]")
                        setup = Patcher.extract_setup_steps(response)
                        if setup:
                            console.print(Panel(setup,
                                title="[bold yellow]⚙  Setup Steps[/bold yellow]",
                                border_style="yellow"))
                    continue

                last_errors = error
                last_prompt = error
                last_op     = "fix"
                logger.log_user("fix", error)
                # Parse any @file mentions — they bypass triage and go straight to load
                clean_error, mentioned = _parse_at_mentions(error)
                if mentioned:
                    console.print(f"[dim]📎 Pinned files: {', '.join(mentioned)}[/dim]")
                console.print(Panel("[yellow]Analyzing…[/yellow]", border_style="yellow"))
                response = agent.fix_error(clean_error, force_files=mentioned or None)
                edits    = patcher.parse(response)
                if edits:
                    patcher.apply(edits, response=response, op_type="fix")
                else:
                    console.print("[dim](No file edits detected.)[/dim]")
                    setup = Patcher.extract_setup_steps(response)
                    if setup:
                        console.print(Panel(setup,
                            title="[bold yellow]⚙  Setup Steps[/bold yellow]",
                            border_style="yellow"))

            # ── /feature ────────────────────────────────────────────────────
            elif low == "/feature" or low.startswith("/feature "):
                feat = cmd[8:].strip() if low.startswith("/feature ") else ""
                if not feat:
                    feat = collect_multiline("describe the feature you want", session)
                if not feat.strip():
                    console.print("[dim]Nothing described.[/dim]")
                    continue
                last_prompt = feat
                last_op     = "feature"
                logger.log_user("feature", feat)
                clean_feat, mentioned = _parse_at_mentions(feat)
                if mentioned:
                    console.print(f"[dim]📎 Pinned files: {', '.join(mentioned)}[/dim]")
                console.print(Panel("[green]Planning feature…[/green]", border_style="green"))
                response = agent.add_feature(clean_feat, force_files=mentioned or None)
                edits    = patcher.parse(response)
                if edits:
                    patcher.apply(edits, response=response, op_type="feature")
                else:
                    console.print("[dim](No file edits detected.)[/dim]")
                    setup = Patcher.extract_setup_steps(response)
                    if setup:
                        console.print(Panel(setup,
                            title="[bold yellow]⚙  Setup Steps[/bold yellow]",
                            border_style="yellow"))

            # ── /chat ────────────────────────────────────────────────────────
            elif low == "/chat" or low.startswith("/chat "):
                question = cmd[5:].strip() if low.startswith("/chat ") else ""
                if not question:
                    question = collect_multiline(
                        "ask a question about Godot or your project", session
                    )
                if not question.strip():
                    console.print("[dim]Nothing to ask.[/dim]")
                    continue
                last_prompt = question
                last_op     = "chat"
                logger.log_user("chat", question)
                response = agent.chat(question)
                # Chat can still return code blocks — apply them if the user wants
                edits = patcher.parse(response)
                if edits:
                    console.print(f"\n[yellow]{len(edits)} file change(s) suggested.[/yellow]")
                    patcher.apply(edits, response=response, op_type="chat")
                else:
                    setup = Patcher.extract_setup_steps(response)
                    if setup:
                        console.print(Panel(setup,
                            title="[bold yellow]⚙  Setup Steps[/bold yellow]",
                            border_style="yellow"))

            # ── unknown input ────────────────────────────────────────────────
            else:
                if cmd.startswith("/"):
                    console.print(
                        f"[yellow]Unknown command:[/yellow] [white]{cmd}[/white]\n"
                        "[dim]Type [bold]/help[/bold] for the full command list.[/dim]"
                    )
                else:
                    console.print(
                        "[dim]Unrecognised input. Use a command:\n"
                        "  /fix      → paste an error\n"
                        "  /feature  → describe a feature\n"
                        "  /chat     → ask a question\n"
                        "  /help     → full list[/dim]"
                    )

    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        if log_watcher:
            log_watcher.stop()
        if godot_runner:
            godot_runner.stop()
        if git.enabled and git.branch:
            console.print(
                f"\n[dim]Session branch: [cyan]{git.branch}[/cyan]\n"
                f"  git checkout main                    # abandon all AI changes\n"
                f"  git diff main {git.branch}           # review all changes[/dim]"
            )
        console.print("\n[dim]See you next session! 🎮[/dim]")


def _handle_models(agent: GodotAgent) -> None:
    available = fetch_available_models(agent.client)
    t = Table(title="Available Models", border_style="cyan")
    t.add_column("#",     style="dim", width=3)
    t.add_column("Model", style="white")
    t.add_column("Role",  style="yellow")
    for i, m in enumerate(available, 1):
        role = ""
        if m == agent.code_model: role = "● code"
        if m == agent.chat_model: role += (" / " if role else "") + "● chat"
        if m == Config.EMBED_MODEL: role = "embeddings"
        t.add_row(str(i), m, role)
    console.print(t)
    console.print(f"\n[cyan]Code model:[/cyan] {agent.code_model}")
    console.print(f"[cyan]Chat model:[/cyan] {agent.chat_model}")
    which = Prompt.ask("\nChange which?",
                       choices=["code", "chat", "both", "none"], default="none")
    if which in ("code", "both") and available:
        pick = Prompt.ask("Code model (name or #)", default=agent.code_model)
        if pick.isdigit() and 1 <= int(pick) <= len(available):
            pick = available[int(pick) - 1]
        agent._code_model = pick
        console.print(f"[green]✓ Code model → {pick}[/green]")
    if which in ("chat", "both") and available:
        pick = Prompt.ask("Chat model (name or #)", default=agent.chat_model)
        if pick.isdigit() and 1 <= int(pick) <= len(available):
            pick = available[int(pick) - 1]
        agent._chat_model = pick
        console.print(f"[green]✓ Chat model → {pick}[/green]")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="godot_ai",
        description="Codebase-aware AI assistant for Godot 4 (v4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python godot_ai.py
  python godot_ai.py C:/Users/you/Documents/MyGame
  python godot_ai.py ~/games/mygame --host http://192.168.2.49:11434
  python godot_ai.py . --no-git
  python godot_ai.py . --model qwen2.5-coder:32b
  OLLAMA_HOST=http://192.168.2.49:11434 python godot_ai.py ./my-game
""",
    )
    parser.add_argument("project", nargs="?", default=None)
    parser.add_argument("--host", default=None, metavar="URL")
    parser.add_argument("--no-git", action="store_true",
                        help="Disable git safety net")
    parser.add_argument("--model", default=None, metavar="NAME",
                        help="Override the code model (e.g. qwen2.5-coder:32b)")
    args = parser.parse_args()

    if args.model:
        Config.CODE_MODEL = args.model

    project_path = args.project
    if not project_path:
        project_path = Prompt.ask("📁 Godot project path", default=".")
    project_path = str(Path(project_path).resolve())

    if not Path(project_path).exists():
        console.print(f"[red]Path not found: {project_path}[/red]")
        sys.exit(1)

    if not list(Path(project_path).glob("project.godot")):
        console.print(f"[yellow]⚠  No project.godot found in {project_path}[/yellow]")
        if not Confirm.ask("Continue anyway?", default=True):
            sys.exit(0)

    if args.host:
        ollama_host = args.host
    elif os.environ.get("OLLAMA_HOST"):
        ollama_host = os.environ["OLLAMA_HOST"]
    else:
        ollama_host = Prompt.ask("🖥  Ollama host", default="http://localhost:11434")

    repl(project_path, ollama_host, git_enabled=not args.no_git)


if __name__ == "__main__":
    main()