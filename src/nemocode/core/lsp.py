# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""LSP client — connects to language servers for code intelligence.

Auto-detects language servers by file extension, communicates over
stdio using the JSON-RPC / LSP protocol with Content-Length headers.
Provides async API for diagnostics, hover, go-to-definition, and
find-references.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Location:
    """A source location returned by definition / references."""

    uri: str
    line: int
    col: int
    end_line: int = 0
    end_col: int = 0

    @property
    def path(self) -> str:
        """Convert file:// URI to a local path."""
        if self.uri.startswith("file://"):
            return self.uri[len("file://"):]
        return self.uri


@dataclass
class Diagnostic:
    """A single diagnostic (error / warning / info)."""

    message: str
    severity: int = 1  # 1=Error, 2=Warning, 3=Info, 4=Hint
    line: int = 0
    col: int = 0
    end_line: int = 0
    end_col: int = 0
    source: str = ""
    code: str | int = ""

    @property
    def severity_label(self) -> str:
        return {1: "error", 2: "warning", 3: "info", 4: "hint"}.get(
            self.severity, "unknown"
        )


# ---------------------------------------------------------------------------
# Language server detection
# ---------------------------------------------------------------------------

# Mapping from language id to (binary name, extra args)
_SERVER_MAP: dict[str, tuple[str, list[str]]] = {
    "python": ("pyright-langserver", ["--stdio"]),
    "python-pylsp": ("pylsp", []),
    "typescript": ("typescript-language-server", ["--stdio"]),
    "javascript": ("typescript-language-server", ["--stdio"]),
    "rust": ("rust-analyzer", []),
    "go": ("gopls", ["serve"]),
    "c": ("clangd", []),
    "cpp": ("clangd", []),
}

# File extension to language id
_EXT_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
}


def detect_language(file_path: str) -> str | None:
    """Return a language id from a file extension, or None if unknown."""
    ext = Path(file_path).suffix.lower()
    return _EXT_MAP.get(ext)


def find_server(language: str) -> tuple[str, list[str]] | None:
    """Find the binary + args for a language server.

    Falls back from pyright to pylsp for Python if pyright is absent.
    Returns None if no server is found on PATH.
    """
    candidates: list[str] = [language]
    if language == "python":
        candidates.append("python-pylsp")

    for lang_key in candidates:
        entry = _SERVER_MAP.get(lang_key)
        if entry is None:
            continue
        binary, args = entry
        if shutil.which(binary):
            return binary, args
    return None


# ---------------------------------------------------------------------------
# JSON-RPC transport (stdio, Content-Length framing)
# ---------------------------------------------------------------------------


class _JsonRpcTransport:
    """Async JSON-RPC transport over stdin/stdout of a subprocess."""

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._seq = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._notifications: dict[str, list[dict[str, Any]]] = {}

    async def start(self, cmd: list[str], cwd: str | None = None) -> None:
        env = os.environ.copy()
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        self._reader_task = asyncio.create_task(self._read_loop())

    async def stop(self) -> None:
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._proc:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except (ProcessLookupError, asyncio.TimeoutError):
                self._proc.kill()
            self._proc = None
        # Cancel any pending futures
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    # -- send / request ------------------------------------------------

    def _next_id(self) -> int:
        self._seq += 1
        return self._seq

    async def send_request(
        self, method: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Send a JSON-RPC request and wait for the response."""
        if not self.is_running:
            raise RuntimeError("LSP server is not running")
        msg_id = self._next_id()
        body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
        }
        if params is not None:
            body["params"] = params

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[Any] = loop.create_future()
        self._pending[msg_id] = fut

        self._write(body)
        try:
            return await asyncio.wait_for(fut, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending.pop(msg_id, None)
            raise

    async def send_notification(
        self, method: str, params: dict[str, Any] | None = None
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        if not self.is_running:
            raise RuntimeError("LSP server is not running")
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            body["params"] = params
        self._write(body)

    def _write(self, body: dict[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        encoded = json.dumps(body).encode("utf-8")
        header = f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii")
        self._proc.stdin.write(header + encoded)

    # -- read loop -----------------------------------------------------

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        reader = self._proc.stdout
        try:
            while True:
                content_length = await self._read_headers(reader)
                if content_length <= 0:
                    break
                data = await reader.readexactly(content_length)
                msg = json.loads(data.decode("utf-8"))
                self._dispatch(msg)
        except (asyncio.CancelledError, asyncio.IncompleteReadError):
            pass
        except Exception:
            logger.debug("LSP read loop error", exc_info=True)

    async def _read_headers(self, reader: asyncio.StreamReader) -> int:
        """Read LSP headers and return Content-Length."""
        content_length = 0
        while True:
            line = await reader.readline()
            if not line:
                return -1  # EOF
            decoded = line.decode("ascii", errors="replace").strip()
            if not decoded:
                break  # empty line signals end of headers
            if decoded.lower().startswith("content-length:"):
                content_length = int(decoded.split(":", 1)[1].strip())
        return content_length

    def _dispatch(self, msg: dict[str, Any]) -> None:
        if "id" in msg and "method" not in msg:
            # Response to a request
            msg_id = msg["id"]
            fut = self._pending.pop(msg_id, None)
            if fut and not fut.done():
                if "error" in msg:
                    fut.set_exception(
                        RuntimeError(
                            f"LSP error {msg['error'].get('code')}: "
                            f"{msg['error'].get('message', '')}"
                        )
                    )
                else:
                    fut.set_result(msg.get("result"))
        elif "method" in msg and "id" not in msg:
            # Server notification
            method = msg["method"]
            self._notifications.setdefault(method, []).append(msg)
        # else: server request — we ignore these for now

    def pop_notifications(self, method: str) -> list[dict[str, Any]]:
        return self._notifications.pop(method, [])


# ---------------------------------------------------------------------------
# High-level LSP client
# ---------------------------------------------------------------------------


@dataclass
class _OpenFile:
    uri: str
    language_id: str
    version: int
    text: str


class LSPClient:
    """Async LSP client that wraps a single language server process."""

    def __init__(self) -> None:
        self._transport = _JsonRpcTransport()
        self._language: str = ""
        self._root_uri: str = ""
        self._open_files: dict[str, _OpenFile] = {}
        self._server_capabilities: dict[str, Any] = {}

    # -- lifecycle -----------------------------------------------------

    async def start(self, root_path: str, language: str) -> None:
        """Start the language server for *language* rooted at *root_path*.

        Raises ``RuntimeError`` if no suitable language server binary is
        found on PATH.
        """
        server = find_server(language)
        if server is None:
            raise RuntimeError(
                f"No language server found for {language!r}. "
                "Install the appropriate server binary."
            )
        binary, args = server
        cmd = [binary, *args]
        self._language = language
        self._root_uri = Path(root_path).resolve().as_uri()

        await self._transport.start(cmd, cwd=root_path)
        await self._initialize()

    async def stop(self) -> None:
        """Shutdown the language server gracefully."""
        if self._transport.is_running:
            try:
                await self._transport.send_request("shutdown")
                await self._transport.send_notification("exit")
            except Exception:
                logger.debug("LSP shutdown error", exc_info=True)
        await self._transport.stop()
        self._open_files.clear()

    @property
    def is_running(self) -> bool:
        return self._transport.is_running

    # -- LSP initialize ------------------------------------------------

    async def _initialize(self) -> None:
        params: dict[str, Any] = {
            "processId": os.getpid(),
            "rootUri": self._root_uri,
            "capabilities": {
                "textDocument": {
                    "synchronization": {
                        "dynamicRegistration": False,
                        "willSave": False,
                        "didSave": True,
                        "willSaveWaitUntil": False,
                    },
                    "hover": {
                        "dynamicRegistration": False,
                        "contentFormat": ["plaintext", "markdown"],
                    },
                    "definition": {"dynamicRegistration": False},
                    "references": {"dynamicRegistration": False},
                    "diagnostic": {"dynamicRegistration": False},
                    "publishDiagnostics": {"relatedInformation": True},
                },
                "workspace": {
                    "workspaceFolders": True,
                },
            },
            "workspaceFolders": [
                {"uri": self._root_uri, "name": Path(self._root_uri).name}
            ],
        }
        result = await self._transport.send_request("initialize", params)
        self._server_capabilities = result.get("capabilities", {}) if result else {}
        await self._transport.send_notification("initialized", {})

    # -- document sync -------------------------------------------------

    async def _ensure_open(self, file_path: str) -> _OpenFile:
        """Open a file in the language server if not already open."""
        abs_path = str(Path(file_path).resolve())
        if abs_path in self._open_files:
            return self._open_files[abs_path]

        text = Path(abs_path).read_text(errors="replace")
        uri = Path(abs_path).as_uri()
        lang_id = detect_language(abs_path) or self._language
        of = _OpenFile(uri=uri, language_id=lang_id, version=1, text=text)
        self._open_files[abs_path] = of

        await self._transport.send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": lang_id,
                    "version": of.version,
                    "text": text,
                }
            },
        )
        return of

    async def notify_change(self, file_path: str, new_text: str) -> None:
        """Notify the server about a full document change."""
        abs_path = str(Path(file_path).resolve())
        of = self._open_files.get(abs_path)
        if of is None:
            of = await self._ensure_open(file_path)

        of.version += 1
        of.text = new_text
        await self._transport.send_notification(
            "textDocument/didChange",
            {
                "textDocument": {"uri": of.uri, "version": of.version},
                "contentChanges": [{"text": new_text}],
            },
        )

    # -- diagnostics ---------------------------------------------------

    async def get_diagnostics(self, file_path: str) -> list[Diagnostic]:
        """Pull diagnostics for a file.

        Tries ``textDocument/diagnostic`` first; falls back to cached
        ``textDocument/publishDiagnostics`` notifications from the server.
        """
        of = await self._ensure_open(file_path)

        # Try pull diagnostics (LSP 3.17+)
        try:
            result = await self._transport.send_request(
                "textDocument/diagnostic",
                {"textDocument": {"uri": of.uri}},
            )
            if result and "items" in result:
                return [self._parse_diagnostic(d) for d in result["items"]]
        except RuntimeError:
            pass  # Server may not support pull diagnostics

        # Fall back to pushed publishDiagnostics notifications
        notes = self._transport.pop_notifications(
            "textDocument/publishDiagnostics"
        )
        diagnostics: list[Diagnostic] = []
        for note in notes:
            params = note.get("params", {})
            if params.get("uri") == of.uri:
                for d in params.get("diagnostics", []):
                    diagnostics.append(self._parse_diagnostic(d))
        return diagnostics

    # -- hover ---------------------------------------------------------

    async def get_hover(self, file_path: str, line: int, col: int) -> str:
        """Return hover information (as plain text) for a position."""
        of = await self._ensure_open(file_path)
        result = await self._transport.send_request(
            "textDocument/hover",
            {
                "textDocument": {"uri": of.uri},
                "position": {"line": line, "character": col},
            },
        )
        if not result:
            return ""
        contents = result.get("contents", "")
        return self._extract_hover_text(contents)

    # -- definition ----------------------------------------------------

    async def get_definition(
        self, file_path: str, line: int, col: int
    ) -> Location | None:
        """Go to definition for the symbol at the given position."""
        of = await self._ensure_open(file_path)
        result = await self._transport.send_request(
            "textDocument/definition",
            {
                "textDocument": {"uri": of.uri},
                "position": {"line": line, "character": col},
            },
        )
        locs = self._parse_locations(result)
        return locs[0] if locs else None

    # -- references ----------------------------------------------------

    async def get_references(
        self, file_path: str, line: int, col: int
    ) -> list[Location]:
        """Find all references to the symbol at the given position."""
        of = await self._ensure_open(file_path)
        result = await self._transport.send_request(
            "textDocument/references",
            {
                "textDocument": {"uri": of.uri},
                "position": {"line": line, "character": col},
                "context": {"includeDeclaration": True},
            },
        )
        return self._parse_locations(result)

    # -- parse helpers -------------------------------------------------

    @staticmethod
    def _parse_diagnostic(d: dict[str, Any]) -> Diagnostic:
        rng = d.get("range", {})
        start = rng.get("start", {})
        end = rng.get("end", {})
        return Diagnostic(
            message=d.get("message", ""),
            severity=d.get("severity", 1),
            line=start.get("line", 0),
            col=start.get("character", 0),
            end_line=end.get("line", 0),
            end_col=end.get("character", 0),
            source=d.get("source", ""),
            code=d.get("code", ""),
        )

    @staticmethod
    def _parse_locations(result: Any) -> list[Location]:
        if result is None:
            return []
        # Can be a single Location, a list of Locations, or a list of LocationLinks
        items = result if isinstance(result, list) else [result]
        locations: list[Location] = []
        for item in items:
            if "targetUri" in item:
                # LocationLink
                rng = item.get("targetSelectionRange", item.get("targetRange", {}))
                start = rng.get("start", {})
                end = rng.get("end", {})
                locations.append(
                    Location(
                        uri=item["targetUri"],
                        line=start.get("line", 0),
                        col=start.get("character", 0),
                        end_line=end.get("line", 0),
                        end_col=end.get("character", 0),
                    )
                )
            elif "uri" in item:
                rng = item.get("range", {})
                start = rng.get("start", {})
                end = rng.get("end", {})
                locations.append(
                    Location(
                        uri=item["uri"],
                        line=start.get("line", 0),
                        col=start.get("character", 0),
                        end_line=end.get("line", 0),
                        end_col=end.get("character", 0),
                    )
                )
        return locations

    @staticmethod
    def _extract_hover_text(contents: Any) -> str:
        """Normalise hover contents to a plain string."""
        if isinstance(contents, str):
            return contents
        if isinstance(contents, dict):
            # MarkupContent: {"kind": "...", "value": "..."}
            return contents.get("value", "")
        if isinstance(contents, list):
            parts: list[str] = []
            for item in contents:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(item.get("value", ""))
            return "\n".join(parts)
        return str(contents)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# Cache of running clients keyed by (root_path, language)
_active_clients: dict[tuple[str, str], LSPClient] = {}


async def get_client(root_path: str, language: str) -> LSPClient:
    """Return (or create) a cached LSPClient for the given root+language."""
    key = (str(Path(root_path).resolve()), language)
    client = _active_clients.get(key)
    if client and client.is_running:
        return client
    client = LSPClient()
    await client.start(root_path, language)
    _active_clients[key] = client
    return client


async def shutdown_all() -> None:
    """Stop all cached LSP clients."""
    for client in _active_clients.values():
        await client.stop()
    _active_clients.clear()
