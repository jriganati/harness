"""Sync wrapper around an MCP stdio server (e.g. Playwright MCP).

The MCP Python SDK is async-first, but our executor loop is sync. This module
runs a persistent event loop in a background thread that owns the MCP session,
and exposes blocking `list_tools()` and `call_tool()` methods that dispatch
into that loop via `run_coroutine_threadsafe`.

One process-wide instance is held by module-level helpers `get_client()` /
`shutdown()`. The Playwright browser stays open for the entire harness run, so
the executor can navigate once and reuse the page across many tool calls.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self, command: str, args: list[str]):
        self.command = command
        self.args = args
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._stdio_ctx = None
        self._session_ctx = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None

    # --------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="mcp-client")
        self._thread.start()
        self._ready.wait(timeout=30)
        if self._startup_error is not None:
            raise RuntimeError(f"MCP server failed to start: {self._startup_error}") from self._startup_error
        if not self._ready.is_set():
            raise RuntimeError("MCP server startup timed out after 30s")

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._disconnect(), self._loop)
            try:
                fut.result(timeout=5)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
        except BaseException as e:
            self._startup_error = e
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()

    async def _connect(self) -> None:
        params = StdioServerParameters(command=self.command, args=self.args)
        self._stdio_ctx = stdio_client(params)
        read, write = await self._stdio_ctx.__aenter__()
        self._session_ctx = ClientSession(read, write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()

    async def _disconnect(self) -> None:
        if self._session_ctx:
            try:
                await self._session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
        if self._stdio_ctx:
            try:
                await self._stdio_ctx.__aexit__(None, None, None)
            except Exception:
                pass

    # ------------------------------------------------------------- API

    def list_tools(self) -> list[dict]:
        """Return tools converted to OpenAI tool-call schema."""
        assert self._loop and self._session
        fut = asyncio.run_coroutine_threadsafe(self._session.list_tools(), self._loop)
        result = fut.result(timeout=30)
        return [
            {
                "type": "function",
                "function": {
                    "name":        t.name,
                    "description": t.description or "",
                    "parameters":  t.inputSchema or {"type": "object", "properties": {}},
                },
            }
            for t in result.tools
        ]

    def tool_names(self) -> set[str]:
        return {t["function"]["name"] for t in self.list_tools()}

    def call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Call an MCP tool and return its content as a plain string."""
        assert self._loop and self._session
        fut = asyncio.run_coroutine_threadsafe(self._session.call_tool(name, args), self._loop)
        result = fut.result(timeout=60)

        if result.isError:
            return f"error: {_content_to_text(result.content)}"
        return _content_to_text(result.content)


def _content_to_text(content) -> str:
    """Flatten MCP content blocks to a single string (skips images/binaries)."""
    parts: list[str] = []
    for item in content or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(f"<{type(item).__name__}>")
    return "\n".join(parts)[:6000]  # cap to avoid blowing up the executor context


# ----------------------------------------------------------- singleton

_client: MCPClient | None = None


def get_client(command: str, args: list[str]) -> MCPClient:
    """Return the process-wide MCPClient, starting it on first access."""
    global _client
    if _client is None:
        _client = MCPClient(command, args)
        _client.start()
    return _client


def shutdown() -> None:
    global _client
    if _client is not None:
        _client.stop()
        _client = None
