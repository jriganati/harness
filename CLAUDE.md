# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About this workspace

This is a general-purpose Claude Code harness directory — not a software project. It serves as a home base for tasks that are not scoped to a specific codebase (research, scripting, exploration, cross-project work, etc.).

There is no build system, test suite, or application code here. The `.claude/memory/` subdirectory holds project-scoped memory files for this workspace.

## Configuration

Global settings live in `~/.claude/settings.json`. This workspace inherits those defaults:
- Default permission mode: `auto`
- Auto-compact threshold: 80% context usage
- Effort level: `high`
