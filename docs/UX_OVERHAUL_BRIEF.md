# NeMoCode UX Overhaul Brief

**Goal:** Make NeMoCode's terminal output feel polished, readable, and delightful. Right now it's functional but janky — the thinking spinner vanishes instantly, tool calls are a wall of dim text, the AI response has no clear start, and there's no sense of "flow" from thinking → working → responding. Fix all of it.

**Target vibe:** Claude Code's output, but NVIDIA-green-themed. The user should always know what the AI is doing, never feel lost in noise, and the actual response should feel like the main event.

---

## Current Architecture

All rendering lives in two files:

### `src/nemocode/cli/render.py` — Event renderer
- `EventRenderer` class receives `AgentEvent` objects one at a time
- Event kinds: `text`, `thinking`, `tool_call`, `tool_result`, `phase`, `error`, `usage`
- Tool calls print a dim line with `▸`, tool results append inline
- Text streams via Rich `Live` + `Markdown` in interactive mode
- Falls back to plain `print()` in pipes/tests

### `src/nemocode/cli/commands/repl.py` — REPL loop
- `_run_turn()` (line ~1213) creates a "Thinking…" spinner before events start
- `_TurnRenderer` wraps `EventRenderer` and tracks metrics
- The spinner is killed on the FIRST event of any kind (text, thinking, tool_call, phase, error)
- After the turn, `_format_context_usage()` shows token/cost info

### Event flow for a typical turn:
```
User types: "add error handling to the parser"
  1. Thinking spinner starts ("Tensor cores warming up…")
  2. First event arrives (usually tool_call for read_file) — spinner DIES
  3. Multiple tool_call/tool_result pairs stream in
  4. text events stream in (the actual AI response)
  5. Turn ends, context usage shown
```

---

## Problems to Fix

### P1: Thinking spinner is pointless
The spinner shows for ~200ms then dies when the first tool_call arrives. The user never sees it. It needs to persist through the tool-gathering phase and only disappear when the AI starts its text response.

**Fix:** Don't kill the spinner on `tool_call` events. Only kill it on `text` (the actual response starting). Better yet, UPDATE the spinner text as tools execute so the user sees what's happening: "Reading scheduler.py…" → "Searching for patterns…" → "Running tests…"

### P2: No sense of "phases" in single-model mode
Formations show phase labels (`━━ executor ▸ Executing... ━━`) but single-model mode (the default) has nothing. The user sees: spinner → wall of tool calls → text. There's no "now I'm gathering info" → "now I'm responding" transition.

**Fix:** Add implicit phases even in single-model mode. When tools start, show a subtle "working" indicator. When text starts after tools, show a clear transition.

### P3: Tool calls are still noisy
Even with inline results, 10+ tool calls in a row create a wall of dim text that obscures the response. For information-gathering turns (model reads 5 files, searches 3 patterns), the user doesn't need to see each one individually.

**Fix ideas:**
- **Collapse read-only tools into a single summary line** after they complete: `▸ Read 5 files, searched 3 patterns, checked git status` instead of 9 separate lines
- **Or** use a progress-style display: show the current tool on ONE line that updates in-place (like a spinner), then show a summary when done
- **Or** hide read-only tools entirely and only show mutations (writes, edits, bash commands)

### P4: The AI's text response has no visual weight
After a block of dim tool calls, the AI's markdown response just... starts. There's no visual cue that "this is the answer." In Claude Code, the response has clear visual separation.

**Fix:** Add a subtle but clear separator before the response. Ideas:
- A thin green line: `───────────────────`
- A role label: `▸ nemotron` or just a blank line + un-dimmed text
- The first line of the response could be slightly bolder

### P5: Context/cost info after each turn is distracting
`[$0.0023 session]` or `[Context: 15K / 1M tokens (1.4%)]` appears after every turn. It's useful info but it breaks the conversational flow.

**Fix:** Only show when interesting (>10% context used, or cost > $0.01). Or put it in the toolbar instead.

### P6: The banner takes too much vertical space
The ASCII art + 8 lines of info + 3 lines of tips = 15+ lines before the first prompt. On a small terminal this pushes the actual interaction offscreen.

**Fix:** Compact it. One line for model info, one for directory, one for tips. Save the ASCII art for `--verbose` or first-ever run.

### P7: Multi-line tool results still dump too much
Bash commands that succeed with output show up to 5 lines. For things like `pytest` output or `git log`, even 5 lines is too much during a multi-tool turn.

**Fix:** During a multi-tool sequence (before the text response), cap at 1-2 lines for non-error results. Show full output only for errors or when it's the only tool call.

### P8: No indication of how long things are taking
The user types a message and... waits. No progress. Is it stuck? Is it thinking? Is it executing tools? The elapsed time only shows after each tool completes.

**Fix:** Show a running elapsed timer somewhere. Could be in the spinner text, could be in the toolbar.

### P9: Errors are too abrupt
`Error: something broke` in bold red with no context. The user doesn't know if it's recoverable, if the session is dead, or what to do.

**Fix:** Add recovery hints. "Error: API rate limit exceeded. Retry in 30s or switch endpoint with /endpoint."

---

## Technical Constraints

- Must work in non-interactive mode (pipes, tests) — check `self._interactive`
- Rich `Live` is used for streaming markdown — can't nest `Live` inside `Status`
- The `EventRenderer` gets events ONE AT A TIME — can't look ahead
- But you CAN buffer events and render them in batches (just store them in a list and flush periodically)
- Tests in `tests/test_render.py` use `no_color=True` consoles — don't depend on ANSI codes in assertions
- The toolbar in `_InputReader._toolbar()` updates on each keypress — good place for persistent status info

## Files to Modify

1. **`src/nemocode/cli/render.py`** — The main rendering engine. Most changes go here.
2. **`src/nemocode/cli/commands/repl.py`** — The thinking spinner, turn wrapper, banner, context display, toolbar.
3. **`tests/test_render.py`** — Update tests for any rendering format changes.

## What NOT to Change

- The `AgentEvent` dataclass and event kinds
- The `Scheduler` or `CodeAgent` logic
- Tool implementations
- Config schema
- The prompt_toolkit input handling

## Style Guide

- NVIDIA green = `bright_green` (Rich style name), stored as `_NV_GREEN` constant in render.py
- Dim for background work, bright for user-facing content
- Use Unicode box-drawing and arrows sparingly: `▸ ━ ─ ✓ ✗`
- Don't use emoji
- Keep it readable on both dark and light terminal backgrounds
- Test with `python -m pytest tests/ -q` after every change

## Reference: What "Good" Looks Like

```
code > add error handling to the parser

  Reasoning at GPU speed…
  ▸ Read parser.py, utils.py, tests/test_parser.py
  ▸ Search /raise|except/ in src/
  ▸ git diff (clean)

  The parser currently doesn't handle malformed input gracefully.
  Here's what I'll change:

  1. Add a `ParseError` exception class...
  [rest of markdown response streams naturally]

  ▸ Edit src/parser.py ✓
    +class ParseError(Exception):
    +    ...
  ▸ Edit src/utils.py ✓
  ▸ pytest tests/ ✓ (12 passed)

  All changes applied. The parser now raises `ParseError` with
  line number context for malformed input.

[ctx: 25K/1M · $0.003]
```

Note how:
- The thinking indicator persists through the reading phase
- Read-only tools are collapsed into one summary line
- The text response starts clearly after a blank line
- Mutations show inline with diffs
- The whole turn reads like a narrative, not a log dump

---

## Verification

After changes:
1. `python -m pytest tests/ -q` — all pass
2. `python -m ruff check src/nemocode/ tests/` — no lint errors
3. `pip install -e . && nemo` — visual inspection in real terminal
4. Test in pipe mode: `echo "hello" | nemo code -p "explain this"` — no crashes
5. Test non-interactive: rendering should degrade gracefully
</content>
</invoke>