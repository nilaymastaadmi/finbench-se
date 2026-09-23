"""Judge transport: Claude Haiku 4.5 through the authenticated Claude Code CLI (AMENDMENTS.md A5).

Same invocation as market-query-agent's ClaudeCLIProvider: headless print mode, JSON output, every tool
and MCP server disabled, one turn. The CLI adds its own ~20k-token system prompt; that overhead is the
transport's, not the judge's, and does not change what the judge sees in the user turn.
"""
import json
import subprocess
import sys
import time

MODEL = "claude-haiku-4-5-20251001"
SYSTEM = "You grade and compare answers. Reply with JSON only, no prose, no code fences."
NO_TOOLS = ["Bash", "Edit", "Write", "Read", "Glob", "Grep", "WebFetch", "WebSearch", "Task",
            "NotebookEdit", "TodoWrite", "Skill", "Agent"]


def generate(prompt, tries=4):
    cmd = ["claude", "-p", "--output-format", "json", "--model", MODEL, "--system-prompt", SYSTEM,
           "--disallowed-tools", *NO_TOOLS, "--disable-slash-commands", "--strict-mcp-config", "--max-turns", "1"]
    for attempt in range(tries):
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=240)
        try:
            payload = json.loads(proc.stdout)
            if proc.returncode == 0 and not payload.get("is_error") and payload.get("result"):
                return payload["result"].strip()
            detail = f"exit {proc.returncode} subtype={payload.get('subtype')} error={payload.get('is_error')}"
        except json.JSONDecodeError:
            detail = f"exit {proc.returncode} stdout={proc.stdout[:120]!r} stderr={proc.stderr[:120]!r}"
        print(f"[claude] {detail}, retry {attempt + 1}", file=sys.stderr, flush=True)
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"claude CLI failed after {tries} tries: {detail}")
