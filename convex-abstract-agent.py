#!/usr/bin/env python3
"""
Convex → Abstract Backend Layer Agent
Runs in GitHub Actions to wrap all Convex-specific code
in a clean provider-agnostic abstraction layer.

Supports checkpoint/resume: after every batch the generated files are committed
to the target repo so a timed-out run can be continued without losing work.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import requests

# ─── CONFIG ───────────────────────────────────────────────────────────────────

ANTHROPIC_API_URL     = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL       = "claude-sonnet-4-6"
MAX_TOKENS            = 8000
REQUEST_DELAY_SECONDS = 1.5
BATCH_SIZE            = 5

# Stop this many seconds before the job hard-timeout so we can do a final commit
SAFETY_BUFFER_SECONDS = 120

IGNORE_DIRS = {
    "node_modules", ".git", ".expo", "dist", "build",
    ".next", "__pycache__", ".convex", ".cache", "android", "ios",
}
IGNORE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".ttf", ".otf",
    ".woff", ".woff2", ".mp4", ".mp3", ".zip", ".tar", ".gz",
}
CODE_EXTENSIONS = {
    ".ts", ".tsx", ".js", ".jsx", ".json", ".md",
    ".env", ".example", ".sql",
}

CHECKPOINT_FILE = ".abstraction-checkpoint.json"


# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert React Native / TypeScript architect specialising in creating
clean, provider-agnostic backend abstraction layers.

## Goal
Take existing code that calls Convex directly and wrap every backend interaction behind a thin,
replaceable abstraction layer. The original runtime behaviour must be preserved — only the
import paths and call sites change.

## Abstraction layer structure to create under `src/backend/`

```
src/backend/
  types.ts                 # Shared domain types / DTOs (no provider imports)
  IBackendProvider.ts      # The provider interface — one method per logical operation
  convex/
    ConvexProvider.ts      # Implements IBackendProvider using the existing Convex SDK calls
    index.ts               # export default new ConvexProvider()
  index.ts                 # re-exports the active provider
```

### IBackendProvider interface rules
- One async method per logical data operation.
- Real-time / subscription methods return `Unsubscribe = () => void`.
- Auth methods live on the same interface.
- Never import from 'convex/*' inside `IBackendProvider.ts` or `types.ts`.

### ConvexProvider rules
- Encapsulate ALL existing useQuery / useMutation / useAction / ConvexReactClient logic.
- Keep every original Convex call intact — just wrap it.

### Updating call sites
- Replace Convex imports with `import backend from 'src/backend'`
- Replace useQuery → useEffect calling backend method + local state
- Replace useMutation → async handler calling backend method
- Remove ConvexProvider wrappers

### What NOT to change
- Business logic, UI, styles, navigation.
- Files with zero backend references — skip entirely.
- package.json — keep all existing Convex packages.

## Output format

<FILE path="relative/path/to/file">
file content here
</FILE>

<DELETE path="relative/path/to/file" />

Only output files that actually need to change or be created."""


# ─── CHECKPOINT ───────────────────────────────────────────────────────────────

class Checkpoint:
    """
    Persists progress to CHECKPOINT_FILE inside the target repo.
    A resumed run reads this and skips already-completed steps/batches.
    """

    def __init__(self, target_root: Path, source_repo_url: str = ""):
        self.path        = target_root / CHECKPOINT_FILE
        self.target_root = target_root
        self._data: dict = {}
        self._load(source_repo_url)

    def _load(self, source_repo_url: str) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
                print(f"  ♻️  Checkpoint found — resuming previous run")
                print(f"      Started:           {self._data.get('started_at', '?')}")
                print(f"      Completed steps:   {self._data.get('completed_steps', [])}")
                print(f"      Completed batches: {self._data.get('completed_batches', [])}")
                return
            except Exception as e:
                print(f"  ⚠️  Unreadable checkpoint ({e}) — starting fresh")
        self._data = {
            "started_at":        _now(),
            "updated_at":        _now(),
            "source_repo":       source_repo_url,
            "completed_steps":   [],
            "completed_batches": [],
            "total_batches":     0,
        }

    def _save(self) -> None:
        self._data["updated_at"] = _now()
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def is_step_done(self, step: str) -> bool:
        return step in self._data.get("completed_steps", [])

    def mark_step_done(self, step: str) -> None:
        steps = self._data.setdefault("completed_steps", [])
        if step not in steps:
            steps.append(step)
        self._save()

    def is_batch_done(self, idx: int) -> bool:
        return idx in self._data.get("completed_batches", [])

    def mark_batch_done(self, idx: int) -> None:
        batches = self._data.setdefault("completed_batches", [])
        if idx not in batches:
            batches.append(idx)
        self._save()

    def set_total_batches(self, n: int) -> None:
        self._data["total_batches"] = n
        self._save()

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


# ─── TIME BUDGET ──────────────────────────────────────────────────────────────

class TimeBudget:
    """Stop processing before GitHub hard-kills the job."""

    def __init__(self, total_seconds: int, buffer: int = SAFETY_BUFFER_SECONDS):
        self._deadline = time.monotonic() + total_seconds - buffer

    def expired(self) -> bool:
        return time.monotonic() >= self._deadline

    def remaining(self) -> int:
        return max(0, int(self._deadline - time.monotonic()))


# ─── GIT HELPERS ──────────────────────────────────────────────────────────────

def git(args: list[str], cwd: Path) -> str:
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{r.stderr}")
    return r.stdout.strip()


def commit_and_push(target_root: Path, message: str) -> bool:
    """Stage all, commit, push. Returns True if a commit was made."""
    git(["add", "-A"], target_root)
    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=target_root
    )
    if status.returncode == 0:
        print("    (nothing new to commit)")
        return False
    git(["commit", "-m", message], target_root)
    git(["push", "--force", "origin", "HEAD:main"], target_root)
    print(f"    ✅ Pushed: {message[:80]}")
    return True


# ─── ANTHROPIC API ────────────────────────────────────────────────────────────

def call_api(system: str, user: str, api_key: str, retries: int = 3) -> str:
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": ANTHROPIC_MODEL,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": MAX_TOKENS,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(
                ANTHROPIC_API_URL, headers=headers, json=payload, timeout=120
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]
        except requests.exceptions.HTTPError:
            body = resp.text[:500]
            print(f"  ❌ HTTP {resp.status_code}: {body}")
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"  ⏳ Rate-limited — waiting {wait}s …")
                time.sleep(wait)
            elif attempt < retries - 1:
                print(f"  ⚠️  Retrying ({attempt+1}/{retries}) …")
                time.sleep(3)
            else:
                raise RuntimeError(f"API failed: {resp.status_code} {body}")
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                print(f"  ⏰ Timeout — retrying ({attempt+1}/{retries}) …")
                time.sleep(5)
            else:
                raise
    return ""


# ─── FILE UTILITIES ───────────────────────────────────────────────────────────

def iter_source_files(root: Path) -> Generator[Path, None, None]:
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        if path.suffix in IGNORE_EXTENSIONS:
            continue
        yield path


def read_file_safe(path: Path, max_bytes: int = 80_000) -> str:
    try:
        size = path.stat().st_size
        if size > max_bytes:
            return f"[FILE TOO LARGE: {size} bytes, skipped]"
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[ERROR READING: {e}]"


def parse_file_blocks(response: str) -> list[dict]:
    changes: list[dict] = []
    for m in re.finditer(r'<FILE\s+path="([^"]+)">(.*?)</FILE>', response, re.DOTALL):
        changes.append(
            {"action": "write", "path": m.group(1).lstrip("/"), "content": m.group(2).strip()}
        )
    for m in re.finditer(r'<DELETE\s+path="([^"]+)"\s*/>', response):
        changes.append({"action": "delete", "path": m.group(1).lstrip("/"), "content": ""})
    return changes


def uses_backend(content: str) -> bool:
    markers = [
        "from 'convex",  'from "convex',
        "useQuery",       "useMutation",      "useAction",
        "ConvexProvider", "ConvexReactClient",
        "api.",           "convex/",
    ]
    return any(m in content for m in markers)


def batch_list(items: list, size: int) -> list[list]:
    return [items[i: i + size] for i in range(0, len(items), size)]


def build_context_summary(source_root: Path) -> str:
    priority = [
        "package.json", "app.json", "tsconfig.json",
        "App.tsx", "app/_layout.tsx", "app/index.tsx",
    ]
    parts = []
    for name in priority:
        p = source_root / name
        if p.exists():
            content = read_file_safe(p)
            if len(content) > 2000:
                content = content[:2000] + "\n…[truncated]"
            parts.append(f"=== {name} ===\n{content}")
    summary = "\n\n".join(parts)
    return summary[:4000]


def apply_changes(target_root: Path, changes: list[dict]) -> int:
    count = 0
    for c in changes:
        dest = target_root / c["path"]
        if c["action"] == "write":
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(c["content"], encoding="utf-8")
            count += 1
        elif c["action"] == "delete" and dest.exists():
            dest.unlink()
            count += 1
    return count


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── STEPS ────────────────────────────────────────────────────────────────────

def step_analyze(source_root: Path) -> dict:
    all_files = list(iter_source_files(source_root))
    convex_dir = [
        f for f in all_files
        if f.relative_to(source_root).parts
        and f.relative_to(source_root).parts[0] == "convex"
    ]
    backend_using = [
        f for f in all_files
        if f.suffix in CODE_EXTENSIONS and uses_backend(read_file_safe(f))
    ]
    print(f"  Total source files:          {len(all_files)}")
    print(f"  Convex dir files:            {len(convex_dir)}")
    print(f"  Files that call the backend: {len(backend_using)}")
    return {"all_files": all_files, "convex_dir_files": convex_dir, "backend_using": backend_using}


def step_copy_source(source_root: Path, target_root: Path) -> int:
    """Copy source tree to target. Skips files that already exist (safe to re-run)."""
    count = 0
    for src in iter_source_files(source_root):
        dest = target_root / src.relative_to(source_root)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            count += 1
    return count


def step_generate_scaffold(
    source_root: Path, target_root: Path,
    api_key: str, ckpt: Checkpoint, budget: TimeBudget,
) -> None:
    if ckpt.is_step_done("scaffold"):
        print("  ✓ Scaffold already done — skipping")
        return
    if budget.expired():
        print("  ⏱️  Time budget gone — skipping scaffold (resume next run)")
        return

    convex_parts: list[str] = []
    convex_dir = source_root / "convex"
    if convex_dir.exists():
        for f in sorted(convex_dir.rglob("*")):
            if f.is_file() and f.suffix in CODE_EXTENSIONS:
                convex_parts.append(
                    f"=== {f.relative_to(source_root)} ===\n{read_file_safe(f)}"
                )

    prompt = f"""## Project context
{build_context_summary(source_root)}

## package.json
{read_file_safe(source_root / "package.json")}

## Existing Convex backend files (encapsulate, do NOT delete)
{chr(10).join(convex_parts) or "(no convex dir found)"}

---

Generate the complete `src/backend/` abstraction scaffold:
1. src/backend/types.ts
2. src/backend/IBackendProvider.ts
3. src/backend/convex/ConvexProvider.ts
4. src/backend/convex/index.ts
5. src/backend/index.ts"""

    print("  Calling API for scaffold …")
    response = call_api(SYSTEM_PROMPT, prompt, api_key)
    n = apply_changes(target_root, parse_file_blocks(response))
    print(f"  Wrote {n} scaffold files")

    commit_and_push(target_root, "feat(abstraction): add src/backend/ scaffold [agent]")
    ckpt.mark_step_done("scaffold")
    time.sleep(REQUEST_DELAY_SECONDS)


def step_migrate_call_sites(
    source_root: Path, target_root: Path,
    backend_using: list[Path], context_summary: str,
    api_key: str, ckpt: Checkpoint, budget: TimeBudget,
) -> bool:
    """Returns True if all batches completed, False if time ran out."""
    batches = batch_list(backend_using, BATCH_SIZE)
    ckpt.set_total_batches(len(batches))

    for idx, file_batch in enumerate(batches):
        if ckpt.is_batch_done(idx):
            print(f"  ✓ Batch {idx+1}/{len(batches)} already done — skipping")
            continue

        if budget.expired():
            pending = sum(1 for i in range(len(batches)) if not ckpt.is_batch_done(i))
            print(
                f"  ⏱️  Time budget exhausted — {pending} batch(es) still pending.\n"
                f"     Re-trigger the workflow to continue."
            )
            return False

        print(f"  Batch {idx+1}/{len(batches)}: {len(file_batch)} files …")

        files_content = [
            f"=== {f.relative_to(source_root)} ===\n{read_file_safe(f)}"
            for f in file_batch
        ]

        prompt = f"""## Project context
{context_summary}

---

## Files to refactor

Replace all direct Convex imports and hook calls with:
  import backend from 'src/backend'   (adjust relative path per file location)

- Keep ALL business logic and UI code unchanged.
- Remove ConvexProvider wrappers — the client initialises inside ConvexProvider.ts.
- Mark any hard-to-abstract Convex type with:
  // TODO: remove if switching backend provider

{chr(10).join(files_content)}"""

        response = call_api(SYSTEM_PROMPT, prompt, api_key)
        n = apply_changes(target_root, parse_file_blocks(response))
        print(f"    → wrote {n} files")

        commit_and_push(
            target_root,
            f"feat(abstraction): refactor call-sites batch {idx+1}/{len(batches)} [agent]",
        )
        ckpt.mark_batch_done(idx)
        time.sleep(REQUEST_DELAY_SECONDS)

    return True


def write_report(target_root: Path, ckpt_data: dict) -> None:
    lines = [
        "# Abstraction Layer Report",
        "",
        f"Started: {ckpt_data.get('started_at', '?')}",
        f"Finished: {_now()}",
        "",
        "## What was done",
        "All direct Convex backend calls are now routed through `IBackendProvider`.",
        "The original Convex SDK is preserved inside `src/backend/convex/ConvexProvider.ts`.",
        "",
        "## Structure",
        "```",
        "src/backend/",
        "  types.ts                ← domain DTOs, no provider imports",
        "  IBackendProvider.ts     ← the interface every provider must implement",
        "  index.ts                ← change ONE import here to swap backends",
        "  convex/",
        "    ConvexProvider.ts     ← original Convex SDK calls, encapsulated",
        "    index.ts",
        "```",
        "",
        "## To add a new backend",
        "1. `src/backend/<name>/MyProvider.ts` — implement `IBackendProvider`",
        "2. Update the import in `src/backend/index.ts`",
        "3. Done — no other files need to change",
        "",
        "## Next steps",
        "- `npx tsc --noEmit` — verify TypeScript compiles",
        "- Search `// TODO: remove if switching backend provider` for tricky spots",
        "- Run your test suite to confirm runtime behaviour is unchanged",
    ]
    (target_root / "ABSTRACTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",               required=True)
    parser.add_argument("--target",               required=True)
    parser.add_argument("--dry-run",              default="false", choices=["true", "false"])
    parser.add_argument("--source-repo-url",      default="")
    parser.add_argument("--job-timeout-seconds",  type=int, default=1800,
                        help="Total job timeout in seconds — agent stops SAFETY_BUFFER before this")
    args = parser.parse_args()

    dry_run     = args.dry_run.lower() == "true"
    source_root = Path(args.source).resolve()
    target_root = Path(args.target).resolve()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    budget = TimeBudget(args.job_timeout_seconds)

    print("\n🤖 Convex → Abstract Backend Layer Agent")
    print("=" * 55)
    print(f"   Source:      {source_root}")
    print(f"   Target:      {target_root}")
    print(f"   Dry run:     {dry_run}")
    print(f"   Time budget: ~{budget.remaining()}s")
    print("=" * 55)

    print("\n🔍 Step 1: Analysing codebase …")
    analysis = step_analyze(source_root)
    if not analysis["convex_dir_files"] and not analysis["backend_using"]:
        print("⚠️  No Convex usage detected. Nothing to abstract.")
        sys.exit(0)

    print("\n📋 Step 2: Loading checkpoint …")
    ckpt            = Checkpoint(target_root, args.source_repo_url)
    context_summary = build_context_summary(source_root)

    if dry_run:
        nb   = len(batch_list(analysis["backend_using"], BATCH_SIZE))
        done = sum(1 for i in range(nb) if ckpt.is_batch_done(i))
        print(f"  [DRY RUN] scaffold: {'done' if ckpt.is_step_done('scaffold') else 'pending'}")
        print(f"  [DRY RUN] call-site batches: {done}/{nb} done")
        sys.exit(0)

    if not ckpt.is_step_done("source_copied"):
        print("\n📂 Step 3: Copying source tree …")
        n = step_copy_source(source_root, target_root)
        print(f"  Copied {n} new files")
        commit_and_push(target_root, "chore: initial source tree copy [agent]")
        ckpt.mark_step_done("source_copied")
    else:
        print("\n📂 Step 3: Source copy already done — skipping")

    print("\n🏗️  Step 4: Generating src/backend/ scaffold …")
    step_generate_scaffold(source_root, target_root, api_key, ckpt, budget)

    print(f"\n🔄 Step 5: Refactoring {len(analysis['backend_using'])} call-site files …")
    all_done = step_migrate_call_sites(
        source_root, target_root,
        analysis["backend_using"], context_summary,
        api_key, ckpt, budget,
    )

    print("\n📄 Step 6: Writing report …")
    write_report(target_root, ckpt._data)

    if all_done and ckpt.is_step_done("scaffold"):
        ckpt.clear()
        commit_and_push(
            target_root,
            "docs: add ABSTRACTION_REPORT; remove checkpoint [agent complete]",
        )
        print("\n🎉 All done!")
    else:
        commit_and_push(target_root, "chore: save progress checkpoint [agent partial]")
        batches = batch_list(analysis["backend_using"], BATCH_SIZE)
        pending = sum(1 for i in range(len(batches)) if not ckpt.is_batch_done(i))
        print(f"\n⏸️  Paused — {pending} batch(es) remaining. Re-trigger workflow to resume.")
        sys.exit(2)   # exit code 2 = partial run, CI can detect and re-queue


if __name__ == "__main__":
    main()