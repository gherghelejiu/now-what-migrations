#!/usr/bin/env python3
"""
Convex → Supabase Migration Agent
Runs in GitHub Actions to port a React Native + Convex app to Supabase.
Uses the Anthropic API (Claude 3.5 Sonnet) for intelligent code transformation.
"""

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Generator

import requests

# ─── CONFIG ───────────────────────────────────────────────────────────────────

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-4-6-sonnet"
MAX_TOKENS = 8000
REQUEST_DELAY_SECONDS = 1.5

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

# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert React Native and backend developer specializing in migrating
applications from Convex to Supabase.

## Core Transformation Rules

### Backend (convex/ → supabase/)
- Replace `convex/schema.ts` with `supabase/migrations/001_initial.sql` (CREATE TABLE statements with RLS)
- Replace Convex server functions (queries, mutations, actions) with Supabase Edge Functions where needed
- Create `supabase/config.ts` — Supabase client singleton using env vars
- Create `supabase/types.ts` — TypeScript types matching the schema

### Client-side hooks
| Convex | Supabase replacement |
|--------|---------------------|
| `useQuery(api.x.list)` | `useState` + `useEffect` with `supabase.from('x').select()`, or real-time subscription via `supabase.channel()` |
| `useMutation(api.x.create)` | async function calling `supabase.from('x').insert()` |
| `useMutation(api.x.update)` | async function calling `supabase.from('x').update()` |
| `useMutation(api.x.delete)` | async function calling `supabase.from('x').delete()` |
| `useAction(api.x.action)` | async function calling a Supabase Edge Function |
| `ConvexProvider` | wrap app with nothing special; Supabase client is a singleton |
| `ConvexAuthNextjsServerProvider` | `supabase.auth` |

### Auth
- Replace Convex auth with `supabase.auth.signUp()`, `supabase.auth.signInWithPassword()`, etc.
- For session: use `supabase.auth.getSession()` and `supabase.auth.onAuthStateChange()`

### Package changes
- Remove: `convex`, `@convex-dev/*`
- Add: `@supabase/supabase-js`
- Env vars: `EXPO_PUBLIC_SUPABASE_URL`, `EXPO_PUBLIC_SUPABASE_ANON_KEY`

## Output Format

For files to create or modify, use EXACTLY this format (no markdown code fences around it):
<FILE path="relative/path/to/file">
file content here
</FILE>

For files to delete:
<DELETE path="relative/path/to/file" />

Only output files that need to change. Skip files with no Convex references."""


# ─── ANTHROPIC API ────────────────────────────────────────────────────────────

def call_api(system: str, user: str, api_key: str, retries: int = 3) -> str:
    """Call the Anthropic API with retry logic."""
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
            resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]
        except requests.exceptions.HTTPError as e:
            body = resp.text[:500]
            print(f"  ❌ HTTP {resp.status_code}: {body}")
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"  ⏳ Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            elif attempt < retries - 1:
                print(f"  ⚠️  Retrying ({attempt+1}/{retries})...")
                time.sleep(3)
            else:
                raise RuntimeError(f"Anthropic API failed: {resp.status_code} {body}") from e
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                print(f"  ⏰ Timeout, retrying ({attempt+1}/{retries})...")
                time.sleep(5)
            else:
                raise

    return ""


# ─── FILE UTILITIES ───────────────────────────────────────────────────────────

def iter_source_files(root: Path) -> Generator[Path, None, None]:
    """Yield all relevant source files, skipping ignored dirs/extensions."""
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        if path.suffix in IGNORE_EXTENSIONS:
            continue
        yield path


def read_file_safe(path: Path, max_bytes: int = 80_000) -> str:
    """Read a file safely, truncating if too large."""
    try:
        size = path.stat().st_size
        if size > max_bytes:
            return f"[FILE TOO LARGE: {size} bytes, skipped]"
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[ERROR READING: {e}]"


def parse_file_blocks(response: str) -> list[dict]:
    """Parse <FILE> and <DELETE> blocks from the AI response."""
    changes = []
    file_pattern = re.compile(r'<FILE\s+path="([^"]+)">(.*?)</FILE>', re.DOTALL)
    for match in file_pattern.finditer(response):
        changes.append({
            "action": "write",
            "path": match.group(1).lstrip("/"),
            "content": match.group(2).strip(),
        })
    delete_pattern = re.compile(r'<DELETE\s+path="([^"]+)"\s*/>')
    for match in delete_pattern.finditer(response):
        changes.append({"action": "delete", "path": match.group(1).lstrip("/"), "content": ""})
    return changes


def uses_convex(content: str) -> bool:
    """Quick check if a file likely references Convex."""
    markers = [
        "from 'convex", 'from "convex',
        "useQuery", "useMutation", "useAction",
        "ConvexProvider", "ConvexReactClient",
        "api.", "convex/",
    ]
    return any(m in content for m in markers)


def batch(items: list, size: int) -> Generator[list, None, None]:
    for i in range(0, len(items), size):
        yield items[i: i + size]


# ─── MIGRATION STEPS ──────────────────────────────────────────────────────────

def step_analyze(source_root: Path) -> dict:
    all_files = list(iter_source_files(source_root))
    convex_files = [
        f for f in all_files
        if f.relative_to(source_root).parts and f.relative_to(source_root).parts[0] == "convex"
    ]
    convex_using = [
        f for f in all_files
        if f.suffix in CODE_EXTENSIONS and uses_convex(read_file_safe(f))
    ]
    print(f"  Total source files:        {len(all_files)}")
    print(f"  convex/ backend files:     {len(convex_files)}")
    print(f"  Files that import Convex:  {len(convex_using)}")
    return {"all_files": all_files, "convex_files": convex_files, "convex_using": convex_using}


def step_generate_schema(source_root: Path, api_key: str) -> list[dict]:
    print("  Reading Convex schema + backend files...")
    convex_dir = source_root / "convex"
    if not convex_dir.exists():
        print("  ⚠️  No convex/ directory found — will infer schema from app code")
        convex_content = "(No convex/ directory found)"
    else:
        parts = []
        for f in sorted(convex_dir.rglob("*.ts")):
            rel = f.relative_to(source_root)
            parts.append(f"=== {rel} ===\n{read_file_safe(f)}")
        convex_content = "\n\n".join(parts) if parts else "(convex/ is empty)"

    pkg_json = read_file_safe(source_root / "package.json")
    app_json_path = source_root / "app.json"
    app_json = read_file_safe(app_json_path) if app_json_path.exists() else ""

    prompt = f"""Analyze this Convex backend and generate the complete Supabase foundation files.

## package.json
{pkg_json}

## app.json
{app_json}

## Convex backend files
{convex_content}

Please generate ALL of the following files:
1. `supabase/migrations/001_initial.sql` — Full schema with CREATE TABLE, RLS policies, and indexes
2. `supabase/types.ts` — TypeScript types matching every table
3. `supabase/config.ts` — Supabase client singleton (use EXPO_PUBLIC_ prefixed env vars for Expo)
4. `.env.example` — Template with EXPO_PUBLIC_SUPABASE_URL and EXPO_PUBLIC_SUPABASE_ANON_KEY
5. Updated `package.json` — Remove convex packages, add @supabase/supabase-js"""

    print("  Calling Anthropic API for schema generation...")
    response = call_api(SYSTEM_PROMPT, prompt, api_key)
    changes = parse_file_blocks(response)
    print(f"  Generated {len(changes)} schema/config files")
    return changes


def step_migrate_files(
    source_root: Path,
    convex_using: list[Path],
    context_summary: str,
    api_key: str,
) -> list[dict]:
    all_changes = []
    file_batches = list(batch(convex_using, 5))

    for i, file_batch in enumerate(file_batches):
        print(f"  Batch {i+1}/{len(file_batches)}: migrating {len(file_batch)} files...")
        files_content = []
        for f in file_batch:
            rel = f.relative_to(source_root)
            content = read_file_safe(f)
            files_content.append(f"=== {rel} ===\n{content}")

        prompt = f"""## Project context (for reference)
{context_summary}

---

## Files to migrate

Migrate each of the following files from Convex to Supabase.
Import the Supabase client from `supabase/config` (adjust relative path as needed).
Import types from `supabase/types` (adjust relative path as needed).

{chr(10).join(files_content)}"""

        response = call_api(SYSTEM_PROMPT, prompt, api_key)
        changes = parse_file_blocks(response)
        all_changes.extend(changes)
        print(f"    → {len(changes)} file changes generated")
        time.sleep(REQUEST_DELAY_SECONDS)

    return all_changes


def step_apply_changes(
    source_root: Path,
    target_root: Path,
    all_changes: list[dict],
    dry_run: bool,
) -> dict:
    stats = {"copied": 0, "written": 0, "deleted": 0, "skipped": 0}

    if dry_run:
        print("  [DRY RUN] Skipping file writes")
        stats["written"] = len([c for c in all_changes if c["action"] == "write"])
        stats["deleted"] = len([c for c in all_changes if c["action"] == "delete"])
        return stats

    print("  Copying source → target (excluding convex/)...")
    for src_file in iter_source_files(source_root):
        rel = src_file.relative_to(source_root)
        if rel.parts and rel.parts[0] == "convex":
            continue
        dest = target_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest)
        stats["copied"] += 1

    print(f"  Applying {len(all_changes)} AI-generated changes...")
    for change in all_changes:
        dest = target_root / change["path"]
        if change["action"] == "write":
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(change["content"], encoding="utf-8")
            stats["written"] += 1
        elif change["action"] == "delete":
            if dest.exists():
                dest.unlink()
                stats["deleted"] += 1
            else:
                stats["skipped"] += 1

    return stats


def build_context_summary(source_root: Path) -> str:
    parts = []
    priority_files = [
        "package.json", "app.json", "tsconfig.json",
        "App.tsx", "app/_layout.tsx", "app/index.tsx",
    ]
    for name in priority_files:
        p = source_root / name
        if p.exists():
            content = read_file_safe(p)
            if len(content) > 2000:
                content = content[:2000] + "\n...[truncated]"
            parts.append(f"=== {name} ===\n{content}")
    summary = "\n\n".join(parts)
    return summary[:4000] if len(summary) > 4000 else summary


def write_report(all_changes: list[dict], stats: dict, dry_run: bool) -> None:
    written = [c for c in all_changes if c["action"] == "write"]
    deleted = [c for c in all_changes if c["action"] == "delete"]
    lines = [
        "# Migration Report: Convex → Supabase",
        "",
        f"**Dry run:** `{dry_run}`",
        "",
        "## Summary",
        "",
        "| Action | Count |",
        "|--------|-------|",
        f"| Files copied (unchanged) | {stats.get('copied', 0)} |",
        f"| Files written/created by AI | {stats.get('written', 0)} |",
        f"| Files deleted | {stats.get('deleted', 0)} |",
        "",
        "## Files Created / Modified",
        "",
    ]
    for c in written:
        lines.append(f"- `{c['path']}`")
    if deleted:
        lines += ["", "## Files Deleted", ""]
        for c in deleted:
            lines.append(f"- `{c['path']}`")
    lines += [
        "",
        "## Next Steps",
        "",
        "1. Create a Supabase project at https://supabase.com",
        "2. Run `supabase/migrations/001_initial.sql` in the SQL Editor",
        "3. Copy `.env.example` → `.env` and fill in your Supabase credentials",
        "4. Run `npm install` then `npx expo start`",
        "5. Review RLS policies in the migration SQL — tighten as needed",
    ]
    report_path = Path("migration_report.md")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄 Report written to {report_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convex → Supabase migration agent")
    parser.add_argument("--source", required=True, help="Path to cloned source repo")
    parser.add_argument("--target", required=True, help="Path to cloned target repo")
    parser.add_argument("--dry-run", default="false", choices=["true", "false"])
    args = parser.parse_args()

    dry_run = args.dry_run.lower() == "true"
    source_root = Path(args.source).resolve()
    target_root = Path(args.target).resolve()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY environment variable is not set", file=sys.stderr)
        sys.exit(1)

    print("\n🤖 Convex → Supabase Migration Agent")
    print("=" * 50)
    print(f"   Source:  {source_root}")
    print(f"   Target:  {target_root}")
    print(f"   Dry run: {dry_run}")
    print(f"   Model:   {ANTHROPIC_MODEL}")
    print("=" * 50)

    print("\n🔍 Step 1: Analyzing codebase...")
    analysis = step_analyze(source_root)

    if not analysis["convex_files"] and not analysis["convex_using"]:
        print("⚠️  No Convex usage detected. Nothing to migrate.")
        sys.exit(0)

    print("\n📋 Step 2: Building project context...")
    context_summary = build_context_summary(source_root)

    print("\n📊 Step 3: Generating Supabase schema, types, and config...")
    schema_changes = step_generate_schema(source_root, api_key)
    time.sleep(REQUEST_DELAY_SECONDS)

    print(f"\n🔄 Step 4: Migrating {len(analysis['convex_using'])} files that use Convex...")
    app_changes = step_migrate_files(
        source_root, analysis["convex_using"], context_summary, api_key,
    )

    all_changes = schema_changes + app_changes
    print(f"\n✅ Total AI-generated changes: {len(all_changes)}")

    print("\n✍️  Step 5: Applying changes to target repo...")
    stats = step_apply_changes(source_root, target_root, all_changes, dry_run)
    print(f"   Copied: {stats['copied']}  Written: {stats['written']}  Deleted: {stats['deleted']}")

    print("\n📄 Step 6: Writing migration report...")
    write_report(all_changes, stats, dry_run)

    print("\n🎉 Agent finished successfully!")
    if dry_run:
        print("   (Dry run — no files were written to the target repo)")


if __name__ == "__main__":
    main()