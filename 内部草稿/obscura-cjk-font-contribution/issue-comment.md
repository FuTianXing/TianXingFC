## Root cause found — it's intentional, plus a working patch

I hit the same tofu-box issue on Windows (stealth build) and dug into the
source. This is **not** a fontdb bug — it's a deliberate design decision in
`crates/obscura-render/src/inline.rs`, `TextEngine::new_with_web_fonts_and_emoji`:

```rust
// Build a database from embedded and page-provided faces. Never call
// load_system_fonts: a host's font set would make layout differ
// machine to machine and add a multi-millisecond startup scan.
```

The render pipeline builds its `fontdb::Database` exclusively from 14 embedded
Latin faces (SANS/SERIF/MONO/SYSTEM × weights, plus optional emoji). It never
calls `db.load_system_fonts()`. So any text whose font-family resolves to a
host system font — which is essentially every CJK/Japanese/Korean page, since
those sites rarely ship web fonts — falls back to a face with no CJK coverage
and renders as hollow rectangles.

What I verified along the way:

- Copying fonts into `C:\Windows\Fonts` / per-user font dirs does nothing,
  because those directories are never scanned by design.
- `@font-face` with a `file://` URL is not loaded either.
- The only no-source-change workaround that works: subset a system CJK font
  with fontTools and inline it as a base64 data-URL `@font-face`. Works, but
  it's per-page boilerplate and doesn't help screenshots of arbitrary sites
  (e.g. `obscura fetch https://www.baidu.com --screenshot page.png`, which is
  exactly this issue's repro).

## Proposed fix: opt-in env flag, default behavior unchanged

I fully understand the motivation behind "never load system fonts" —
deterministic cross-machine layout and fast startup. So instead of flipping
that around, the patch adds an opt-in escape hatch:

- New env var `OBSCURA_LOAD_SYSTEM_FONTS=1` (`true`/`on`/`yes` also accepted).
- When unset (the default), behavior is byte-for-byte identical to today: no
  system scan, same embedded-only database, same layout determinism.
- When set, after the embedded faces are registered, the patch loads a small
  whitelist of common CJK-capable faces (Microsoft YaHei, SimSun, SimHei,
  Source Han Sans) from `C:\Windows\Fonts` and `%LOCALAPPDATA%\Microsoft\Windows\Fonts`
  via `db.load_font_source(Source::Binary(...))`.

Two deliberate choices worth highlighting:

1. **Whitelist instead of full `load_system_fonts()` scan.** On my machine a
   full scan crashed with a stack overflow in the main thread (font loading
   happens fairly deep in the call stack; Windows default stack is 1 MB).
   Loading six known-good files avoids both that fragility and the
   multi-ms startup cost. If upstream prefers the full scan instead, wrapping
   the call site so it runs on a thread with a larger stack
   (`std::thread::Builder::new().stack_size(...)`) also works — I verified
   `/STACK:16777216` at link time fixes the overflow on MSVC.
2. **Missing/unreadable files are skipped silently**, so the flag is safe on
   Linux/macOS or stripped-down Windows where these faces don't exist.

With `OBSCURA_LOAD_SYSTEM_FONTS=1`, `obscura fetch https://www.baidu.com
--screenshot page.png` renders all Chinese text correctly (Microsoft YaHei).
Without the variable, output is unchanged from stock v0.2.x.

Patch against current `main` attached as
[`obscura-system-fonts.patch`](attachment://obscura-system-fonts.patch)
(+14/−1 net, single function, no new dependencies). Happy to open a PR if
you're open to the approach — and I'd also be interested in your long-term
take: whether you'd rather keep determinism sacred and document the env-var /
web-font workaround, or eventually embed a small CJK fallback face like the
emoji one.

---

Repro environment: Windows 11 x64, obscura v0.2.1 stealth (official archive),
also reproduced on self-built main @ `150b434`.
