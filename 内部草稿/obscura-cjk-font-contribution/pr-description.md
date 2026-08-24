## render: opt-in system font loading via `OBSCURA_LOAD_SYSTEM_FONTS` for CJK text (#606)

Fixes #606.

### Problem

The render pipeline builds its text database solely from the 14 embedded Latin
faces and never calls `fontdb`'s `load_system_fonts()` — deliberately, per the
comment in `TextEngine::new_with_web_fonts_and_emoji`, to keep layout
deterministic across machines and avoid a multi-millisecond startup scan.

The side effect: every page whose CSS resolves to host system fonts (i.e.
virtually all CJK pages — those sites almost never ship web fonts) renders
text as tofu boxes in screenshots/PDF. #606's repro
(`obscura fetch https://www.baidu.com --screenshot`) is a canonical case.
Verified on Windows x64, v0.2.0/v0.2.1 stealth and current main.

### Approach

Rather than reversing the determinism guarantee, this adds an opt-in escape
hatch:

- New env var **`OBSCURA_LOAD_SYSTEM_FONTS`** (`1`/`true`/`on`/`yes`).
- **Unset (default): zero behavior change** — no system scan, identical
  embedded-only database, identical layout output. The determinism guarantee
  documented in the code comment is preserved for all existing users/CI.
- **Set:** after the embedded faces, load a small whitelist of common
  CJK-capable faces (Microsoft YaHei / UI, SimSun, SimHei, Source Han Sans SC)
  from the standard Windows font directories through
  `db.load_font_source(Source::Binary(...))`.

Design notes:

- **Whitelist over full scan**: a full `load_system_fonts()` pass crashed with
  a main-thread stack overflow here (font loading runs deep in the call stack;
  Windows default stack is 1 MB), and it reintroduces exactly the startup cost
  the comment warns about. Six known-good files cover zh-CN/Hans rendering.
  Cross-platform dirs (`%LOCALAPPDATA%` per-user fonts + `C:\Windows\Fonts`)
  are probed; missing faces are skipped silently, so the flag is inert on
  Linux/macOS or stripped hosts.
- **No new dependencies**, single function touched (+14/−1).
- Startup diagnostics go to stderr only when the flag is active.

### Verification

- Default run (no env var): byte-identical screenshots before/after patch on a
  local render test page suite.
- With `OBSCURA_LOAD_SYSTEM_FONTS=1`: `obscura fetch https://www.baidu.com
  --screenshot baidu.png` renders all Chinese text correctly in Microsoft
  YaHei; previously all CJK glyphs were hollow boxes (#606 repro).
- Also exercised via MCP `browser_navigate` + `browser_screenshot` on
  Baidu and on a local HTML fixture using `font-family: "Microsoft YaHei"`.

### Notes for reviewers

If upstream would rather have the full system-font scan behind the flag than a
whitelist, that works too — but note the stack overflow above; the call site
would need a bigger-stack thread (e.g.
`std::thread::Builder::new().stack_size(16 << 20)` around font DB init, or
`/STACK:` linker bump). I kept the whitelist because it solves the actual
reported problem with less risk.

Happy to adjust naming (`OBSCURA_LOAD_SYSTEM_FONTS` vs something like
`OBSCURA_EXTRA_FONT_DIRS`, which could take explicit directories instead of a
boolean — arguably more flexible and more deterministic) if you prefer.
