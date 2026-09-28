# DESIGN — NEMO Deep-Sea Ops Console

Replacement visual world for the NEMO dashboard (single-file `dashboard/index.html`). The old neon-on-black look is evidence only; this system replaces it.

## Scene

A sonar room aboard a survey vessel at night: lights low, phosphor displays glowing, operators reading contacts off scopes. Dark is forced by the scene, never a default. Light text is warm-tinted phosphor white, never pure gray on black.

## Color strategy — Restrained with a phosphor carrier

Phosphor seafoam carries ~35% of the surface (scopes, active states, primary actions); abyssal navy grounds everything else; amber and red exist only as signal states.

- `--abyss-0: #04090F` — page ground (deepest)
- `--abyss-1: #081220` — rail and header (one fathom up)
- `--abyss-2: #0C1A2C` — panels (instruments float above ground)
- `--abyss-3: #12263C` — hover / raised instrument wells
- `--ink-line: rgba(120, 200, 190, 0.14)` — hairline borders (green-tinted, never gray)
- `--phosphor: #3CF5C8` — primary signal (seafoam phosphor, not cyan-neon)
- `--phosphor-deep: #0E9F7E` — pressed / secondary phosphor
- `--foam: #EAF6F1` — primary text (warm phosphor white)
- `--murk: #9DB8B2` — secondary text (tinted from phosphor hue)
- `--silt: #647E7A` — muted text (tinted, never gray)
- `--amber: #FFB454` — needs-review / warning signal
- `--red: #FF6B6B` — rejected / danger signal
- `--verify-green: #3CF5C8` shared with phosphor; verified state adds a text label, never color alone

Contrast: foam and murk on abyss grounds ≥ 4.5:1; silt used only for large/meta text ≥ 3:1 or paired with labels.

## Typography

- Display / station headers: "Barlow Condensed" (600, uppercase, letter-spacing 0.08em) — the stenciled station-plate voice of the sonar room. Never body copy.
- Body / UI: system stack (`-apple-system, "Segoe UI", Inter, system-ui, sans-serif`) — quiet, legible, native.
- Data / measurement: "JetBrains Mono" — reserved strictly for coordinates, confidences, IDs, timestamps, code. Never prose, never buttons, never headings.
- Scale: station title clamp(2rem, 4vw, 3rem); section heads 15px condensed; body 14px/1.65, measure ≤ 70ch; display tracking never below −0.02em (condensed faces carry their own density).

## Topology & components

- Command-deck frame: fixed depth-ladder station rail (left) with depth-tick markers and numbered stations 01–05 (numbers earn their place: the rail is the mission workflow in order). Top status bar carries connection state, mission clock, and version plate.
- Instruments, not cards: panels are deep wells (`abyss-2`) with 1px ink-line borders, 14px radius, and inset top highlight (a 1px inner glow suggesting a lit bezel). No nested wells; no icon-plus-heading-plus-text grids.
- Contact log, not stat cards: mission figures render as a sounding-strip — a horizontal depth-sounder readout with tick rules and tabular numerals, one authored instrument instead of four same-size cards.
- Pipeline as signal path: stages joined by a live trace line that fills with phosphor as stages complete; the active stage pulses once (opacity only, no glow halo).
- Scope header on overview: a sonar-scope viewport with a slow rotating sweep (conic gradient, 12s revolution, disabled under `prefers-reduced-motion`) behind the mission readout — the signature interaction and the page's memory test.
- Bathymetry: faint SVG contour lines texture the page ground at 4% opacity; panels stay flat above it.
- Buttons: solid phosphor slabs with 2px downward offset + soft blur shadow (depth, never halo); secondary actions are ink-line outlines; destructive is red outline.
- Tables: report table with sticky header, row hover as depth highlight, status as labeled pills with 6px signal dots (label + dot, never color alone).
- Motion: one authored moment — the scope sweep. Everything else transitions once (180ms exponential ease-out) from visible defaults. No scroll-entrance choreography on an ops tool.

## Bans (world-specific)

- No pure-cyan `#00F0FF` neon, no glow halos, no gradient text.
- No gray secondary text; every muted tone is phosphor-tinted.
- No generic card grids of icon + heading + text; no hero-metric four-card strip; no eyebrow over every section (one station kicker per screen only).
- No `border-left` status bars above 1px; state lives in pills, dots-with-labels, and the signal-path trace.
- No modal; detection detail stays in the drawer beside the scope.

## States & accessibility

Hover, focus-visible (2px phosphor outline, 2px offset), disabled (silt, no pointer), loading (trace shimmer + labeled status), error (named problem + recovery action, e.g. backend unreachable → retry + check `uvicorn` command), empty (named station + next action). Touch targets ≥ 44px on coarse pointers. Sweep, shimmer, and transitions fully disabled under `prefers-reduced-motion`.
