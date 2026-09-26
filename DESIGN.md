# DESIGN — NEMO Dashboard

## Mode
Operate — the visitor completes tasks. Scanability, consistency, native affordances, and the real usage scene outrank expression. Brand lives in precise details.

## Visual world: Deep Ocean Monitoring Station

THEIS: A clinical instrument panel submerged in darkness — every pixel serves data reading, not decoration. The category-default arrangement of stacked cards with icons is refused in favor of a continuous workspace with a persistent status bar.

OWN-WORLD: Near-black base (#060B14), cyan glow (#00F0FF) as the primary data accent, teal (#10B981) for positive states, amber (#F59E0B) for warnings, red (#EF4444) for critical alerts. JetBrains Mono for all numeric/data display, system sans-serif (Inter) for UI labels. Borders are 1px solid rgba(0,240,255,0.15).

STORY: The visitor opens the dashboard and immediately sees the system health bar. Missions queue at the top. Clicking a mission reveals its detections in a table with color-coded confidence. Every data point is readable at a glance. The interface feels like a submarine command panel — precise, alive, trustworthy.

FIRST VIEWPORT: Full-width dark background. Top: slim nav bar with "NEMO" logo, health status indicator, and API connection status. Left sidebar: mission list (scrollable). Center: selected mission's detection table with columns for ID, Frame, Detector Conf, Class, Verifier, Artificialness, Priority, Status. Bottom: persistent footer with export controls and system info. The primary action "Process Mission" sits prominent in the top-right.

FORM: Single-file vanilla HTML/CSS/JS. No framework, no build step. CSS custom properties for all tokens. CSS grid for layout. All interactions wired to real API calls. Loading states, error states, empty states all handled.

## Color strategy
Restrained with one committed accent. Near-black surfaces with cyan (#00F0FF) carrying ~15% of the surface as primary accent lines and active states. Teal (#10B981) for verified/positive. Amber (#F59E0B) for warnings/pending. Red (#EF4444) for rejected/critical.

## Typography
JetBrains Mono for all data (detection IDs, scores, coordinates). Inter for all labels, headings, navigation. Display max 1.5rem. Body measure 65-75ch.

## Bans
- No gradient text
- No glassmorphism
- No icon+heading+text card grid as page structure
- No hero-metric template
- No section numbers
- No tracked uppercase eyebrows
- No modals for tasks
- No skeleton loaders — use actual loading states

## Tokens
```css
--bg-primary: #060B14;
--bg-secondary: #0D1520;
--bg-card: #111B2E;
--bg-card-hover: #162035;
--border: rgba(0, 240, 255, 0.12);
--border-active: rgba(0, 240, 255, 0.4);
--text-primary: #E2E8F0;
--text-secondary: #94A3B8;
--text-muted: #64748B;
--accent: #00F0FF;
--accent-dim: rgba(0, 240, 255, 0.15);
--positive: #10B981;
--warning: #F59E0B;
--danger: #EF4444;
--radius: 8px;
--radius-lg: 12px;
```

## Component library
- **Nav bar**: Slim, full-width, with logo, status indicators, API connection pulse
- **Sidebar**: Mission list, scrollable, active mission highlighted
- **Detection table**: Full-width, columns sortable, rows with status badges
- **Button**: Primary (cyan fill), secondary (outline), disabled (muted)
- **Badge**: Status pills — candidate (amber), verified (teal), rejected (red)
- **Code block**: Pre-formatted JSON responses in monospace
- **Input**: Dark background, cyan border on focus
- **Alert**: Inline error/success messages
- **Spinner**: CSS-only, cyan, for loading states
- **Status dot**: Pulse animation for live indicators
