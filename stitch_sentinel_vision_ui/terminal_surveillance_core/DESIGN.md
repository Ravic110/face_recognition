---
name: Terminal Surveillance Core
colors:
  surface: '#051424'
  surface-dim: '#051424'
  surface-bright: '#2c3a4c'
  surface-container-lowest: '#010f1f'
  surface-container-low: '#0d1c2d'
  surface-container: '#122131'
  surface-container-high: '#1c2b3c'
  surface-container-highest: '#273647'
  on-surface: '#d4e4fa'
  on-surface-variant: '#c6c6cd'
  inverse-surface: '#d4e4fa'
  inverse-on-surface: '#233143'
  outline: '#909097'
  outline-variant: '#45464c'
  surface-tint: '#c1c6db'
  primary: '#c1c6db'
  on-primary: '#2a3040'
  primary-container: '#0b1120'
  on-primary-container: '#777c90'
  inverse-primary: '#585e70'
  secondary: '#4ae176'
  on-secondary: '#003915'
  secondary-container: '#00b954'
  on-secondary-container: '#004119'
  tertiary: '#ffb3ad'
  on-tertiary: '#68000a'
  tertiary-container: '#2e0002'
  on-tertiary-container: '#e63d3e'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#dde2f8'
  primary-fixed-dim: '#c1c6db'
  on-primary-fixed: '#151b2b'
  on-primary-fixed-variant: '#414658'
  secondary-fixed: '#6bff8f'
  secondary-fixed-dim: '#4ae176'
  on-secondary-fixed: '#002109'
  on-secondary-fixed-variant: '#005321'
  tertiary-fixed: '#ffdad7'
  tertiary-fixed-dim: '#ffb3ad'
  on-tertiary-fixed: '#410004'
  on-tertiary-fixed-variant: '#930013'
  background: '#051424'
  on-background: '#d4e4fa'
  surface-variant: '#273647'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  data-mono:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  label-caps:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.08em
spacing:
  grid-unit: 4px
  container-padding: 16px
  gutter: 1px
  panel-gap: 8px
---

## Brand & Style

The design system is engineered for high-stakes monitoring and real-time data analysis. It targets operators in security, network operations, and aerospace who require immediate cognitive processing of complex data sets. 

The aesthetic is **Technical Minimalism** mixed with **Modern Brutalism**. It prioritizes information density and ocular efficiency. The UI evokes a "Control Center" atmosphere—utilitarian, authoritative, and precise. By utilizing a desktop-first approach at 1280x740px, the system treats the screen as a single-pane-of-glass dashboard where every pixel serves a functional purpose. There are no decorative elements; form follows function with mathematical rigor.

## Colors

The palette is anchored in a deep "Midnight Obsidian" (`#0B1120`) to minimize eye strain during long surveillance shifts. 

*   **Primary Background:** `#0B1120` (Deep Dark)
*   **Success / Known State:** `#22C55E` (Signal Green) — Used for active connections, authorized personnel, and stable systems.
*   **Alert / Critical State:** `#EF4444` (Emergency Red) — Reserved for breaches, system failures, and high-priority targets.
*   **UI Neutral:** `#94A3B8` (Slate) — Used for secondary data, timestamps, and inactive iconography.
*   **Borders/Grid:** `#1E293B` — Low-contrast dividers to maintain structure without visual noise.

## Typography

The typography system leverages **Inter** (as the modern web-standard evolution of Helvetica's neutral spirit) for UI controls and **JetBrains Mono** for technical data readouts.

*   **Hierarchy:** High contrast in weight, rather than size, is used to differentiate information.
*   **Technical Data:** All dynamic values, coordinates, and timestamps must use the `data-mono` style to ensure character alignment and readability.
*   **Labels:** Use `label-caps` for table headers and section titles to create a clear architectural "map" of the dashboard.

## Layout & Spacing

This design system utilizes a **Fixed Module Grid** optimized for a 1280x740px viewport. The layout is divided into a 12-column grid, but functionality is driven by "Panels."

*   **Panel System:** Content is housed in distinct rectangular zones separated by 8px gaps.
*   **Gutter logic:** Internal panel borders are 1px solid `#1E293B`, creating a technical blueprint feel.
*   **Density:** Padding is tight (12px - 16px) to maximize data visualization real estate.
*   **Desktop-First:** On 1280px, the sidebar is fixed at 240px. The main viewport utilizes the remaining 1040px for multi-column data feeds.

## Elevation & Depth

In a surveillance context, shadows are discarded in favor of **Tonal Layering**. Depth is communicated through color luminosity rather than light source simulation.

*   **Level 0 (Background):** `#0B1120` (Global canvas).
*   **Level 1 (Panels):** `#0D1526` (Slightly lighter than background to define work areas).
*   **Level 2 (Active States/Modals):** `#1E293B` (Used for hovered items or dropdown menus).
*   **Outlines:** Instead of shadows, use 1px stroke for all interactive elements to maintain a "wireframe" technical aesthetic.

## Shapes

The design system employs a **Strict 0px Radius** (Sharp) policy. 

Rounding suggests approachability and consumer-friendliness, which contradicts the professional, high-fidelity nature of surveillance software. Sharp corners reinforce the grid and emphasize the mathematical precision of the system. This applies to buttons, panels, input fields, and status indicators.

## Components

*   **Buttons:** Rectangular with 1px borders. Primary action buttons use a subtle `#22C55E` border and text; Critical buttons use `#EF4444`. No gradients.
*   **Status Indicators:** Small 8x8px squares. Solid `#22C55E` for "Clear," pulsing `#EF4444` for "Alert."
*   **Data Tables:** Zebra-striping is forbidden. Use 1px horizontal lines only. All numeric data is right-aligned and monospaced.
*   **Input Fields:** Ghost-style inputs (background-less) with a bottom-border only, or fully outlined 1px rectangles. Focus state is indicated by a color shift of the border to the primary green.
*   **Terminal Feed:** A dedicated component for raw logs using `data-mono` typography, featuring a subtle scanline overlay effect (2px horizontal stripes at 5% opacity).
*   **Radar/Map Widgets:** High-contrast monochromatic maps with `#22C55E` vector overlays.