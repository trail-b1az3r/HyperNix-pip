# HyperNix logo

## Concept

Three equal parallelogram bars, staggered up and to the right, going dark
→ grey → red. It reads as layers in a stack — raw input at the bottom,
climbing through a pipeline to a fast, optimized result at the top. That's
a literal description of what HyperNix does (download → train/quantize →
ship), not a decorative shape bolted onto the name.

Flat, no bevels, no gradients. Three identical shapes at three positions —
nothing to lose at favicon size.

## Files

| File | Use |
|---|---|
| `hypernix-icon.svg` | Primary icon — charcoal/grey/red. Default choice; reads fine on white or light grey. |
| `hypernix-icon-ondark.svg` | Same red top bar, lighter charcoal/grey swapped for off-white/light-grey. Use wherever the backdrop is dark (this site's nav/hero/footer). |
| `hypernix-icon-mono.svg` | `currentColor` fill (middle bar at 72% opacity so the layers stay legible even in one flat tone). For single-ink contexts — stamped merch, inline with text. |
| `hypernix-icon-white.svg` | Solid white, all three bars. Dark photos, dark marketing slides. |
| `hypernix-icon-dark.svg` | Solid charcoal, all three bars. Light backgrounds needing one flat tone (letterhead, engraving). |
| `hypernix-lockup-light.svg` | Icon + wordmark, dark text. README, PyPI, social cards, anywhere light. |
| `hypernix-lockup-dark.svg` | Icon + wordmark, light text. This site's own header/footer. |
| `hypernix-favicon.svg` | Same as the primary icon, but the bottom two bars flip from charcoal/grey to off-white/light-grey under `prefers-color-scheme: dark` so the mark doesn't wash out in a dark browser tab strip. The red top bar never changes. Reference this first; browsers without SVG-favicon support fall back to the PNG/ICO below. |
| `favicon.ico`, `favicon-16.png`, `favicon-32.png`, `favicon-48.png`, `apple-touch-icon.png` (180), `icon-192.png`, `icon-512.png` | Static raster fallbacks, built from `hypernix-icon.svg`. |

## Color

```
charcoal   #141414   bottom bar, light-bg icon
mid grey   #6b6b6b   middle bar, light-bg icon
off-white  #f2f2f0   bottom bar, dark-bg icon
light grey #8f8f8f   middle bar, dark-bg icon
accent red #c8192e   top bar — always this red, every version
```

The red top bar never changes between variants. It's the one constant
that ties every application back to the same mark.

## Wordmark

"HyperNix", Inter (already self-hosted by the docs site) at weight 800,
letter-spacing −0.5px. Set as real `<text>`, not outlined — keeps the
files small and the copy editable. Falls back to system sans-serif
anywhere Inter isn't installed.

## Clear space & minimum size

Keep clear space of at least half the icon's width on every side. Don't
put anything — logo, wordmark, buttons — inside that margin.

Icon alone: don't render smaller than 16px (favicon floor). Lockup:
don't render narrower than 120px, or the wordmark starts fighting for
pixels with the icon.

## What not to do

Don't reorder the bars or change which one is red — the bottom-to-top
dark-to-red progression is the whole idea. Don't add a drop shadow,
bevel, or gradient fill. Don't stretch the icon non-uniformly. Don't set
the wordmark in anything other than a geometric grotesk at heavy weight.
