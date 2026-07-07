# ARTIFACT BRAND IDENTITY & UI/UX DESIGN HANDOFF
## Ground-Truth Context for Codex / Copilot Refactoring

> This file defines the brand identity, visual language, tone, and UI/UX direction for the **Artifact** platform.
>
> It should be used together with:
>
> - `CAPSTONE_IDENTITY_PROJECT_PLAN_CONTEXT.md`
> - the current implementation/SRS handoff
>
> When refactoring the platform, prioritize this document for **brand, styling, component appearance, copy tone, visual hierarchy, and interaction personality**.
>
> Do not add new product features merely because they are common in modern apps. This document governs presentation and brand expression, not scope expansion.

---

# 1. Brand Architecture

## Company

**ChickenScratch Co.**

Abbreviation:

**CSC**

## Product

**Artifact**

## Relationship

ChickenScratch Co. is the broader creator-technology company.

Artifact is the company’s digital artwork provenance verification product.

Preferred public presentation:

```text
Artifact
by ChickenScratch Co.
```

Do not present “ChickenScratch Co.” as the app name.

Do not present “Artifact” as the legal/company identity when company-level branding is needed.

---

# 2. Final Product Identity

## Recommended Capstone / Product Title

> **Artifact: A Registry-Centered Digital Artwork Provenance Verification Platform with Modular Invisible Watermarking**

## One-Sentence Product Definition

> Artifact is a registry-centered digital artwork provenance verification platform that allows creators to register artworks, generate watermarked distribution copies, verify suspected images against selected artwork records, preserve verification history, and export technical verification reports.

## Core Product Idea

Artifact is not primarily a watermark algorithm demo.

Artifact is a creator-facing platform built around:

```text
Register artwork
↓
Create provenance record
↓
Generate watermarked copy
↓
Store record in registry
↓
Verify suspected image against selected record
↓
Preserve verification history
↓
Export technical report
```

The current DWT-QIM engine is the MVP’s baseline technical backend.

The broader platform identity should remain independent of any single watermarking method.

---

# 3. Brand Positioning

## Positioning Statement

> For digital artists and creators who want a clearer way to preserve and verify the provenance context of their work, Artifact is a registry-centered provenance platform that combines artwork records, invisible watermarking, selected-record verification, history, and technical reporting in one creator-facing workflow.

## Primary Differentiator

Artifact should feel like:

> **a creative workflow tool with technical verification capabilities**

not:

> **a forensic laboratory dashboard with an art theme**

This distinction is important.

The platform should balance:

- creator friendliness;
- technical credibility;
- visible personality;
- serious treatment of provenance;
- approachable visual language.

---

# 4. Target Audience

Primary audience:

- digital illustrators;
- independent artists;
- student artists;
- creative professionals;
- creators who distribute visual work online.

Secondary audience:

- researchers;
- evaluators;
- capstone testers;
- creative teams;
- future studio users.

## Audience Mindset

Users may care about:

- keeping a recognizable record of their work;
- distributing watermarked copies;
- checking suspected reposts;
- preserving technical verification records;
- avoiding overly complicated security software.

The UI should not assume users are watermarking experts.

---

# 5. Brand Purpose

> Help digital creators maintain clearer provenance records for their work through an accessible platform that combines registry management and technical watermark-based verification.

---

# 6. Vision

> To make digital artwork provenance workflows more accessible to creators through practical, transparent, and extensible verification tools.

---

# 7. Mission

> To develop creator-centered software that helps register, trace, and technically verify digital artworks through registry-backed workflows and modular watermarking mechanisms.

---

# 8. Brand Values

## 8.1 Creator-First

The platform should feel designed for artists, not merely adapted from a research interface.

## 8.2 Transparent

Do not overclaim certainty.

Show technical results honestly.

## 8.3 Practical

Prefer usable workflows over unnecessary complexity.

## 8.4 Traceable

Records, verification events, and reports should feel organized and inspectable.

## 8.5 Extensible

The watermarking engine may evolve without changing the platform identity.

## 8.6 Playful with Restraint

The company identity may be expressive and artistic, but the product must still feel credible enough for provenance and verification.

---

# 9. Brand Personality

Artifact should feel:

- **creative**
- **confident**
- **technical**
- **playful**
- **bold**
- **clear**
- **independent**
- **artist-aware**

Artifact should not feel:

- childish;
- corporate-banking-like;
- sterile;
- cyberpunk;
- overly academic;
- luxury-minimalist;
- generic SaaS blue;
- excessively futuristic.

## Personality Balance

Use this approximate balance:

```text
Creative        30%
Technical       25%
Confident       20%
Playful         15%
Formal          10%
```

The product should have visible personality without sacrificing readability.

---

# 10. Visual Inspiration Source

The primary visual reference is the team’s CSRP defense slide cover.

Key visual cues from the supplied reference:

- near-black dominant field;
- strong geometric blocks;
- saturated red;
- warm yellow;
- bright cyan;
- oversized bold typography;
- layered shapes;
- asymmetrical composition;
- expressive but controlled energy.

Purple from the old presentation should **not** be carried into the new Artifact product identity.

Reason:

- the user explicitly prefers a four-color core;
- purple risks making the system resemble unrelated fintech/mobile branding;
- removing purple gives Artifact a more distinctive black/red/yellow/cyan identity.

---

# 11. Core Color Palette

The values below are approximated from the supplied CSRP slide screenshot and refined for UI use.

They should be treated as the working digital brand palette.

## 11.1 Artifact Black — Primary

```text
Name: Artifact Black
Hex:  #12141C
RGB:  18, 20, 28
```

Role:

- main application background;
- sidebar;
- top-level navigation;
- dark cards;
- hero sections;
- high-emphasis surfaces.

This is the dominant brand color.

Do not replace it with generic pure black everywhere.

Use `#12141C` as the primary dark.

---

## 11.2 Signal Red — Primary Accent

```text
Name: Signal Red
Hex:  #C8102E
RGB:  200, 16, 46
```

Role:

- important primary CTA accents;
- brand marks;
- active emphasis;
- strong section accents;
- destructive actions only when contextually appropriate.

Do not use red for every button.

Red should feel intentional and energetic.

---

## 11.3 Studio Yellow — Secondary Accent

```text
Name: Studio Yellow
Hex:  #FFC72C
RGB:  255, 199, 44
```

Role:

- highlights;
- badges;
- illustration accents;
- onboarding emphasis;
- selected statistics;
- playful decorative shapes.

Yellow should provide warmth and artist energy.

Avoid large blocks of yellow behind long text.

---

## 11.4 Trace Cyan — Secondary Accent

```text
Name: Trace Cyan
Hex:  #07BED5
RGB:  7, 190, 213
```

Role:

- technical emphasis;
- links;
- active states;
- verification-related accents;
- informational statuses;
- data highlights.

Cyan represents the technical / traceability side of the brand.

---

# 12. Supporting Neutral Colors

The brand has four core colors, but the UI still requires functional neutrals.

These are not additional brand accents.

## Canvas White

```text
Hex: #F7F7F2
```

Use for:

- light text on dark surfaces;
- light page variants;
- cards where needed.

## Soft Gray

```text
Hex: #B8BDC7
```

Use for:

- secondary text;
- metadata;
- inactive labels.

## Graphite

```text
Hex: #242833
```

Use for:

- secondary dark surfaces;
- cards;
- form fields;
- table rows.

## Border Gray

```text
Hex: #3A3F4B
```

Use for:

- subtle borders;
- separators;
- input outlines.

---

# 13. Semantic UI Colors

Do not force every semantic state into the core brand palette.

Use restrained functional colors.

## Success

```text
#2DBE78
```

Use for:

- confirmed successful operations;
- verified success states.

## Warning

Prefer Studio Yellow where readable.

Use darker text on yellow surfaces.

## Error

Use Signal Red.

## Information

Use Trace Cyan.

## Neutral / Unknown

Use Soft Gray.

Important:

Do not communicate state through color alone.

Always pair color with:

- text;
- icon;
- label.

---

# 14. Color Usage Ratio

Recommended overall UI distribution:

```text
Artifact Black / dark neutrals   60–70%
Canvas White / light neutrals    15–20%
Signal Red                        5–10%
Studio Yellow                     3–8%
Trace Cyan                        3–8%
```

The app should not look like all four accent colors are competing equally.

Black is the visual anchor.

---

# 15. Typography System

Use two primary typefaces.

## 15.1 Display / Main Headings

### Recommended: Bricolage Grotesque

Role:

- page titles;
- major headings;
- hero text;
- dashboard headline numbers;
- key empty-state statements;
- marketing-style labels.

Why it fits:

- expressive without looking childish;
- has a handmade/editorial personality;
- feels more artistic than a generic geometric sans;
- still credible for a serious software product.

Desired tone:

> playful like an art studio, structured like a modern product.

### Font weights

Use primarily:

- 600
- 700
- 800

Avoid thin display weights.

---

## 15.2 UI / Body / Subheading Typeface

### Recommended: Manrope

Role:

- navigation;
- buttons;
- body text;
- table text;
- form labels;
- metadata;
- verification details;
- technical reports displayed in UI.

Why it fits:

- highly readable;
- modern;
- neutral enough to balance the expressive heading font;
- compatible with technical data and creator-facing interfaces.

### Font weights

Use:

- 400 body;
- 500 labels;
- 600 controls;
- 700 strong subheadings.

---

# 16. Typography Fallbacks

Display:

```css
font-family:
  "Bricolage Grotesque",
  "Arial Black",
  sans-serif;
```

UI:

```css
font-family:
  "Manrope",
  "Segoe UI",
  Arial,
  sans-serif;
```

Do not introduce a third primary font.

Monospace may be used only for:

- technical payload hashes;
- IDs where beneficial;
- debug-only data.

---

# 17. Typography Hierarchy

## Display XL

Use for:

- landing/hero;
- major product identity.

Suggested:

```text
48–64 px desktop
36–44 px tablet
30–36 px mobile
```

## H1

```text
36–44 px desktop
30–34 px mobile
```

## H2

```text
28–34 px
```

## H3

```text
22–26 px
```

## UI Subheading

```text
18–20 px
```

## Body

```text
15–17 px
```

## Metadata

```text
13–14 px
```

Do not use extremely tiny text for technical details.

---

# 18. Logo Direction

A final logo is not defined in this document.

However, future logo work should follow these rules.

## Product Logo Direction

Artifact should avoid:

- generic shield logos;
- padlocks;
- fingerprint icons;
- blockchain cubes;
- AI brain icons;
- generic water-drop watermark symbols.

Preferred visual concepts:

- layered artwork frame;
- registration mark;
- trace line;
- crop corner;
- canvas/document mark;
- hidden mark revealed through layering;
- abstract `A` built from record + image geometry.

## Company Logo Direction

ChickenScratch Co. may be more playful than Artifact.

Possible concepts:

- rough line mark;
- scribble;
- imperfect sketch;
- chicken-scratch line texture;
- handwritten accent.

Do not make Artifact itself visually comedic.

---

# 19. Shape Language

Use:

- bold geometric blocks;
- clipped corners;
- layered rectangles;
- asymmetrical accent shapes;
- slightly irregular artistic framing;
- clean cards with occasional expressive corner treatment.

Avoid:

- excessive glassmorphism;
- floating neon blobs;
- generic purple gradients;
- pill-shaped everything;
- overly rounded fintech UI.

## Border Radius

Recommended:

```text
Small controls:  8 px
Cards:          12 px
Large panels:   16 px
Hero blocks:    20 px max
```

Do not use fully rounded cards by default.

---

# 20. Layout Philosophy

The UI should feel:

> **structured platform + expressive art-direction**

Use:

- strong grids;
- clear spacing;
- asymmetric accents;
- deliberate empty space;
- visual grouping.

Do not reproduce the CSRP presentation slide literally inside the app.

The slide is a palette and energy reference, not a page-layout template.

---

# 21. Spacing System

Use a consistent 4 px base grid.

Recommended spacing tokens:

```text
4
8
12
16
20
24
32
40
48
64
```

Preferred component spacing:

- compact metadata gap: 4–8 px
- form field gap: 12–16 px
- card padding: 20–24 px
- section spacing: 32–48 px

---

# 22. Iconography

Recommended style:

- clean outline icons;
- consistent stroke width;
- minimal fill;
- simple metaphors.

Current Lucide-style icons are appropriate if already used.

Preferred icon themes:

- artwork/image;
- archive/registry;
- verification/check;
- report/document;
- clock/history;
- download;
- search/select.

Avoid mixing icon families.

---

# 23. Illustration and Image Style

Artifact is for digital artists, so imagery should feel creator-centered.

Preferred:

- drawing tablets;
- digital illustration canvases;
- stylus interactions;
- layered artwork;
- close-ups of creative process;
- abstract art textures.

Avoid:

- generic cybersecurity stock photos;
- hooded hackers;
- server racks;
- blockchain graphics;
- AI robot heads.

---

# 24. Graphic Motifs

Recommended motifs inspired by the defense slide:

## 24.1 Layered Color Blocks

Use angular blocks in:

- red;
- yellow;
- cyan.

## 24.2 Trace Lines

Thin cyan lines may suggest:

- provenance;
- connection;
- record linkage.

## 24.3 Registration Corners

Use crop-mark or frame-corner motifs to suggest:

- artwork registration;
- bounded records;
- image identity.

## 24.4 Scribble Accent

A restrained rough-line accent may connect Artifact to ChickenScratch Co.

Use sparingly.

---

# 25. UI Surface System

## Main App Background

Default:

```text
Artifact Black
#12141C
```

## Sidebar

Use:

```text
#0D0F15
```

Slightly darker than the main page.

## Primary Cards

Use:

```text
#1C202A
```

or:

```text
#242833
```

## Elevated Cards

Use subtle border plus minimal shadow.

Avoid heavy glow.

---

# 26. Button System

## Primary CTA

Preferred default:

- Signal Red background;
- Canvas White text.

Use for:

- Register Artwork;
- Verify Image;
- high-priority action.

## Secondary CTA

Preferred:

- transparent/dark background;
- cyan border or neutral border;
- light text.

## Highlight CTA

Yellow may be used sparingly for:

- onboarding;
- special creator action;
- non-destructive emphasis.

## Danger

Use red only with clear destructive copy.

Do not make every action red.

---

# 27. Navigation Style

Sidebar navigation should feel clean and stable.

Items:

- Dashboard
- My Artworks
- Register Artwork
- Verify Image
- Verification History

Recommended behavior:

- active item: cyan left rail or cyan text;
- hover: dark graphite elevation;
- primary create/register action may receive red emphasis;
- do not rainbow-color each navigation item.

---

# 28. Page-Level Visual Identity

## Dashboard

Feel:

- concise;
- editorial;
- creator workspace.

Use:

- dark base;
- large metrics;
- one or two accent colors per section;
- recent activity as a clean timeline/list.

## Register Artwork

Feel:

- creative studio intake flow.

Use:

- prominent upload canvas;
- clear image preview;
- minimal technical settings;
- strong primary CTA.

## My Artworks

Feel:

- registry + portfolio management.

Use:

- artwork thumbnail;
- ART-XXXX label;
- metadata;
- clear Verify and View actions.

## Artwork Detail

Feel:

- provenance record.

Use:

- artwork preview as hero;
- metadata in structured blocks;
- subtle technical section;
- clear download and verify actions.

## Verify Image

Feel:

- deliberate comparison workflow.

Critical order:

```text
1. Select ART-XXXX
2. Upload suspected image
3. Verify
```

Do not design it as automatic matching.

## Verification History

Feel:

- audit trail;
- record log.

Use:

- status;
- artwork ID;
- filename;
- date;
- technical detail link.

---

# 29. Status Badge System

Status labels should be concise and readable.

Examples:

- Watermark Embedded
- Verified Match
- Partial Detection
- No Valid Watermark
- Processing
- Failed

## Style

Use:

- compact badge;
- icon + text where useful;
- semantic color;
- no exaggerated glow.

Do not use overly technical internal enum names in UI.

---

# 30. Brand Voice

Artifact should sound:

- clear;
- calm;
- direct;
- creator-aware;
- technically honest.

Avoid:

- legal certainty;
- hype;
- AI buzzwords;
- cybersecurity fear language;
- overly academic phrasing.

---

# 31. Tone Examples

## Good

> Register an artwork and generate a watermarked copy for distribution.

> Verify this image against ART-0001.

> The extracted watermark did not sufficiently match the selected artwork record.

> This report is a technical verification record and does not constitute legal proof of ownership.

## Avoid

> Secure your art forever.

> Prove you own this artwork.

> AI-powered military-grade provenance.

> Guaranteed theft protection.

> We detected the original creator.

---

# 32. Product Copy Rules

Use:

- “Register Artwork”
- “My Artworks”
- “Verify Image”
- “Verification History”
- “Technical Report”

Prefer:

> “selected artwork record”

over:

> “detected artwork”

Prefer:

> “technical verification”

over:

> “proof of ownership”

Prefer:

> “suspected/reposted image”

over:

> “stolen image”

unless user context explicitly supports it.

---

# 33. Accessibility Rules

Minimum expectations:

- sufficient contrast;
- visible keyboard focus;
- labels for icon-only buttons;
- no color-only status meaning;
- readable text sizes;
- responsive layouts;
- alt text for meaningful images;
- upload errors announced clearly.

Do not sacrifice readability for artistic styling.

---

# 34. Responsive Design Rules

The app must work at:

- desktop;
- tablet;
- mobile.

## Mobile

- sidebar becomes drawer/bottom navigation as appropriate;
- primary actions remain visible;
- artwork cards stack;
- long filenames wrap;
- status badges remain inside containers;
- images preserve aspect ratio;
- no horizontal page overflow.

---

# 35. Motion and Interaction

Motion should feel:

- quick;
- purposeful;
- subtle.

Recommended:

```text
150–220 ms
```

Use motion for:

- hover feedback;
- card transitions;
- upload completion;
- status appearance;
- drawer navigation.

Avoid:

- long bouncy animations;
- excessive parallax;
- decorative loading animations.

---

# 36. Design Tokens

Recommended CSS variables:

```css
:root {
  --artifact-black: #12141C;
  --artifact-black-deep: #0D0F15;
  --artifact-graphite: #242833;
  --artifact-card: #1C202A;
  --artifact-border: #3A3F4B;

  --artifact-red: #C8102E;
  --artifact-yellow: #FFC72C;
  --artifact-cyan: #07BED5;

  --artifact-white: #F7F7F2;
  --artifact-gray: #B8BDC7;

  --artifact-success: #2DBE78;

  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;

  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;
  --space-10: 40px;
  --space-12: 48px;
  --space-16: 64px;
}
```

---

# 37. Typography Tokens

```css
:root {
  --font-display: "Bricolage Grotesque", "Arial Black", sans-serif;
  --font-ui: "Manrope", "Segoe UI", Arial, sans-serif;
}
```

Suggested:

```css
.page-title {
  font-family: var(--font-display);
  font-weight: 800;
}

.section-title {
  font-family: var(--font-display);
  font-weight: 700;
}

body,
button,
input,
textarea,
select {
  font-family: var(--font-ui);
}
```

---

# 38. Codex Refactor Rules

When refactoring the platform:

1. Inspect the existing UI before changing structure.
2. Preserve working workflows.
3. Do not add product features.
4. Do not add CNN/AI features.
5. Do not add login.
6. Do not add automatic artwork matching.
7. Replace inconsistent colors with brand tokens.
8. Replace inconsistent typography with the two-font system.
9. Preserve accessibility.
10. Keep the selected-record verification flow explicit.
11. Keep technical data secondary to the creator-facing workflow.
12. Use the brand accents selectively.
13. Do not convert the interface into a generic analytics dashboard.
14. Do not recreate the CSRP slide layout literally.
15. Treat the slide as color/energy inspiration only.

---

# 39. Component Refactor Priorities

Refactor in this order:

## Priority 1

- global app shell;
- sidebar;
- page backgrounds;
- typography;
- buttons;
- inputs.

## Priority 2

- dashboard metric cards;
- artwork cards;
- upload component;
- result card;
- status badges.

## Priority 3

- detail panels;
- verification history;
- technical report actions;
- empty states;
- loading states.

## Priority 4

- decorative motifs;
- micro-interactions;
- motion polish.

---

# 40. Brand Do / Do Not

## DO

- use black as dominant color;
- use red for strong emphasis;
- use yellow for warmth/playfulness;
- use cyan for technical traceability;
- use expressive headings;
- keep body text highly readable;
- emphasize creator workflows;
- show ART-XXXX IDs clearly;
- preserve technical credibility;
- use asymmetrical accents selectively.

## DO NOT

- reintroduce purple;
- use generic SaaS blue as primary;
- make every card brightly colored;
- use more than two primary fonts;
- add cyberpunk visuals;
- add shields/locks everywhere;
- overuse gradients;
- overuse rounded pills;
- claim ownership proof;
- make the interface look like a research experiment console.

---

# 41. Suggested Brand Taglines

These are optional.

Preferred:

> **Trace the work. Preserve the record.**

Alternative:

> **Give every artwork a record.**

Alternative:

> **Built for the work behind the work.**

Alternative:

> **Register. Mark. Verify.**

Do not hardcode a tagline across the app until the team selects one.

---

# 42. Recommended Default Tagline

For current branding work, use:

> **Trace the work. Preserve the record.**

Reason:

- connects to provenance;
- avoids legal proof claims;
- feels creator-focused;
- works with the registry concept;
- remains independent of a specific watermark engine.

---

# 43. Brand Summary

## Company

ChickenScratch Co.

## Product

Artifact

## Product Category

Digital artwork provenance verification platform

## Product Framing

Registry-centered platform with modular invisible watermarking

## Current Engine

DWT-QIM baseline

## Primary Color

Artifact Black

## Accent Colors

Signal Red  
Studio Yellow  
Trace Cyan

## Display Typeface

Bricolage Grotesque

## UI Typeface

Manrope

## Personality

Creative, technical, bold, playful, honest

## Core UX Principle

> Creator workflow first, technical verification second.

## Core Scope Principle

> Verify a suspected image against a selected artwork record.

## Future-Proofing Principle

> The watermark engine may change; the platform identity remains.
