# Capstone Coding Handoff
## Digital Artwork Provenance Management Prototype

> Purpose: This document is a coding handoff for Codex, GitHub Copilot, Claude Code, or another coding assistant.
>
> Priority: Build a working prototype within **one week**.
>
> Core rule: Prefer the **simplest implementation that works**. Do not add features outside this document unless explicitly requested.

---

# 1. Product Framing

Build a **digital artwork provenance management platform** that combines:

- artwork registration;
- registry-based payload storage;
- invisible watermark embedding;
- selected-record watermark verification;
- verification history;
- technical report export.

The watermarking engine is **one technical verification mechanism inside a broader provenance management workflow**.

The product is specifically scoped to:

- digital artworks;
- digital illustrations;
- creator-uploaded image files;
- suspected reposted or redistributed copies.

The system is **not** a general-purpose image authentication platform.

---

# 2. One-Sentence Product Definition

> A digital artwork provenance management prototype that allows creators to register artworks, embed invisible watermark payloads, store expected payloads in an artwork registry, verify suspected reposted images against a selected artwork record, review verification history, and export technical verification reports.

---

# 3. Important Product Claims and Limits

## The system may claim that it:

- registers digital artwork records;
- generates unique artwork IDs;
- stores expected watermark payloads;
- embeds invisible watermark payloads into images;
- attempts to recover embedded payloads from uploaded images;
- compares an extracted payload against a selected registry record;
- records verification attempts;
- generates technical verification reports.

## The system must NOT claim that it:

- legally proves copyright ownership;
- prevents art theft;
- prevents scraping;
- guarantees watermark survival under all transformations;
- automatically finds stolen artwork across the internet;
- performs reverse image search;
- automatically identifies the correct artwork from the entire registry;
- guarantees 100% extraction after degradation;
- detects AI-generated content;
- replaces legal chain-of-custody systems.

Recommended disclaimer:

> This system provides technical provenance support through registry records and watermark verification. A successful result does not constitute legal proof of ownership.

---

# 4. Scope for the One-Week Prototype

Implement only these core product features:

1. Artwork Registry
2. Register & Watermark Workflow
3. Selected-Record Verification
4. Verification History
5. Technical Verification Report
6. Lightweight Dashboard

Do not implement anything else unless all core features are already stable.

---

# 5. Explicit Non-Goals

Do NOT implement:

- automatic matching against all registry records;
- AI-based registry search;
- nearest-neighbor payload search;
- version history;
- public verification links;
- QR verification pages;
- shareable certificates as a separate system;
- portfolio folders or collections;
- monitoring queues;
- web scraping;
- reverse image search;
- social-media crawling;
- payment processing;
- subscription billing;
- full authentication system;
- team workspaces;
- cloud-scale deployment;
- degradation simulation as a normal user-facing page.

A degradation/attack suite may exist elsewhere for research or internal validation, but it should not be part of the normal product UI.

---

# 6. Existing Technical Context

The current codebase already contains or may contain technical modules such as:

- DWT-QIM embedding;
- classical extraction;
- optional CNN-assisted extraction;
- BER calculation;
- SSIM and PSNR;
- attack/degradation utilities;
- CLI tools;
- FastAPI backend;
- React frontend;
- configuration files;
- tests.

For this capstone prototype:

- reuse existing working modules where practical;
- do not rewrite stable watermarking code unnecessarily;
- isolate the watermarking engine behind a clear interface;
- build product workflow around the engine;
- treat robustness as a validated capability, not an assumed guarantee.

The current classical watermarking implementation may be used as the initial engine if it is stable enough for controlled embed-save-reload-extract operation.

---

# 7. Required Architecture

Use a modular architecture.

```text
Frontend
   ↓
Application / Service Layer
   ↓
Registry + Verification History + Reports
   ↓
Watermark Engine Interface
   ↓
Current Engine:
Classical DWT-QIM

Optional Future Engine:
CNN-Assisted Extractor
```

The UI must not depend directly on one specific extractor implementation.

---

# 8. Watermark Engine Interface

Create or preserve a standard interface.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class ExtractionResult:
    extracted_payload: Optional[str]
    ber: Optional[float]
    success: bool
    error_message: Optional[str] = None

class WatermarkEngine(ABC):
    @abstractmethod
    def embed(self, image, payload: str):
        """Return a watermarked image."""
        raise NotImplementedError

    @abstractmethod
    def extract(self, image, expected_payload: Optional[str] = None) -> ExtractionResult:
        """Extract payload and optionally compute BER against expected payload."""
        raise NotImplementedError
```

The rest of the system should call the interface rather than hardcoding DWT-QIM logic into page components.

---

# 9. Product Pages

Build only these pages:

```text
1. Dashboard
2. My Artworks
3. Register Artwork
4. Verify Image
5. Verification History
```

Optional:

```text
6. About / System Limitations
```

Do not create more pages unless required.

---

# 10. Page 1 — Dashboard

## Purpose

Provide a simple overview of the system.

## Required cards

Show:

- total registered artworks;
- total watermarked artworks;
- total verification attempts;
- successful matches;
- partial/corrupted detections;
- no-valid-watermark results.

## Recent activity

Show the latest 5–10 events:

```text
ART-0004 registered
VER-0012 verified against ART-0001
VER-0011 partial detection for ART-0003
```

## UI sketch

```text
--------------------------------------------------
Dashboard
--------------------------------------------------

[ 12 Registered Artworks ]
[ 10 Watermarked        ]
[  8 Verifications      ]

[  5 Matches ]
[  2 Partial ]
[  1 No Valid Watermark ]

Recent Activity
--------------------------------------------------
ART-0012 registered                         2h ago
VER-0008 matched ART-0004                  5h ago
VER-0007 partial result for ART-0002       1d ago
```

Keep this page simple.

---

# 11. Page 2 — My Artworks

## Purpose

Display all registered artwork records.

## Required table columns

```text
Artwork ID
Title
Creator
Registration Date
Watermark Status
Actions
```

## Required actions

- View
- Verify
- Download watermarked file, if available

## UI sketch

```text
--------------------------------------------------
My Artworks
--------------------------------------------------

[ + Register New Artwork ]

| ID       | Title              | Creator | Status      | Actions       |
|----------|--------------------|---------|-------------|---------------|
| ART-0001 | Frieren Sketch     | Kenny   | Watermarked | View | Verify |
| ART-0002 | Landscape Study    | Kenny   | Watermarked | View | Verify |
```

Clicking `Verify` should open the Verify page with that artwork preselected.

---

# 12. Page 3 — Register Artwork

## Purpose

Register an artwork, generate a unique ID and payload, embed the watermark, save the record, and let the user download the result.

## Required fields

- artwork title;
- creator name;
- image upload;
- optional notes.

## System-generated values

- artwork ID;
- payload;
- registration timestamp.

## Required flow

```text
User uploads original artwork
        ↓
User enters title and creator
        ↓
System validates image
        ↓
System generates ART-XXXX
        ↓
System generates unique payload
        ↓
System embeds payload
        ↓
System saves artwork record
        ↓
System saves watermarked output
        ↓
User downloads watermarked image
```

## UI sketch

```text
--------------------------------------------------
Register Artwork
--------------------------------------------------

Artwork Title
[____________________________]

Creator Name
[____________________________]

Notes (optional)
[____________________________]

Upload Artwork
[ Choose File ]

Preview
[ IMAGE PREVIEW ]

[ Register & Embed Watermark ]

--------------------------------------------------
Success

Artwork ID: ART-0001
Status: Watermark Embedded
Registry Record: Saved

[ Download Watermarked Image ]
[ View Artwork Record ]
```

Do not show raw payload by default. Put it inside a collapsible `Technical Details` section if needed.

---

# 13. Page 4 — Verify Image

## Purpose

Verify one suspected image against one selected artwork record.

## Important scope rule

The user must select the expected artwork record first.

Do NOT implement automatic matching across the full registry.

## Required flow

```text
User selects ART-0001
        ↓
System loads stored payload for ART-0001
        ↓
User uploads suspected/reposted image
        ↓
System extracts payload
        ↓
System compares extracted payload with ART-0001 expected payload
        ↓
System computes BER if possible
        ↓
System classifies result
        ↓
System stores verification event
        ↓
User may export report
```

## Required controls

- artwork dropdown or search-select;
- suspected image upload;
- verify button.

## UI sketch

```text
--------------------------------------------------
Verify Image
--------------------------------------------------

Verify Against
[ ART-0001 - Frieren Sketch ▼ ]

Suspected / Reposted Image
[ Choose File ]

Preview
[ IMAGE PREVIEW ]

[ Verify Watermark ]
```

## Result examples

```text
Verified Match

Artwork ID: ART-0001
Title: Frieren Sketch
Creator: Kenny
BER: 0.00
Processing Time: 0.81 sec

[ Download Technical Report ]
```

```text
Partial / Corrupted Watermark Detected

Artwork ID: ART-0001
BER: 0.06

The extracted payload is similar to the stored payload
but contains bit errors.

[ Download Technical Report ]
```

```text
No Valid Watermark Detected

Possible reasons:
- image was not watermarked by this system;
- wrong artwork record was selected;
- watermark was damaged beyond supported conditions;
- extraction failed.

[ View Technical Details ]
```

---

# 14. Verification Classification

Do not hardcode an arbitrary BER threshold without configuration.

Use:

```text
Verified Match:
BER = 0

Partial / Corrupted Watermark Detected:
0 < BER <= T

No Valid Watermark Detected:
BER > T or extraction failure
```

Where:

```text
T = configurable threshold
```

Store the threshold in configuration.

Example:

```python
VERIFICATION_BER_THRESHOLD = 0.15
```

This value is only a placeholder until internal validation determines a better threshold.

The UI should not claim that the threshold is scientifically final unless it has been validated.

---

# 15. Page 5 — Verification History

## Purpose

Store and display all verification attempts.

## Required table columns

```text
Verification ID
Artwork ID
Suspected Filename
Date/Time
Result
BER
Processing Time
Actions
```

## UI sketch

```text
--------------------------------------------------
Verification History
--------------------------------------------------

| ID       | Artwork  | File                 | Result   | BER  | Date |
|----------|----------|----------------------|----------|------|------|
| VER-0003 | ART-0001 | facebook_repost.jpg  | Match    | 0.00 | ... |
| VER-0002 | ART-0001 | compressed_copy.jpg  | Partial  | 0.06 | ... |
| VER-0001 | ART-0002 | random.jpg           | No Match | 0.48 | ... |
```

Required actions:

- View details
- Export report

---

# 16. Technical Verification Report

## Minimum fields

```text
Verification ID
Artwork ID
Artwork Title
Registered Creator
Registration Date
Suspected Filename
Verification Date
Verification Result
BER
Processing Time
System Version
Disclaimer
```

Recommended disclaimer:

> This report represents a technical watermark verification result produced by the system. It does not constitute legal proof of authorship, ownership, or copyright.

## Export format priority

1. CSV — must have
2. PDF — only if time allows

Do not build a separate certificate system.

---

# 17. Artwork Registry Data Model

Use SQLite if possible.

```sql
CREATE TABLE artworks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    artwork_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    creator_name TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    original_file_path TEXT,
    watermarked_filename TEXT,
    watermarked_file_path TEXT,
    payload TEXT NOT NULL,
    registration_date TEXT NOT NULL,
    watermark_status TEXT NOT NULL,
    notes TEXT
);
```

Suggested IDs:

```text
ART-0001
ART-0002
ART-0003
```

Generate sequentially.

---

# 18. Verification History Data Model

```sql
CREATE TABLE verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    verification_id TEXT UNIQUE NOT NULL,
    artwork_id TEXT NOT NULL,
    suspected_filename TEXT NOT NULL,
    suspected_file_path TEXT,
    verification_date TEXT NOT NULL,
    result_status TEXT NOT NULL,
    expected_payload TEXT,
    extracted_payload TEXT,
    ber REAL,
    processing_time_ms REAL,
    error_message TEXT,
    threshold_used REAL,
    FOREIGN KEY (artwork_id) REFERENCES artworks(artwork_id)
);
```

Suggested IDs:

```text
VER-0001
VER-0002
VER-0003
```

---

# 19. Recommended Backend Services

```text
services/
├── artwork_service.py
├── verification_service.py
├── report_service.py
└── dashboard_service.py
```

## artwork_service.py

- create artwork ID;
- generate payload;
- save artwork record;
- fetch artwork record;
- list artworks;
- update watermark status.

## verification_service.py

- retrieve selected artwork;
- retrieve expected payload;
- call extraction engine;
- calculate BER;
- classify result;
- save verification event.

## report_service.py

- create CSV report;
- optionally create PDF.

## dashboard_service.py

- calculate counts;
- fetch recent activity.

---

# 20. Suggested Project Structure

Adapt this to the existing repository rather than rebuilding everything unnecessarily.

```text
project/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── artworks.py
│   │   │   ├── verifications.py
│   │   │   ├── dashboard.py
│   │   │   └── reports.py
│   │   ├── services/
│   │   │   ├── artwork_service.py
│   │   │   ├── verification_service.py
│   │   │   ├── report_service.py
│   │   │   └── dashboard_service.py
│   │   ├── watermarking/
│   │   │   ├── interface.py
│   │   │   ├── classical_engine.py
│   │   │   └── cnn_engine.py
│   │   ├── database/
│   │   │   ├── db.py
│   │   │   ├── models.py
│   │   │   └── schema.sql
│   │   ├── utils/
│   │   │   ├── ids.py
│   │   │   ├── payloads.py
│   │   │   ├── images.py
│   │   │   └── validation.py
│   │   └── config.py
│   └── storage/
│       ├── originals/
│       ├── watermarked/
│       ├── suspected/
│       └── reports/
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── DashboardPage.jsx
│   │   │   ├── ArtworksPage.jsx
│   │   │   ├── RegisterArtworkPage.jsx
│   │   │   ├── VerifyImagePage.jsx
│   │   │   └── VerificationHistoryPage.jsx
│   │   ├── components/
│   │   │   ├── AppShell.jsx
│   │   │   ├── Sidebar.jsx
│   │   │   ├── StatCard.jsx
│   │   │   ├── ArtworkTable.jsx
│   │   │   ├── VerificationTable.jsx
│   │   │   ├── ImageUpload.jsx
│   │   │   ├── ResultCard.jsx
│   │   │   └── TechnicalDetails.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   └── App.jsx
│
├── tests/
├── README.md
└── requirements.txt
```

If the current app already uses React + FastAPI, keep that stack.

Do not migrate frameworks during the one-week sprint.

---

# 21. API Endpoints

## Dashboard

```http
GET /api/dashboard/summary
GET /api/dashboard/recent-activity
```

## Artworks

```http
GET    /api/artworks
GET    /api/artworks/{artwork_id}
POST   /api/artworks/register
GET    /api/artworks/{artwork_id}/download
```

## Verification

```http
POST /api/verifications
GET  /api/verifications
GET  /api/verifications/{verification_id}
```

## Reports

```http
GET /api/verifications/{verification_id}/report.csv
```

Optional:

```http
GET /api/verifications/{verification_id}/report.pdf
```

---

# 22. Example Registration API Behavior

Request:

```http
POST /api/artworks/register
Content-Type: multipart/form-data
```

Fields:

```text
title
creator_name
notes
image
```

Backend flow:

```text
validate image
↓
generate ART ID
↓
generate payload
↓
call watermark engine
↓
save original image
↓
save watermarked image
↓
save registry row
↓
return artwork record
```

Example response:

```json
{
  "artwork_id": "ART-0001",
  "title": "Frieren Sketch",
  "creator_name": "Canard Cyris Quisayang",
  "registration_date": "2026-07-06T18:30:00",
  "watermark_status": "embedded",
  "download_url": "/api/artworks/ART-0001/download"
}
```

Do not return the raw payload in normal responses.

---

# 23. Example Verification API Behavior

Request:

```http
POST /api/verifications
Content-Type: multipart/form-data
```

Fields:

```text
artwork_id
image
```

Backend flow:

```text
fetch artwork record
↓
load expected payload
↓
validate suspected image
↓
call extractor
↓
compute BER
↓
classify result
↓
generate VER ID
↓
save verification event
↓
return result
```

Example response:

```json
{
  "verification_id": "VER-0003",
  "artwork_id": "ART-0001",
  "result_status": "partial",
  "ber": 0.06,
  "processing_time_ms": 812,
  "threshold_used": 0.15
}
```

---

# 24. Payload Strategy

For the prototype:

- generate one unique payload per artwork record;
- use a fixed payload length compatible with the current engine;
- store the expected payload in the registry;
- do not expose it in the normal UI.

Example:

```python
import secrets

def generate_payload_hex(byte_length: int = 16) -> str:
    return secrets.token_hex(byte_length)
```

For 128 bits:

```text
16 bytes = 32 hex characters
```

Use the payload format required by the existing watermark engine.

---

# 25. UI/UX Design Rules

The UI should feel like a creator tool, not a research dashboard.

Use:

- clean layout;
- clear status badges;
- simple copy;
- strong visual hierarchy;
- image previews;
- minimal technical clutter.

Avoid exposing:

- delta sliders;
- seed inputs;
- raw coefficient settings;
- attack settings;
- statistical test controls.

Those may exist in developer settings but not in the normal prototype UI.

---

# 26. Navigation

Recommended sidebar:

```text
Dashboard
My Artworks
Register Artwork
Verify Image
Verification History
```

Footer:

```text
About
System Limitations
```

---

# 27. Status Labels

Use exactly these labels:

```text
Verified Match
Partial / Corrupted Watermark Detected
No Valid Watermark Detected
Extraction Failed
Unsupported Image
```

Do not use unvalidated confidence percentages.

---

# 28. Image Upload UX

For registration and verification:

- drag-and-drop;
- click-to-upload;
- preview;
- allowed formats shown;
- file size shown;
- validation error shown before processing.

Supported formats:

```text
PNG
JPG
JPEG
```

---

# 29. Loading and Error States

Every processing action must have:

- disabled button while running;
- loading spinner;
- clear success state;
- clear error state.

Examples:

```text
Embedding watermark...
Extracting payload...
Saving registry record...
Generating report...
```

---

# 30. Technical Details Drawer

Keep technical metrics out of the primary result view.

Use:

```text
[ Technical Details ▼ ]
```

Possible fields:

```text
BER
Processing time
Threshold used
Expected payload hash
Extracted payload hash
Engine name
Engine version
```

Do not expose raw payload unless needed.

---

# 31. Prototype Acceptance Criteria

## Artwork registration

- upload PNG/JPEG;
- enter title and creator;
- generate ART ID;
- generate payload;
- embed watermark;
- store artwork record;
- download watermarked image.

## Registry

- list artworks;
- view artwork details;
- launch verification from selected record.

## Verification

- select ART-XXXX;
- upload suspected image;
- retrieve expected payload;
- attempt extraction;
- compute BER when possible;
- classify result;
- store verification event.

## Verification history

- list events;
- open details;
- events survive app restart.

## Reports

- export CSV report.

## Dashboard

- summary counts are correct;
- recent activity is shown.

---

# 32. Minimum Testing Requirements

## Unit tests

Test:

- artwork ID generation;
- verification ID generation;
- payload generation;
- BER calculation;
- classification thresholds;
- database creation;
- artwork retrieval.

## Integration tests

Test:

```text
register → embed → save → retrieve
```

and:

```text
select artwork → upload suspected image → extract → classify → save history
```

## Critical round-trip test

Test:

```text
original
↓
embed
↓
save
↓
reload
↓
extract
```

The clean controlled workflow should be highly reliable before claiming the verifier is ready.

---

# 33. Internal Validation vs Product Features

Internal validation may test:

- clean watermarked copies;
- JPEG compression;
- resizing;
- negative images;
- wrong payloads.

This is for validating classification behavior.

It is not a required user-facing feature.

Do not add:

```text
Apply JPEG 70
Simulate Resize Attack
Run Degradation Suite
```

to the normal creator interface.

---

# 34. Monetization Framing

Do not implement billing in the one-week prototype.

Possible future tiers:

## Free
- limited registered artworks;
- basic verification;
- basic reports.

## Creator
- more artworks;
- batch tools;
- advanced history;
- enhanced reports.

## Studio
- shared registry;
- multiple users;
- API access;
- bulk processing.

For the current prototype, do not build payment logic.

---

# 35. One-Week Development Order

## Day 1 — Data layer and architecture
- inspect current codebase;
- preserve existing engine;
- add SQLite;
- create artwork table;
- create verification table;
- create ID utilities;
- create service boundaries.

## Day 2 — Artwork registration
- registration API;
- payload generation;
- embed integration;
- save registry record;
- watermarked image download.

## Day 3 — Registry UI
- My Artworks page;
- artwork table;
- artwork details;
- verify action.

## Day 4 — Verification flow
- selected-record verification;
- expected payload retrieval;
- extraction integration;
- BER calculation;
- status classification;
- history persistence.

## Day 5 — Verification history and reports
- history table;
- detail page/modal;
- CSV report.

## Day 6 — Dashboard and UX polish
- dashboard counts;
- recent activity;
- loading states;
- error states;
- responsive layout.

## Day 7 — Testing and bug fixing
- round-trip tests;
- invalid file tests;
- database persistence;
- report validation;
- demo preparation.

---

# 36. Coding Assistant Instructions

When modifying the repository:

1. Inspect the existing project first.
2. Reuse working modules.
3. Do not replace the current stack.
4. Do not rewrite the watermark engine unless required.
5. Keep changes modular.
6. Implement one feature at a time.
7. Run tests after each feature.
8. Avoid speculative abstractions.
9. Prefer simple SQLite persistence.
10. Do not add features outside this handoff.

---

# 37. First Task for Codex / Copilot

```text
Inspect the repository and identify the existing frontend, backend, watermark embedding, extraction, BER, database, and test modules.

Do not write code yet.

Return:
1. current architecture summary;
2. reusable modules;
3. missing modules required by this handoff;
4. conflicts between the current codebase and the target product architecture;
5. a minimal implementation plan ordered by dependency.

Do not propose features outside CAPSTONE_CODING_HANDOFF.md.
```

---

# 38. Second Task for Codex / Copilot

```text
Implement the persistent data layer for the capstone prototype.

Requirements:
- use SQLite;
- add artworks table;
- add verifications table;
- add ART-XXXX ID generation;
- add VER-XXXX ID generation;
- preserve the existing application stack;
- do not modify watermarking logic;
- include safe table initialization;
- add focused unit tests.

Return a summary of changed files and test results.
```

---

# 39. Third Task for Codex / Copilot

```text
Implement the Register Artwork workflow.

Requirements:
- title;
- creator name;
- optional notes;
- PNG/JPEG upload;
- generate artwork ID;
- generate engine-compatible unique payload;
- call existing watermark embedding module;
- save original and watermarked files;
- save artwork record in SQLite;
- return watermarked image download URL;
- hide raw payload from normal API responses;
- add validation and error handling.

Do not implement verification yet.
```

---

# 40. Fourth Task for Codex / Copilot

```text
Implement selected-record verification only.

Requirements:
- user must provide artwork_id;
- retrieve expected payload from registry;
- accept one suspected image;
- call existing extraction engine;
- compute BER against selected record payload;
- classify using configurable threshold;
- generate VER-XXXX;
- save event to verification history;
- return result status, BER, processing time, and report link.

Do not implement automatic matching against the full registry.
Do not add AI matching.
```

---

# 41. Fifth Task for Codex / Copilot

```text
Implement the required frontend pages:

1. Dashboard
2. My Artworks
3. Register Artwork
4. Verify Image
5. Verification History

Use the existing frontend framework.

UX rules:
- creator-facing design;
- hide technical parameters from normal users;
- image previews;
- clear loading states;
- clear result cards;
- status badges;
- responsive layout;
- technical details in collapsible sections.

Do not add extra pages.
```

---

# 42. Final Definition of Done

The system is ready for prototype demonstration when a user can:

```text
1. Register a digital artwork
2. Receive ART-0001
3. Embed and download a watermarked file
4. See ART-0001 in My Artworks
5. Select ART-0001
6. Upload a suspected/reposted image
7. Run verification
8. Receive a clear result
9. See the event in Verification History
10. Export a technical report
```

That is the complete MVP.

Do not expand scope until this flow works end to end.
