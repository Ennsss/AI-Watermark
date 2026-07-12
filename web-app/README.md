# Artifact Web Application

Artifact is a local, single-user digital artwork provenance registry. It registers an artwork, generates an `ART-XXXX` identifier and system-managed 128-bit payload, embeds that payload with the existing DWT-QIM engine, and stores a watermarked distribution copy. Verification always compares an uploaded suspected image against a user-selected active registry record and stores a `VER-XXXX` event.

It is not a reverse-image-search service, legal ownership adjudicator, generic custom-payload editor, or authentication system.

## Current product workflow

```text
Register artwork
-> generate registry ID and internal payload
-> embed and save the watermarked copy
-> select an active artwork
-> verify a suspected/reposted image
-> review history and export a technical CSV report
```

## Features

- Dashboard metrics for active artworks and verification outcomes
- Responsive, searchable My Artworks registry
- Combined artwork registration and DWT-QIM embedding
- Selected-record image verification with BER classification
- Searchable verification history and individual technical CSV reports
- Expected/extracted payload comparison in collapsible technical details
- Archive-style artwork unregistration that preserves files, verification history, and reports
- Artwork-level creator attribution (separate from any future account identity)

Raw payloads remain system-managed. They are not accepted in the normal registration UI or shown on ordinary artwork cards.

Registration metadata is limited to 120 characters for artwork titles, 80 for creator names, and 1,000 for optional notes. The frontend and API enforce the same limits.

## Architecture

```text
web-app/
  backend/     FastAPI, SQLAlchemy/SQLite, registry/verification/report services
  frontend/    React application
```

The prototype uses SQLite and applies its small `archived_at` compatibility migration during backend startup. Archived artwork remains resolvable for historical detail/report relationships but is excluded from the active artwork list, dashboard active count, and new-verification selection.

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm

## Setup

From `web-app` on Windows:

```powershell
.\setup.ps1
```

Or install and run each side manually:

```powershell
cd backend
py -m pip install -r requirements.txt
py -m pip install -e ..\..\
py main.py
```

```powershell
cd frontend
npm install
npm start
```

The frontend defaults to `http://localhost:3000` and the API to `http://localhost:8000`. Set `REACT_APP_API_URL` when the backend uses a different origin.

## Product API

### Dashboard

- `GET /api/dashboard/summary` - active artwork and verification-result counts
- `GET /api/dashboard/recent-activity` - recent active registrations and verification events

### Artwork registry

- `POST /api/artworks/register` - register metadata, generate the payload, and embed/save a watermarked copy
- `GET /api/artworks` - list active artwork records
- `GET /api/artworks/{artwork_id}` - retrieve a record, including archived records needed by historical workflows
- `GET /api/artworks/{artwork_id}/watermarked` - download the preserved watermarked copy
- `PATCH /api/artworks/{artwork_id}/archive` - unregister from active workflows without deleting provenance data or files

### Verification and reports

- `POST /api/verifications` - compare an upload against a selected active artwork
- `GET /api/verifications` - list verification events
- `GET /api/verifications/{verification_id}` - retrieve summary and technical comparison fields
- `GET /api/verifications/{verification_id}/report.csv` - download an individual technical report
- `GET /api/reports/verifications.csv` - download the compact history export

Older `/api/embed`, `/api/extract`, `/api/detect`, and `/api/remove` research/testing routes remain for backward compatibility. They are not the primary product workflow and the React portal does not expose custom payload entry.

## Validation

```powershell
cd frontend
npm run build
```

```powershell
py -m pytest
```

## Limitations

- Local/single-user prototype; authentication and ownership authorization are intentionally deferred
- Verification checks only the selected registry record; it does not search the whole registry
- Unregistration has no restore UI in this pass
- Watermark robustness depends on the image transformation and current DWT-QIM configuration
- Reports are technical evidence and are not legal proof of authorship, ownership, or copyright

For broader repository and research-engine details, see the root `README.md` and `docs/codebase_guide.md`.
