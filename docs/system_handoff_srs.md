# AI Watermark Provenance Platform - SRS and Handoff Context

## 1. Purpose

This document describes the intended finished behavior of the AI Watermark Provenance Platform. It is written as handoff context for creating a test plan, test cases, and a user manual.

The system is a local web application for registering digital artworks, embedding invisible watermarks, verifying suspected/reposted images against registered artwork records, and producing technical verification records. The platform supports creator provenance workflows, but it does not claim to provide legal proof of copyright or ownership.

## 2. Product Summary

The platform lets a user:

- Register an artwork with title, creator name, optional notes, and an uploaded image.
- Automatically embed a 128-bit invisible watermark into the artwork using the backend watermarking engine.
- Save the artwork record in a SQLite registry.
- Store the original and watermarked image files on disk.
- View all registered artworks in a registry called "My Artworks".
- Open a detailed artwork record with metadata, watermark status, embedded watermark preview, and download actions.
- Verify a suspected/reposted image against a selected registered artwork.
- Classify the verification result as Verified Match, Partial Detection, or No Valid Watermark.
- View verification history and details for each verification event.
- Download CSV technical reports for verification records.
- Review dashboard metrics and recent system activity.

The critical path is:

1. Start backend and frontend.
2. Register an artwork.
3. Download or inspect the watermarked image.
4. Verify an image against the registered artwork.
5. Review the verification result.
6. Review the same event in Verification History.
7. Download the technical report.

## 3. Users and Roles

The current system is a single-user prototype with no authentication. The assumed user is a creator, researcher, evaluator, or capstone tester who is working locally.

Primary user goals:

- Establish a registry record for an artwork.
- Generate a watermarked copy for distribution.
- Check whether a suspected image corresponds to a registered artwork.
- Preserve a technical verification record.

Out of scope for the current system:

- Multi-user accounts.
- Login/logout.
- Role-based permissions.
- Cloud storage.
- Legal claim adjudication.
- Public marketplace or public verification portal.

## 4. Technical Overview

Frontend:

- React 18 application.
- React Router routes.
- Axios API calls.
- Lucide React icons.
- CSS files per feature/page.
- Default frontend URL: `http://localhost:3000`.
- Default backend API URL: `http://localhost:8000`, configurable through `REACT_APP_API_URL`.

Backend:

- FastAPI application.
- Uvicorn server.
- SQLite registry database.
- SQLAlchemy models.
- PIL, OpenCV, and NumPy for image handling.
- Existing watermarking implementation from `src/watermark`.
- Default backend URL: `http://localhost:8000`.

Storage:

- Database file: `web-app/backend/watermark_registry.db`.
- Original images: `web-app/backend/storage/originals`.
- Watermarked images: `web-app/backend/storage/watermarked`.
- Suspected verification uploads: `web-app/backend/storage/suspected`.

Important storage expectation:

- The database stores image filenames and file paths, not image blobs.
- Image preview and download require the corresponding image file to exist on disk.
- If the project is moved between machines, stored absolute paths may become stale. The intended behavior is that the backend should still resolve watermarked images by filename from the current `storage/watermarked` directory when possible.

## 5. Watermarking Behavior

The product watermarking workflow uses:

- Wavelet: `haar`.
- DWT level: `2`.
- Target subbands: `LH2` and `HL2`.
- Default quantization delta: `16.0`.
- Payload size: `128 bits`, stored as 16 bytes / 32 hex characters.
- Fixed seed: `42`.
- Registered artwork images are resized to `512 x 512` before embedding/verifying.
- Supported image uploads are any browser/HTTP image MIME type accepted by the app, with intended practical support for PNG and JPEG.
- Maximum upload size: `50 MB`.

Registration creates a unique payload for each artwork. Verification extracts a 128-bit payload from the suspected image and compares it against the stored payload for the selected artwork.

## 6. Result Classification

Verification results are based on BER, or Bit Error Rate.

Expected labels:

- `match`: displayed as "Verified Match".
- `partial`: displayed as "Partial Detection" or "Partial / Corrupted Watermark Detected" depending on context.
- `no_match`: displayed as "No Valid Watermark" or "No Valid Watermark Detected".
- Error/extraction failure states should be handled gracefully and should not crash the UI.

Classification rules:

- If BER is exactly `0`, result is `match`.
- If BER is greater than `0` and less than or equal to the threshold stored for the verification, result is `partial`.
- If BER is greater than the stored threshold, result is `no_match` for the selected artwork record.
- If extraction cannot produce a result, the system should show a clear failure/error state.

The current committed boundary is `0.15` under policy `provisional-2026-07`. It is a provisional legacy boundary, not a CSRP-approved acceptance threshold. A replacement requires labeled positive and negative calibration evidence and explicit team approval. Historical rows retain their original status and `threshold_used`.

## 7. Data Model

### Artwork

Stored in the `artworks` table.

Fields:

- `id`: internal integer primary key.
- `artwork_id`: public unique artwork ID, for example `ART-0001`.
- `title`: required artwork title.
- `creator_name`: required creator name.
- `original_filename`: original upload filename.
- `original_file_path`: stored file path for original image.
- `watermarked_filename`: generated watermarked filename, for example `ART-0001_watermarked.png`.
- `watermarked_file_path`: stored file path for watermarked image.
- `payload`: unique 128-bit payload stored as hex.
- `registration_date`: timestamp.
- `watermark_status`: expected value after registration is `embedded`.
- `notes`: optional notes.

### Verification

Stored in the `verifications` table.

Fields:

- `id`: internal integer primary key.
- `verification_id`: public unique verification ID, for example `VER-0005`.
- `artwork_id`: linked artwork ID.
- `suspected_filename`: uploaded suspected image filename.
- `suspected_file_path`: stored path for suspected image.
- `verification_date`: timestamp.
- `result_status`: `match`, `partial`, `no_match`, `error`, or extraction failure state.
- `expected_payload`: artwork payload from registry.
- `extracted_payload`: payload extracted from suspected image.
- `ber`: Bit Error Rate.
- `processing_time_ms`: processing duration.
- `error_message`: optional error detail.
- `threshold_used`: classification threshold.
- `policy_version`: classification-policy identifier for new verification rows; older rows may be null.

## 8. Main UI Areas

### 8.1 Global Layout

The application uses a sidebar navigation layout.

Navigation items:

- Dashboard.
- My Artworks.
- Register Artwork.
- Verify Image.
- Verification History.

The UI should be responsive on desktop, tablet, and mobile. Mobile layout should not overlap, clip important controls, or hide primary actions.

The sidebar includes a disclaimer:

"This system provides technical provenance support through registry records and watermark verification. A successful result does not constitute legal proof of ownership."

### 8.2 Dashboard

Purpose:

- Give a high-level summary of registry and verification activity.

Expected content:

- Registered Artworks count.
- Watermarked count.
- Total Verifications count.
- Verified Matches count.
- Partial Detections count.
- No Valid Watermark count.
- Recent Activity list.

Expected behavior:

- On page load, fetch dashboard summary and recent activity.
- Show a loading state while fetching.
- Show an error message if dashboard data cannot be loaded.
- Recent activity should combine recent artwork registrations and verification events, sorted newest first.
- Empty activity should display a clear empty state.

### 8.3 Register Artwork

Purpose:

- Create a registry record and generate a watermarked copy.

Required inputs:

- Artwork Title.
- Creator Name.
- Upload Artwork image.

Optional inputs:

- Notes.

Expected upload behavior:

- User may click to select an image or drag and drop an image.
- File must be an image.
- File size must be at most 50 MB.
- Show local image preview before submission.
- Show selected filename.

Expected submission behavior:

- Validate required fields before sending.
- Disable or indicate loading while processing.
- Button text should indicate registration and embedding are in progress.
- Backend creates an artwork record, embeds watermark, stores files, and returns a watermarked image.
- On success, show:
  - Artwork ID.
  - Title.
  - Creator.
  - Watermark embedded status.
  - Watermarked image preview.
  - Download Watermarked Image action.
  - View Artwork Record action.
  - Register Another Artwork action.
- On error, show a clear error message.

Correct UI expectation:

- The watermarked image preview should stay within its container and preserve aspect ratio.
- Long input values should not break the layout.
- The status label should render as human-readable text, for example "Watermark Embedded".

### 8.4 My Artworks

Purpose:

- Show the artwork registry.

Expected content per artwork card:

- Artwork ID.
- Title.
- Creator.
- Registration date.
- Watermark status.
- View details action.
- Verify action.

Expected behavior:

- Fetch all artworks on page load.
- Show loading state while fetching.
- Show empty state when no artworks are registered.
- Show error state if artworks cannot be loaded.
- "View details" opens the artwork detail page.
- "Verify" opens the Verify Image page with that artwork pre-selected.

Correct UI expectation:

- Cards should remain visually contained and readable with long artwork titles or creator names.
- Status badges should not overflow their cards.

### 8.5 Artwork Detail

Purpose:

- Provide a complete provenance snapshot for one registered artwork.

Expected content:

- Back to artworks action.
- Artwork title.
- Artwork ID.
- Watermark status.
- Embedded Watermark Preview.
- Download image action.
- Creator.
- Registration date/time.
- Notes.
- Verify This Artwork action.
- Back to Registry action.

Embedded Watermark Preview:

- The preview should display the watermarked image associated with the artwork.
- The preferred preview source is the direct image endpoint `/api/artworks/{artwork_id}/watermarked`.
- Base64 returned from the artwork detail API may be used as fallback.
- If the watermarked file is missing, show "Watermarked image preview unavailable."
- The preview should fit within the card, preserve aspect ratio, and not overflow the viewport.

Download behavior:

- Download should retrieve the generated watermarked PNG.
- The filename should be the watermarked filename, for example `ART-0001_watermarked.png`.

Correct storage expectation:

- The image must exist in `web-app/backend/storage/watermarked` or at the valid stored path.
- The database itself does not store image bytes.

### 8.6 Verify Image

Purpose:

- Compare a suspected/reposted image against a selected registered artwork.

Inputs:

- Artwork selection dropdown.
- Suspected/reposted image upload.

Expected behavior:

- If opened from an artwork card/detail, the artwork should be pre-selected using the `artwork_id` query parameter.
- User may click to select or drag and drop an image.
- File must be an image.
- File size must be at most 50 MB.
- Show local preview of suspected image.
- Validate that both artwork and image are selected before submission.
- Show loading state while extracting/verifying.

Expected result display:

- Result classification card:
  - Verified Match.
  - Partial / Corrupted Watermark Detected.
  - No Valid Watermark Detected.
- Verification ID.
- Artwork ID.
- BER.
- Processing time.
- Human-readable result message.
- Download Technical Report action.
- Verify Another Image action.

Correct UI expectation:

- Result cards should use clear visual hierarchy and status coloring.
- BER should be shown with reasonable precision, for example 4 decimal places.
- Processing time should be shown in milliseconds with reasonable precision.
- Error cases should show a clear message and keep the form usable.

### 8.7 Verification History

Purpose:

- Show all previous verification records and allow inspection/report download.

Expected content per verification card:

- Verification ID.
- Suspected filename.
- Result badge.
- Artwork ID.
- Verification date.
- BER.
- View details action.
- Download report action.

Expected result badges:

- Verified Match.
- Partial Detection.
- No Valid Watermark.

Correct UI expectation for long filenames:

- Long suspected filenames must not push result badges outside the card.
- The result badge must remain inside the card and visible.
- The filename area may be horizontally scrollable on desktop so the full filename can be inspected.
- On mobile, long filenames may wrap within the card.
- The filename should also be available through hover/title behavior where appropriate.

Expected detail view:

- Back to History action.
- Verification Details heading.
- Result pill.
- Verification ID.
- Artwork ID.
- Artwork title, if available.
- Creator, if available.
- Suspected filename.
- Verification date/time.
- BER.
- Processing time.
- Download Technical Report action.
- Back to History action.

Correct UI expectation for detail values:

- Long filenames and values should wrap or break within their panel.
- Detail panels should not overflow horizontally.

### 8.8 Technical Report CSV

Purpose:

- Provide a downloadable technical record for a verification event.

Expected CSV contents:

- Report title.
- Verification ID.
- Verification date.
- Verification result.
- Artwork ID.
- Artwork title.
- Creator.
- Registration date.
- Suspected filename.
- BER.
- Processing time.
- Threshold used.
- Error message, if applicable.
- Disclaimer.

Expected disclaimer:

"This report represents a technical watermark verification result produced by the system. It does not constitute legal proof of authorship, ownership, or copyright."

## 9. API Endpoints

Base URL: `http://localhost:8000`.

### Health

`GET /health`

Expected response:

```json
{
  "status": "healthy",
  "service": "AI Watermark API"
}
```

### Dashboard

`GET /api/dashboard/summary`

Returns dashboard counts.

`GET /api/dashboard/recent-activity`

Returns recent artwork and verification events.

### Artworks

`POST /api/artworks/register`

Multipart form data:

- `title`: required.
- `creator_name`: required.
- `notes`: optional.
- `file`: required image.

Expected success response includes:

- `status`.
- `artwork_id`.
- `title`.
- `creator_name`.
- `registration_date`.
- `watermark_status`.
- `image`: base64 PNG.
- `format`.

`GET /api/artworks`

Returns all registered artworks.

`GET /api/artworks/{artwork_id}`

Returns detail for one artwork, including watermarked preview fields.

`GET /api/artworks/{artwork_id}/watermarked`

Returns the watermarked PNG as an image/file response.

### Verifications

`POST /api/verifications`

Multipart form data:

- `artwork_id`: required.
- `file`: required suspected image.

Expected success response includes:

- `status`.
- `verification_id`.
- `artwork_id`.
- `result_status`.
- `ber`.
- `processing_time_ms`.
- `threshold_used`.
- `message`.

`GET /api/verifications`

Returns all verification records, newest first. Optional query parameter: `artwork_id`.

`GET /api/verifications/{verification_id}`

Returns detail for one verification.

`GET /api/verifications/{verification_id}/report.csv`

Returns a CSV technical report.

### Legacy/Research Endpoints

The backend also includes lower-level watermark utility endpoints:

- `POST /api/extract`.
- `POST /api/detect`.
- `POST /api/remove`.
- `GET /api/config`.

The code also contains a legacy embed handler function, but the product-facing registration workflow should use `POST /api/artworks/register`.

## 10. Validation and Error Handling Expectations

File upload validation:

- Reject non-image files.
- Reject files larger than 50 MB.
- Reject images that cannot be decoded.
- Resize valid images to 512 x 512 for watermark operations.

Required field validation:

- Registration requires title, creator name, and image file.
- Verification requires selected artwork and suspected image.

API error behavior:

- Missing artwork should return 404.
- Missing verification should return 404.
- Missing watermarked image file should return 404 for the download/preview endpoint.
- Processing errors should return a clear message.

Frontend error behavior:

- Loading states should be visible.
- Error messages should be visible and understandable.
- A failed operation should not clear valid user input unless intentionally resetting after success.
- The user should be able to retry after an error.

## 11. Important Test Planning Notes

Critical path tests should cover:

1. Dashboard loads with zero records.
2. Register artwork with valid PNG.
3. Register artwork with valid JPEG.
4. Reject non-image upload during registration.
5. Reject oversized upload during registration.
6. Registration success shows watermarked preview and download action.
7. My Artworks lists the new record.
8. Artwork detail shows metadata and embedded watermark preview.
9. Artwork detail download returns a PNG.
10. Verify the downloaded watermarked image against the same artwork.
11. Verified Match result appears when BER is 0.
12. Verification history shows the new verification event.
13. Verification detail opens from history.
14. CSV report downloads and contains expected fields/disclaimer.
15. Verify an unrelated image and expect No Valid Watermark.
16. Verify a degraded/corrupted watermarked image and expect Partial Detection when BER is within threshold.
17. Long suspected filename remains contained in Verification History.
18. Long suspected filename can be inspected through the scrollable filename area on desktop.
19. Mobile layout wraps long filenames without horizontal page overflow.
20. Backend unavailable produces clear dashboard/data loading errors.

Edge cases:

- Empty registry.
- Empty verification history.
- Missing watermarked file on disk.
- Stale image path in database after moving project folder.
- Long artwork title.
- Long creator name.
- Long suspected filename.
- Repeated verifications against same artwork.
- Invalid artwork ID in URL.
- Invalid verification ID in URL.
- Very small or very large valid image dimensions.

## 12. User Manual Outline Suggestions

Recommended user manual sections:

1. Introduction and purpose.
2. Important disclaimer about technical verification vs legal proof.
3. System requirements.
4. Starting the backend and frontend.
5. Navigating the sidebar.
6. Registering an artwork.
7. Downloading the watermarked image.
8. Viewing My Artworks.
9. Opening artwork details.
10. Understanding the embedded watermark preview.
11. Verifying a suspected image.
12. Understanding verification outcomes.
13. Viewing verification history.
14. Downloading technical reports.
15. Troubleshooting common issues.

Suggested user-facing explanations:

- "Verified Match" means the extracted watermark exactly matches the registered payload.
- "Partial Detection" means the watermark appears related to the registered payload but contains bit errors, possibly due to compression, resizing, or degradation.
- "No Valid Watermark" means the extracted watermark does not sufficiently match the registered artwork payload.
- "BER" means Bit Error Rate, the fraction of extracted payload bits that differ from the registered payload.
- "Processing Time" is the backend time needed for extraction/comparison.

Troubleshooting topics:

- Backend must be running for dashboard, registry, and verification data to load.
- If image preview is unavailable, the watermarked image file may be missing from storage.
- If verification fails, confirm the selected artwork and uploaded image are correct.
- If the UI does not update after file changes, refresh the browser and restart the dev server if needed.

## 13. Non-Functional Expectations

Usability:

- Primary workflows should be discoverable from the sidebar.
- Forms should clearly mark required fields.
- Success and error states should be visible.
- Status language should be consistent.
- Long user-provided text should not break layouts.

Performance:

- Typical image processing should complete in a few seconds for normal PNG/JPEG files.
- Frontend should show loading states during longer operations.

Reliability:

- Records should persist in SQLite.
- Generated watermarked files should remain retrievable as long as storage files are present.
- Verification records should persist after browser refresh.

Responsiveness:

- Pages should work on desktop and mobile widths.
- Cards and detail panels should not overflow the viewport.
- Buttons should remain tappable on mobile.

Security and privacy limitations:

- The prototype has no authentication.
- Uploaded image files are stored locally.
- The app is intended for local/capstone use unless additional production hardening is added.

## 14. Known Intended Behavior for Recently Fixed UI Areas

The following describe the correct expected behavior for tests and documentation:

- Dashboard data should load only when the backend is reachable. If the backend is unreachable, the UI should show a clear "Failed to load dashboard data" style error.
- Artwork detail embedded watermark preview should display the watermarked image when the stored file exists. The preview should use the direct watermarked image endpoint and may use base64 fallback.
- If the database path is stale but the watermarked filename exists in the current `storage/watermarked` directory, the preview/download should still work.
- Verification history result pills should never stick out of cards.
- Long suspected filenames in verification history should be contained. On desktop, the filename area may scroll horizontally. On mobile, it may wrap.

## 15. Suggested Acceptance Criteria

The system is acceptable when:

- A user can complete the full register-download-verify-history-report flow without manual database editing.
- Uploaded images are validated before processing.
- Watermarked images are generated and previewable.
- Verification results are classified according to the BER rules.
- Verification records are visible in history and downloadable as CSV reports.
- Dashboard counts update after registration and verification.
- UI remains readable and contained with long filenames and long text values.
- Missing data or backend errors show understandable messages rather than blank pages or crashes.
