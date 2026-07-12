import React, { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  ArrowLeft,
  BadgeCheck,
  CheckCircle,
  Clock3,
  FileText,
  ShieldCheck,
  User,
  CalendarDays,
  Sparkles,
  ArrowUpRight,
  Download,
  Trash2,
  Pencil,
} from 'lucide-react/dist/cjs/lucide-react';
import axios from 'axios';
import ArtworkPreviewFallback from '../components/ArtworkPreviewFallback';
import Toast from '../components/Toast';
import '../styles/ArtworkDetailPage.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function ArtworkDetailPage() {
  const { artworkId } = useParams();
  const navigate = useNavigate();
  const [artwork, setArtwork] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showUnregister, setShowUnregister] = useState(false);
  const [unregistering, setUnregistering] = useState(false);
  const [previewBroken, setPreviewBroken] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [editing, setEditing] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [editForm, setEditForm] = useState({
    title: '',
    creator_name: '',
    notes: '',
  });

  useEffect(() => {
    const loadArtwork = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${API_BASE_URL}/api/artworks/${artworkId}`);
        setArtwork(response.data.data);
        setError(null);
      } catch (err) {
        setError('Artwork not found or unavailable');
        // eslint-disable-next-line no-console
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    if (artworkId) {
      loadArtwork();
    }
  }, [artworkId]);

  if (loading) {
    return <div className="loading">Loading artwork details...</div>;
  }

  if (error || !artwork) {
    return (
      <div className="artwork-detail-page">
        <button className="back-button" onClick={() => navigate('/artworks')}>
          <ArrowLeft size={18} /> Back to artworks
        </button>
        <div className="detail-empty-state">
          <BadgeCheck size={36} />
          <h1>Artwork details unavailable</h1>
          <p>{error || 'No artwork was found for this record.'}</p>
        </div>
      </div>
    );
  }

  const statusLabel = artwork.watermark_status === 'embedded' ? 'Watermarked' : artwork.watermark_status;
  const watermarkedPreviewSrc = artwork.watermarked_download_url
    ? `${API_BASE_URL}${artwork.watermarked_download_url}?preview=${Date.now()}`
    : artwork.watermarked_image_base64
      ? `data:image/png;base64,${artwork.watermarked_image_base64}`
      : null;

  const handleUnregister = async () => {
    try {
      setUnregistering(true);
      await axios.patch(`${API_BASE_URL}/api/artworks/${artwork.artwork_id}/archive`);
      navigate('/artworks', { state: { message: 'Artwork unregistered successfully.' } });
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to unregister artwork');
      setShowUnregister(false);
    } finally {
      setUnregistering(false);
    }
  };

  const openEditModal = () => {
    setEditForm({
      title: artwork.title || '',
      creator_name: artwork.creator_name || '',
      notes: artwork.notes || '',
    });
    setShowEditModal(true);
  };

  const handleSaveEdit = async () => {
    try {
      setEditing(true);
      const response = await axios.patch(`${API_BASE_URL}/api/artworks/${artwork.artwork_id}`, editForm);
      setArtwork(response.data.data);
      setShowEditModal(false);
      setSuccessMessage('Artwork details updated successfully.');
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to update artwork details');
    } finally {
      setEditing(false);
    }
  };

  return (
    <div className="artwork-detail-page">
      <button className="back-button" onClick={() => navigate('/artworks')}>
        <ArrowLeft size={18} /> Back to artworks
      </button>

      <section className="detail-hero card">
        <div className="detail-hero-copy">
          <p className="page-kicker">
            <Sparkles size={14} /> Artwork record
          </p>
          <h1 className="scrollable-detail-title" title={artwork.title}>{artwork.title}</h1>
          <p className="detail-hero-subtitle">
            A clean provenance snapshot for this registered artwork, including registry data and current watermark state.
          </p>
        </div>
        <div className="detail-hero-aside">
          <div className="hero-stat">
            <span className="hero-stat-label">Artwork ID</span>
            <span className="hero-stat-value">{artwork.artwork_id}</span>
          </div>
          <div className="hero-stat status">
            <span className="hero-stat-label">Status</span>
            <span className="hero-stat-value status-pill">
              <ShieldCheck size={16} /> {statusLabel}
            </span>
          </div>
        </div>
      </section>

      <section className="card artwork-preview-card">
        <div className="preview-header">
          <h2>
            <ShieldCheck size={20} /> Embedded Watermark Preview
          </h2>
          {artwork.watermarked_download_url && (
            <a
              className="btn btn-outline btn-small"
              href={`${API_BASE_URL}${artwork.watermarked_download_url}`}
              download
            >
              <Download size={16} /> Download Image
            </a>
          )}
        </div>

        {watermarkedPreviewSrc && !previewBroken ? (
          <div className="preview-stage">
            <img
              className="preview-image-large"
              src={watermarkedPreviewSrc}
              alt={`${artwork.title} watermarked preview`}
              onError={(event) => {
                if (artwork.watermarked_image_base64 && !event.currentTarget.src.startsWith('data:')) {
                  event.currentTarget.src = `data:image/png;base64,${artwork.watermarked_image_base64}`;
                } else {
                  setPreviewBroken(true);
                }
              }}
            />
          </div>
        ) : (
          <ArtworkPreviewFallback />
        )}
      </section>

      <details className="card technical-artwork-details">
        <summary>Technical Details</summary>
        <div className="detail-list">
          <div className="detail-row"><span className="detail-label">Artwork ID</span><span className="detail-value">{artwork.artwork_id}</span></div>
          <div className="detail-row"><span className="detail-label">Watermark status</span><span className="detail-value">{statusLabel}</span></div>
          <div className="detail-row"><span className="detail-label">Payload length</span><span className="detail-value">{artwork.payload_length} bits</span></div>
          <div className="detail-row"><span className="detail-label">Payload preview</span><code className="payload-preview">{artwork.payload_preview}</code></div>
          <div className="detail-row"><span className="detail-label">Watermark engine</span><span className="detail-value">DWT-QIM</span></div>
          <div className="detail-row"><span className="detail-label">Registered</span><span className="detail-value">{new Date(artwork.registration_date).toLocaleString()}</span></div>
        </div>
      </details>

      <div className="detail-grid">
        <section className="card detail-panel">
          <h2>
            <FileText size={20} /> Provenance Record
          </h2>

          <div className="detail-list">
            <div className="detail-row">
              <span className="detail-label"><User size={14} /> Creator</span>
              <span className="detail-value scrollable-detail-value" title={artwork.creator_name}>{artwork.creator_name}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label"><Clock3 size={14} /> Registered</span>
              <span className="detail-value">
                {new Date(artwork.registration_date).toLocaleString()}
              </span>
            </div>
            <div className="detail-row">
              <span className="detail-label"><CheckCircle size={14} /> Watermark</span>
              <span className="detail-value">{statusLabel}</span>
            </div>
          </div>
        </section>

        <section className="card detail-panel">
          <h2>
            <CalendarDays size={20} /> Notes
          </h2>
          <div className="notes-box">
            {artwork.notes ? artwork.notes : 'No notes were added for this artwork.'}
          </div>
          <div className="detail-actions">
            <button className="btn btn-outline" onClick={openEditModal}>
              <Pencil size={16} /> Edit Artwork
            </button>
            <button className="btn btn-primary" onClick={() => navigate(`/verify?artwork_id=${artwork.artwork_id}`)}>
              Verify This Artwork <ArrowUpRight size={16} />
            </button>
            {artwork.watermarked_download_url && (
              <a
                className="btn btn-outline"
                href={`${API_BASE_URL}${artwork.watermarked_download_url}`}
                download
              >
                <Download size={16} />
                Download Watermarked Image
              </a>
            )}
            <button className="btn btn-outline" onClick={() => navigate('/artworks')}>
              Back to Registry
            </button>
            <button className="btn btn-danger" onClick={() => setShowUnregister(true)}>
              <Trash2 size={16} /> Unregister Artwork
            </button>
          </div>
        </section>
      </div>
      {showUnregister && (
        <div className="modal-overlay" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setShowUnregister(false); }}>
          <div className="unregister-modal" role="dialog" aria-modal="true" aria-labelledby="unregister-title">
            <h2 id="unregister-title">Unregister artwork?</h2>
            <p>This artwork will be removed from your active registry and can no longer be selected for new verification attempts. Existing verification history and technical records will be preserved. This action cannot be restored from the application.</p>
            <div className="modal-actions">
              <button className="btn btn-outline" onClick={() => setShowUnregister(false)} disabled={unregistering}>Cancel</button>
              <button className="btn btn-danger" onClick={handleUnregister} disabled={unregistering}>{unregistering ? 'Unregistering…' : 'Unregister Artwork'}</button>
            </div>
          </div>
        </div>
      )}
      {showEditModal && (
        <div className="modal-overlay" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !editing) setShowEditModal(false); }}>
          <div className="unregister-modal edit-modal" role="dialog" aria-modal="true" aria-labelledby="edit-artwork-title">
            <h2 id="edit-artwork-title">Edit artwork details</h2>
            <div className="edit-form-grid">
              <label className="edit-field" htmlFor="edit-title">
                <span>Title</span>
                <input
                  id="edit-title"
                  type="text"
                  value={editForm.title}
                  onChange={(event) => setEditForm((current) => ({ ...current, title: event.target.value }))}
                  maxLength={120}
                />
              </label>
              <label className="edit-field" htmlFor="edit-creator">
                <span>Creator</span>
                <input
                  id="edit-creator"
                  type="text"
                  value={editForm.creator_name}
                  onChange={(event) => setEditForm((current) => ({ ...current, creator_name: event.target.value }))}
                  maxLength={80}
                />
              </label>
              <label className="edit-field" htmlFor="edit-notes">
                <span>Notes</span>
                <textarea
                  id="edit-notes"
                  value={editForm.notes}
                  onChange={(event) => setEditForm((current) => ({ ...current, notes: event.target.value }))}
                  maxLength={1000}
                  rows={4}
                />
              </label>
            </div>
            <div className="modal-actions">
              <button className="btn btn-outline" onClick={() => setShowEditModal(false)} disabled={editing}>Cancel</button>
              <button className="btn btn-primary" onClick={handleSaveEdit} disabled={editing}>{editing ? 'Saving...' : 'Save Changes'}</button>
            </div>
          </div>
        </div>
      )}
      <Toast message={successMessage} onDismiss={() => setSuccessMessage('')} />
    </div>
  );
}
