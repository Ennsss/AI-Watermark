import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Plus,
  Eye,
  CheckCircle,
  AlertCircle,
  Sparkles,
  ArrowUpRight,
  CalendarDays,
  User,
  Trash2,
} from 'lucide-react/dist/cjs/lucide-react';
import axios from 'axios';
import SearchInput from '../components/SearchInput';
import ArtworkPreviewFallback from '../components/ArtworkPreviewFallback';
import Toast from '../components/Toast';
import '../styles/ArtworksPage.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function ArtworksPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const [artworks, setArtworks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [query, setQuery] = useState('');
  const [artworkToUnregister, setArtworkToUnregister] = useState(null);
  const [unregistering, setUnregistering] = useState(false);
  const [successMessage, setSuccessMessage] = useState(location.state?.message || '');
  const [brokenPreviews, setBrokenPreviews] = useState({});
  const [exitingArtworkId, setExitingArtworkId] = useState(null);

  useEffect(() => {
    fetchArtworks();
  }, []);

  const fetchArtworks = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/api/artworks`);
      setArtworks(response.data.data);
      setError(null);
    } catch (err) {
      setError('Failed to load artworks');
      // eslint-disable-next-line no-console
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleVerify = (artworkId) => {
    navigate(`/verify?artwork_id=${artworkId}`);
  };

  const handleUnregister = async () => {
    if (!artworkToUnregister) return;
    try {
      setUnregistering(true);
      await axios.patch(`${API_BASE_URL}/api/artworks/${artworkToUnregister.artwork_id}/archive`);
      const archivedId = artworkToUnregister.artwork_id;
      setExitingArtworkId(archivedId);
      setArtworkToUnregister(null);
      window.setTimeout(() => {
        setArtworks((current) => current.filter((art) => art.artwork_id !== archivedId));
        setExitingArtworkId(null);
        setSuccessMessage('Artwork unregistered successfully.');
      }, 280);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(detail === 'Not Found'
        ? 'The archive API is unavailable. Restart the backend server and try again.'
        : detail || 'Failed to unregister artwork');
      setArtworkToUnregister(null);
    } finally {
      setUnregistering(false);
    }
  };

  const normalizedQuery = query.trim().toLowerCase();
  const filteredArtworks = artworks.filter((art) => [art.artwork_id, art.title, art.creator_name, art.watermark_status]
    .some((value) => String(value || '').toLowerCase().includes(normalizedQuery)));

  if (loading) return <div className="loading">Loading artworks...</div>;

  return (
    <div className="artworks-page">
      <div className="artworks-header">
        <div>
          <p className="page-kicker">
            <Sparkles size={14} /> Registry
          </p>
          <h1>My Artworks</h1>
          <p className="page-subtitle">
            Browse your registered artworks, inspect each record, or jump straight into verification.
          </p>
        </div>
        <button
          className="btn btn-primary"
          onClick={() => navigate('/register')}
        >
          <Plus size={18} /> Register New Artwork
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {artworks.length === 0 ? (
        <div className="empty-state">
          <p>No artworks registered yet.</p>
          <button
            className="btn btn-primary"
            onClick={() => navigate('/register')}
          >
            Register Your First Artwork
          </button>
        </div>
      ) : (
        <>
        <SearchInput value={query} onChange={setQuery} label="Search artworks" placeholder="Search by artwork ID, title, or artist" />
        {filteredArtworks.length === 0 ? (
          <div className="empty-state">No artworks match your search.</div>
        ) : (
        <div className="artworks-grid">
          {filteredArtworks.map((art) => {
            const isWatermarked = art.watermark_status === 'embedded';
            const previewSrc = art.watermarked_image_base64
              ? `data:image/png;base64,${art.watermarked_image_base64}`
              : art.watermarked_download_url
                ? `${API_BASE_URL}${art.watermarked_download_url}`
                : null;

            return (
              <article key={art.artwork_id} className={`artwork-card${exitingArtworkId === art.artwork_id ? ' is-removing' : ''}`}>
                <div className="artwork-preview">
                  {previewSrc && !brokenPreviews[art.artwork_id] ? (
                    <img
                      src={previewSrc}
                      alt={`${art.title} preview`}
                      className="artwork-preview-image"
                      loading="lazy"
                      onError={() => setBrokenPreviews((current) => ({ ...current, [art.artwork_id]: true }))}
                    />
                  ) : (
                    <ArtworkPreviewFallback compact />
                  )}
                  <span className="artwork-preview-id">{art.artwork_id}</span>
                </div>

                <div className="artwork-card-top">
                  <div>
                    <p className="card-kicker">{art.artwork_id}</p>
                    <h2 className="artwork-card-title" title={art.title}>{art.title}</h2>
                  </div>
                  <div className={`status-badge ${isWatermarked ? 'success' : 'warning'}`}>
                    {isWatermarked ? (
                      <>
                        <CheckCircle size={14} /> Watermarked
                      </>
                    ) : (
                      <>
                        <AlertCircle size={14} /> {art.watermark_status}
                      </>
                    )}
                  </div>
                </div>

                <div className="artwork-meta-grid">
                  <div className="meta-chip">
                    <User size={14} />
                    <div>
                      <span className="meta-label">Creator</span>
                      <span className="meta-value">{art.creator_name}</span>
                    </div>
                  </div>

                  <div className="meta-chip">
                    <CalendarDays size={14} />
                    <div>
                      <span className="meta-label">Registered</span>
                      <span className="meta-value">
                        {new Date(art.registration_date).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="artwork-card-footer">
                  <button
                    className="btn btn-small btn-outline"
                    onClick={() => navigate(`/artworks/${art.artwork_id}`)}
                    title="View details"
                  >
                    <Eye size={16} />
                    View details
                  </button>
                  <button
                    className="btn btn-small btn-primary"
                    onClick={() => handleVerify(art.artwork_id)}
                    title="Verify image"
                  >
                    Verify <ArrowUpRight size={15} />
                  </button>
                  <button
                    className="btn btn-small artwork-trash-button"
                    onClick={() => setArtworkToUnregister(art)}
                    title={`Unregister ${art.title}`}
                    aria-label={`Unregister ${art.title}`}
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </article>
            );
          })}
        </div>
        )}
        </>
      )}
      {artworkToUnregister && (
        <div className="modal-overlay" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !unregistering) setArtworkToUnregister(null); }}>
          <div className="unregister-modal" role="dialog" aria-modal="true" aria-labelledby="card-unregister-title">
            <h2 id="card-unregister-title">Unregister artwork?</h2>
            <p><strong>{artworkToUnregister.title}</strong> will be removed from your active registry and can no longer be selected for new verification attempts. Existing verification history, technical records, and image files will be preserved. This action cannot be restored from the application.</p>
            <div className="modal-actions">
              <button className="btn btn-outline" onClick={() => setArtworkToUnregister(null)} disabled={unregistering}>Cancel</button>
              <button className="btn btn-danger" onClick={handleUnregister} disabled={unregistering}>{unregistering ? 'Unregistering...' : 'Unregister Artwork'}</button>
            </div>
          </div>
        </div>
      )}
      <Toast message={successMessage} onDismiss={() => setSuccessMessage('')} />
    </div>
  );
}
