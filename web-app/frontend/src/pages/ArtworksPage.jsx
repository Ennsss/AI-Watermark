import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Plus,
  Eye,
  CheckCircle,
  AlertCircle,
  Sparkles,
  ArrowUpRight,
  CalendarDays,
  User,
} from 'lucide-react/dist/cjs/lucide-react';
import axios from 'axios';
import '../styles/ArtworksPage.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function ArtworksPage() {
  const [artworks, setArtworks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

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
        <div className="artworks-grid">
          {artworks.map((art) => {
            const isWatermarked = art.watermark_status === 'embedded';

            return (
              <article key={art.artwork_id} className="artwork-card">
                <div className="artwork-card-top">
                  <div>
                    <p className="card-kicker">{art.artwork_id}</p>
                    <h2 className="artwork-card-title">{art.title}</h2>
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
                </div>
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
}
