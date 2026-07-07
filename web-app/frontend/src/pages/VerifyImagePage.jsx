import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Upload, Check, AlertCircle, X, Download } from 'lucide-react/dist/cjs/lucide-react';
import axios from 'axios';
import '../styles/VerifyImagePage.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const ResultCard = ({ verification }) => {
  const getStatusStyles = () => {
    switch (verification.result_status) {
      case 'match':
        return { icon: Check, color: 'success', title: 'Verified Match' };
      case 'partial':
        return { icon: AlertCircle, color: 'warning', title: 'Partial / Corrupted Watermark Detected' };
      default:
        return { icon: X, color: 'error', title: 'No Valid Watermark Detected' };
    }
  };

  const styles = getStatusStyles();
  const IconComponent = styles.icon;

  return (
    <div className={`result-card ${styles.color}`}>
      <div className="result-icon">
        <IconComponent size={48} />
      </div>
      <h2>{styles.title}</h2>
      <div className="result-details">
        <div className="detail-row">
          <span className="label">Verification ID:</span>
          <span className="value">{verification.verification_id}</span>
        </div>
        {verification.artwork_id && (
          <div className="detail-row">
            <span className="label">Against Artwork:</span>
            <span className="value">{verification.artwork_id}</span>
          </div>
        )}
        {verification.ber !== null && (
          <div className="detail-row">
            <span className="label">BER (Bit Error Rate):</span>
            <span className="value">{verification.ber.toFixed(4)}</span>
          </div>
        )}
        <div className="detail-row">
          <span className="label">Processing Time:</span>
          <span className="value">{verification.processing_time_ms.toFixed(2)}ms</span>
        </div>
      </div>
      {verification.message && (
        <div className="result-message">{verification.message}</div>
      )}
      <button
        className="btn btn-primary"
        onClick={() => {
          const link = document.createElement('a');
          link.href = `${API_BASE_URL}/api/verifications/${verification.verification_id}/report.csv`;
          link.download = `verification_${verification.verification_id}.csv`;
          link.click();
        }}
      >
        <Download size={16} /> Download Technical Report
      </button>
    </div>
  );
};

export default function VerifyImagePage() {
  const [searchParams] = useSearchParams();
  const [artworks, setArtworks] = useState([]);
  const [selectedArtworkId, setSelectedArtworkId] = useState(searchParams.get('artwork_id') || '');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [verification, setVerification] = useState(null);
  const [loadingArtworks, setLoadingArtworks] = useState(true);

  useEffect(() => {
    fetchArtworks();
  }, []);

  const fetchArtworks = async () => {
    try {
      setLoadingArtworks(true);
      const response = await axios.get(`${API_BASE_URL}/api/artworks`);
      setArtworks(response.data.data);
    } catch (err) {
      setError('Failed to load artworks');
      // eslint-disable-next-line no-console
      console.error(err);
    } finally {
      setLoadingArtworks(false);
    }
  };

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith('image/')) {
      setError('Please select a valid image file');
      return;
    }

    if (selectedFile.size > 50 * 1024 * 1024) {
      setError('File size exceeds 50MB limit');
      return;
    }

    setFile(selectedFile);
    setError(null);

    const reader = new FileReader();
    reader.onload = (event) => {
      setPreview(event.target?.result);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleVerify = async (e) => {
    e.preventDefault();

    if (!selectedArtworkId || !file) {
      setError('Please select an artwork and image file');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('artwork_id', selectedArtworkId);
      formData.append('file', file);

      const response = await axios.post(
        `${API_BASE_URL}/api/verifications`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 60000,
        }
      );

      setVerification(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Verification failed');
    } finally {
      setLoading(false);
    }
  };

  if (loadingArtworks) return <div className="loading">Loading artworks...</div>;

  if (verification) {
    return (
      <div className="verify-page">
        <ResultCard verification={verification} />
        <button
          className="btn btn-outline"
          onClick={() => {
            setVerification(null);
            setFile(null);
            setPreview(null);
            setSelectedArtworkId('');
          }}
        >
          Verify Another Image
        </button>
      </div>
    );
  }

  return (
    <div className="verify-page">
      <h1>Verify Image</h1>

      <form onSubmit={handleVerify} className="verify-form">
        <div className="form-group margin">
          <label htmlFor="artwork-select">Verify Against * </label>
          <select
            id="artwork-select"
            value={selectedArtworkId}
            onChange={(e) => setSelectedArtworkId(e.target.value)}
            required
          >
            <option value="">Select an artwork from your registry...</option>
            {artworks.map((art) => (
              <option key={art.artwork_id} value={art.artwork_id}>
                {art.artwork_id} - {art.title}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group margin">
          <label>Suspected / Reposted Image *</label>
          <div
            className="file-upload"
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              const droppedFile = e.dataTransfer.files?.[0];
              if (droppedFile) {
                const event = { target: { files: [droppedFile] } };
                handleFileSelect(event);
              }
            }}
          >
            <input
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
              id="file-input"
            />
            <label htmlFor="file-input" className="file-label">
              <Upload size={32} />
              <p>Drag and drop image here, or click to select</p>
              <span className="file-hint">PNG, JPEG up to 50MB</span>
            </label>
          </div>
          {file && <div className="file-selected">Selected: {file.name}</div>}
        </div>

        {preview && (
          <div className="preview-section">
            <h3>Image Preview</h3>
            <img src={preview} alt="Preview" className="preview-image" />
          </div>
        )}

        {error && <div className="error-message">{error}</div>}

        <button
          type="submit"
          className="btn btn-primary btn-large"
          disabled={loading}
        >
          {loading ? 'Extracting & Verifying...' : 'Verify Watermark'}
        </button>
      </form>
    </div>
  );
}
