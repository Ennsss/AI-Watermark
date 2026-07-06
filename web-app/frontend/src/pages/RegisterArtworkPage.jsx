import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Download, Check } from 'lucide-react/dist/cjs/lucide-react';
import axios from 'axios';
import '../styles/RegisterArtworkPage.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function RegisterArtworkPage() {
  const [formData, setFormData] = useState({
    title: '',
    creator_name: '',
    notes: '',
  });
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [result, setResult] = useState(null);
  const navigate = useNavigate();

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

    // Create preview
    const reader = new FileReader();
    reader.onload = (event) => {
      setPreview(event.target?.result);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!formData.title || !formData.creator_name || !file) {
      setError('Please fill in all required fields');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formDataToSend = new FormData();
      formDataToSend.append('title', formData.title);
      formDataToSend.append('creator_name', formData.creator_name);
      formDataToSend.append('notes', formData.notes);
      formDataToSend.append('file', file);

      const response = await axios.post(
        `${API_BASE_URL}/api/artworks/register`,
        formDataToSend,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 60000,
        }
      );

      setResult(response.data);
      setSuccess(`Artwork registered successfully! ID: ${response.data.artwork_id}`);
      setFormData({ title: '', creator_name: '', notes: '' });
      setFile(null);
      setPreview(null);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to register artwork');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!result?.image) return;
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${result.image}`;
    link.download = `${result.artwork_id}_watermarked.png`;
    link.click();
  };

  const handleViewArtwork = () => {
    navigate(`/artworks/${result.artwork_id}`);
  };

  if (result) {
    return (
      <div className="register-page">
        <div className="success-container">
          <div className="success-badge">
            <Check size={48} />
          </div>
          <h2>Artwork Registered Successfully!</h2>

          <div className="result-details">
            <div className="detail-row">
              <span className="label">Artwork ID:</span>
              <span className="value">{result.artwork_id}</span>
            </div>
            <div className="detail-row">
              <span className="label">Title:</span>
              <span className="value">{result.title}</span>
            </div>
            <div className="detail-row">
              <span className="label">Creator:</span>
              <span className="value">{result.creator_name}</span>
            </div>
            <div className="detail-row">
              <span className="label">Status:</span>
              <span className="value watermarked">✓ Watermark Embedded</span>
            </div>
          </div>

          <div className="preview-section">
            <h3>Watermarked Image</h3>
            {result.image && (
              <img
                src={`data:image/png;base64,${result.image}`}
                alt="Watermarked artwork"
                className="watermarked-preview"
              />
            )}
          </div>

          <div className="action-buttons">
            <button className="btn btn-primary" onClick={handleDownload}>
              <Download size={18} /> Download Watermarked Image
            </button>
            <button className="btn btn-outline" onClick={handleViewArtwork}>
              View Artwork Record
            </button>
            <button
              className="btn btn-outline"
              onClick={() => {
                setResult(null);
                setSuccess(null);
              }}
            >
              Register Another Artwork
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="register-page">
      <h1>Register Artwork</h1>

      <form onSubmit={handleSubmit} className="register-form">
        <div className="form-group">
          <label htmlFor="title">Artwork Title *</label>
          <input
            id="title"
            type="text"
            name="title"
            placeholder="Enter artwork title"
            value={formData.title}
            onChange={handleInputChange}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="creator_name">Creator Name *</label>
          <input
            id="creator_name"
            type="text"
            name="creator_name"
            placeholder="Enter creator name"
            value={formData.creator_name}
            onChange={handleInputChange}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="notes">Notes (optional)</label>
          <textarea
            id="notes"
            name="notes"
            placeholder="Add any additional notes about this artwork"
            value={formData.notes}
            onChange={handleInputChange}
            rows="3"
          />
        </div>

        <div className="form-group">
          <label>Upload Artwork *</label>
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
              <p>Drag and drop your image here, or click to select</p>
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
        {success && <div className="success-message">{success}</div>}

        <button
          type="submit"
          className="btn btn-primary btn-large"
          disabled={loading}
        >
          {loading ? 'Registering & Embedding...' : 'Register & Embed Watermark'}
        </button>
      </form>
    </div>
  );
}
