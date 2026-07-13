import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, Download, Check } from 'lucide-react/dist/cjs/lucide-react';
import axios from 'axios';
import '../styles/RegisterArtworkPage.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const TITLE_MAX_LENGTH = 120;
const CREATOR_MAX_LENGTH = 80;
const NOTES_MAX_LENGTH = 1000;
const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png'];

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

    if (!ALLOWED_IMAGE_TYPES.includes(selectedFile.type)) {
      setError('Only JPEG and PNG files are allowed');
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

    const title = formData.title.trim();
    const creatorName = formData.creator_name.trim();
    if (!title || !creatorName || !file) {
      setError('Please fill in all required fields');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formDataToSend = new FormData();
      formDataToSend.append('title', title);
      formDataToSend.append('creator_name', creatorName);
      formDataToSend.append('notes', formData.notes.trim());
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
          <h2>Artwork Registered Successfully</h2>

          <div className="result-details">
            <div className="detail-row">
              <span className="label marginLeft">Artwork ID:</span>
              <span className="value marginRight">{result.artwork_id}</span>
            </div>
            <div className="detail-row">
              <span className="label marginLeft">Title:</span>
              <span className="value marginRight scrollable-result-value" title={result.title}>{result.title}</span>
            </div>
            <div className="detail-row">
              <span className="label marginLeft">Creator:</span>
              <span className="value marginRight scrollable-result-value" title={result.creator_name}>{result.creator_name}</span>
            </div>
            <div className="detail-row">
              <span className="label marginLeft">Status:</span>
              <span className="value marginRight watermarked"><Check size={16} /> Watermark Embedded</span>
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
        <div className="form-group margin">
          <label htmlFor="title">Artwork Title *</label>
          <input
            id="title"
            type="text"
            name="title"
            placeholder="Enter artwork title"
            value={formData.title}
            onChange={handleInputChange}
            maxLength={TITLE_MAX_LENGTH}
            required
          />
          <span className="character-count">{formData.title.length}/{TITLE_MAX_LENGTH}</span>
        </div>

        <div className="form-group margin">
          <label htmlFor="creator_name">Creator Name *</label>
          <input
            id="creator_name"
            type="text"
            name="creator_name"
            placeholder="Enter creator name"
            value={formData.creator_name}
            onChange={handleInputChange}
            maxLength={CREATOR_MAX_LENGTH}
            required
          />
          <span className="character-count">{formData.creator_name.length}/{CREATOR_MAX_LENGTH}</span>
        </div>

        <div className="form-group margin">
          <label htmlFor="notes">Notes (optional)</label>
          <textarea
            id="notes"
            name="notes"
            placeholder="Add any additional notes about this artwork"
            value={formData.notes}
            onChange={handleInputChange}
            maxLength={NOTES_MAX_LENGTH}
            rows="3"
          />
          <span className="character-count">{formData.notes.length}/{NOTES_MAX_LENGTH}</span>
        </div>

        <div className="form-group margin">
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
              accept="image/jpeg,image/png"
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
