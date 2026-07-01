import React from 'react';
import {
  Upload,
  Download,
  Zap,
  Info,
  AlertCircle,
} from 'lucide-react/dist/cjs/lucide-react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function App() {
  const fileInputRef = React.useRef(null);
  const [activeTab, setActiveTab] = React.useState('embed');
  const [imageFile, setImageFile] = React.useState(null);
  const [imagePreview, setImagePreview] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [result, setResult] = React.useState(null);
  const [error, setError] = React.useState(null);
  const [delta, setDelta] = React.useState(16);
  const [payload, setPayload] = React.useState('');

  const handleImageSelect = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file');
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setError('File size exceeds 50MB limit');
      return;
    }

    setImageFile(file);
    setError(null);
    setResult(null);

    // Create preview
    const reader = new FileReader();
    reader.onload = (event) => {
      setImagePreview(event.target?.result);
    };
    reader.readAsDataURL(file);
  };

  const handleEmbed = async () => {
    if (!imageFile) {
      setError('Please select an image');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      formData.append('delta', delta.toString());
      if (payload) {
        formData.append('payload', payload);
      }

      const response = await axios.post(
        `${API_BASE_URL}/api/embed`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 30000,
        }
      );

      setResult({
        type: 'embed',
        data: response.data,
        imageBase64: response.data.image,
        requestedDelta: delta,
        requestedPayload: payload || 'a'.repeat(32),
      });
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        'Failed to embed watermark'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleExtract = async () => {
    if (!imageFile) {
      setError('Please select an image');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      formData.append('delta', delta.toString());

      const response = await axios.post(
        `${API_BASE_URL}/api/extract`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: 30000,
        }
      );

      setResult({
        type: 'extract',
        data: response.data,
      });
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        'Failed to extract watermark'
      );
    } finally {
      setLoading(false);
    }
  };

  const downloadImage = () => {
    if (!result?.imageBase64) return;

    const link = document.createElement('a');
    link.href = `data:image/png;base64,${result.imageBase64}`;
    link.download = `watermarked-${Date.now()}.png`;
    link.click();
  };

  const handleEmbedAnother = () => {
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>AI Watermark</h1>
        <p>DWT-QIM Watermarking for Digital Art</p>
      </header>

      <main className="container">
        {/* Tabs */}
        <div className="tabs">
          <button
            className={`tab ${activeTab === 'embed' ? 'active' : ''}`}
            onClick={() => setActiveTab('embed')}
          >
            <Zap size={18} />
            Embed
          </button>
          <button
            className={`tab ${activeTab === 'extract' ? 'active' : ''}`}
            onClick={() => setActiveTab('extract')}
          >
            <Download size={18} />
            Extract
          </button>
        </div>

        <div className="content">
          {/* Upload Section */}
          <section className="upload-section">
            <div className="upload-area">
              <input
                type="file"
                id="image-input"
                accept="image/*"
                onChange={handleImageSelect}
                disabled={loading}
                className="file-input"
                ref={fileInputRef}
              />
              <label htmlFor="image-input" className="upload-label">
                <Upload size={32} />
                <span>Tap to select an image</span>
                <small>or drag and drop (JPEG, PNG)</small>
              </label>
            </div>

            {imagePreview && (
              <div className="preview-section">
                <h3>Preview</h3>
                <img src={imagePreview} alt="Preview" className="preview-image" />
              </div>
            )}
          </section>

          {/* Controls Section */}
          <section className="controls-section">
            <div className="control-group">
              <label htmlFor="delta">
                Quantization Step (Delta): {delta}
              </label>
              <input
                id="delta"
                type="range"
                min="4"
                max="64"
                step="4"
                value={delta}
                onChange={(e) => setDelta(Number(e.target.value))}
                disabled={loading}
              />
              <small>Higher = more robust, lower = less visible</small>
            </div>

            {activeTab === 'embed' && (
              <div className="control-group">
                <label htmlFor="payload">
                  Payload (Optional, Hex)
                </label>
                <input
                  id="payload"
                  type="text"
                  placeholder="Leave empty for default 128-bit payload"
                  value={payload}
                  onChange={(e) => setPayload(e.target.value)}
                  disabled={loading}
                  className="payload-input"
                />
                <small>Must be 32 hex characters (128 bits)</small>
              </div>
            )}
          </section>

          {/* Error Display */}
          {error && (
            <div className="alert alert-error">
              <AlertCircle size={20} />
              <span>{error}</span>
            </div>
          )}

          {/* Action Button */}
          <button
            className="action-button"
            onClick={activeTab === 'embed' ? handleEmbed : handleExtract}
            disabled={!imageFile || loading}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                Processing...
              </>
            ) : activeTab === 'embed' ? (
              <>
                <Zap size={20} />
                Embed
              </>
            ) : (
              <>
                <Download size={20} />
                Extract
              </>
            )}
          </button>

          {/* Results Section */}
          {result && (
            <section className="results-section">
              <h3>
                {result.type === 'embed' ? '✅ Watermark Embedded' : '✅ Watermark Extracted'}
              </h3>

              {result.type === 'embed' && (
                <>
                  <div className="result-image">
                    <img src={`data:image/png;base64,${result.imageBase64}`} alt="Watermarked" />
                  </div>
                  <div className="result-actions">
                    <button className="download-button" onClick={downloadImage}>
                      <Download size={18} />
                      Download Image
                    </button>
                    <button className="download-button secondary-button" onClick={handleEmbedAnother}>
                      <Upload size={18} />
                      Embed Another Image
                    </button>
                  </div>
                </>
              )}

              {result.type === 'extract' && (
                <>
                  <div className="extraction-results">
                    <div className="result-item">
                      <strong>Extracted Payload:</strong>
                      <code>{result.data.extracted_payload}</code>
                    </div>
                    <div className="result-item">
                      <strong>Bit Error Rate (BER):</strong>
                      <span>{(result.data.bit_error_rate * 100).toFixed(2)}%</span>
                    </div>
                    <div className="result-item">
                      <strong>Confidence:</strong>
                      <span>{(result.data.confidence * 100).toFixed(2)}%</span>
                    </div>
                  </div>
                </>
              )}

              <div className="result-params">
                <h4>
                  <Info size={16} /> Parameters Used
                </h4>
                <ul>
                  {result.requestedDelta !== undefined && (
                    <li>Requested Delta (UI): {result.requestedDelta}</li>
                  )}
                  <li>Wavelet: {result.data.parameters.wavelet}</li>
                  <li>DWT Level: {result.data.parameters.dwt_level}</li>
                  <li>Delta: {result.data.parameters.delta}</li>
                  {result.data.parameters.payload_hex && (
                    <li>Payload: {result.data.parameters.payload_hex}</li>
                  )}
                </ul>
              </div>
            </section>
          )}
        </div>
      </main>

      <footer className="footer">
        <p>DWT-QIM Watermarking | Research Implementation</p>
      </footer>
    </div>
  );
}
