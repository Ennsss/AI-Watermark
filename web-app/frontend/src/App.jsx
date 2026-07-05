import React from 'react';
import {
  Upload,
  Download,
  Zap,
  Info,
  AlertCircle,
  Search,
} from 'lucide-react/dist/cjs/lucide-react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function App() {
  const fileInputRef = React.useRef(null);
  const [activeTab, setActiveTab] = React.useState('embed');
  const [imageFiles, setImageFiles] = React.useState([]);
  const [imagePreviews, setImagePreviews] = React.useState([]);
  const [loading, setLoading] = React.useState(false);
  const [result, setResult] = React.useState(null);
  const [error, setError] = React.useState(null);
  const [payload, setPayload] = React.useState('');
  const [showConfirmDialog, setShowConfirmDialog] = React.useState(false);
  const [pendingTab, setPendingTab] = React.useState(null);

  const handleImageSelect = (e) => {
    const files = e.target.files ? Array.from(e.target.files) : [];
    if (files.length === 0) return;

    // Validate all files
    const validFiles = [...imageFiles]; // Start with existing files
    const previews = [...imagePreviews]; // Start with existing previews

    for (const file of files) {
      if (!file.type.startsWith('image/')) {
        setError(`${file.name} is not a valid image file`);
        return;
      }

      if (file.size > 50 * 1024 * 1024) {
        setError(`${file.name} exceeds 50MB limit`);
        return;
      }

      validFiles.push(file);

      // Create preview
      const reader = new FileReader();
      reader.onload = (event) => {
        previews.push({
          name: file.name,
          data: event.target?.result,
        });
        if (previews.length === validFiles.length) {
          setImagePreviews(previews);
        }
      };
      reader.readAsDataURL(file);
    }

    setImageFiles(validFiles);
    setError(null);
    // Don't clear results when adding more images, only when initially selecting
    if (imageFiles.length === 0) {
      setResult(null);
    }

    // Reset file input so same file can be selected again
    if (e.target) {
      e.target.value = '';
    }
  };

  const handleEmbed = async () => {
    if (imageFiles.length === 0) {
      setError('Please select at least one image');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const results = [];
      for (const file of imageFiles) {
        const formData = new FormData();
        formData.append('file', file);
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

        results.push({
          fileName: file.name,
          ...response.data,
        });
      }

      setResult({
        type: 'embed',
        data: results,
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

  const handleRemove = async () => {
    if (imageFiles.length === 0) {
      setError('Please select at least one image');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const results = [];
      for (const file of imageFiles) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await axios.post(
          `${API_BASE_URL}/api/extract`,
          formData,
          {
            headers: { 'Content-Type': 'multipart/form-data' },
            timeout: 30000,
          }
        );

        results.push({
          fileName: file.name,
          ...response.data,
        });
      }

      setResult({
        type: 'extract',
        data: results,
      });
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        'Failed to remove watermark'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleDetect = async () => {
    if (imageFiles.length === 0) {
      setError('Please select at least one image');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const results = [];
      for (const file of imageFiles) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await axios.post(
          `${API_BASE_URL}/api/detect`,
          formData,
          {
            headers: { 'Content-Type': 'multipart/form-data' },
            timeout: 30000,
          }
        );

        results.push({
          fileName: file.name,
          ...response.data,
        });
      }

      setResult({
        type: 'detect',
        data: results,
      });
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        'Failed to detect watermark'
      );
    } finally {
      setLoading(false);
    }
  };

  const downloadEmbeddedImage = (imageBase64, fileName) => {
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${imageBase64}`;
    // Remove extension from fileName and add .png
    const nameWithoutExt = fileName.substring(0, fileName.lastIndexOf('.')) || fileName;
    link.download = `${nameWithoutExt}-watermarked.png`;
    link.click();
  };

  const downloadImage = () => {
    if (!result?.imageBase64) return;

    const link = document.createElement('a');
    link.href = `data:image/png;base64,${result.imageBase64}`;
    link.download = `watermarked-${Date.now()}.png`;
    link.click();
  };

  const handleRemoveImage = (indexToRemove) => {
    setImageFiles(imageFiles.filter((_, idx) => idx !== indexToRemove));
    setImagePreviews(imagePreviews.filter((_, idx) => idx !== indexToRemove));
  };

  const handleAddMoreImages = () => {
    fileInputRef.current?.click();
  };

  const handleEmbedAnother = () => {
    setImageFiles([]);
    setImagePreviews([]);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleTabChange = (newTab) => {
    // If there's a result and user is switching tabs, show confirmation
    if (result && newTab !== activeTab) {
      setPendingTab(newTab);
      setShowConfirmDialog(true);
    } else {
      // No result, just switch tabs
      setActiveTab(newTab);
    }
  };

  const handleConfirmTabChange = () => {
    if (pendingTab) {
      setActiveTab(pendingTab);
      handleEmbedAnother(); // Clear previous results
      setPendingTab(null);
    }
    setShowConfirmDialog(false);
  };

  const handleCancelTabChange = () => {
    setPendingTab(null);
    setShowConfirmDialog(false);
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
            onClick={() => handleTabChange('embed')}
          >
            <Zap size={18} />
            Embed
          </button>
          <button
            className={`tab ${activeTab === 'remove' ? 'active' : ''}`}
            onClick={() => handleTabChange('remove')}
          >
            <Download size={18} />
            Remove
          </button>
          <button
            className={`tab ${activeTab === 'detect' ? 'active' : ''}`}
            onClick={() => handleTabChange('detect')}
          >
            <Search size={18} />
            Detect
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
                multiple
                onChange={handleImageSelect}
                disabled={loading}
                className="file-input"
                ref={fileInputRef}
              />
              <label htmlFor="image-input" className="upload-label">
                <Upload size={32} />
                <span>Tap to select images</span>
                <small>or drag and drop (JPEG, PNG) - multiple files supported</small>
              </label>
            </div>

            {imagePreviews.length > 0 && (
              <div className="preview-section">
                <h3>Previews ({imagePreviews.length} images)</h3>
                <div className="preview-grid">
                  {imagePreviews.map((preview, idx) => (
                    <div key={idx} className="preview-item">
                      <button
                        className="preview-remove-btn"
                        onClick={() => handleRemoveImage(idx)}
                        title="Remove this image"
                        disabled={loading}
                      >
                        ✕
                      </button>
                      <img src={preview.data} alt={`Preview ${idx + 1}`} className="preview-image" />
                      <small>{preview.name}</small>
                    </div>
                  ))}
                  <button
                    className="preview-add-btn"
                    onClick={handleAddMoreImages}
                    disabled={loading}
                    title="Add more images"
                  >
                    <span className="plus-icon">+</span>
                    <small>Add More</small>
                  </button>
                </div>
              </div>
            )}
          </section>

          {/* Controls Section */}
          <section className="controls-section">
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
            onClick={
              activeTab === 'embed' ? handleEmbed :
              activeTab === 'remove' ? handleRemove :
              handleDetect
            }
            disabled={imageFiles.length === 0 || loading}
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
            ) : activeTab === 'remove' ? (
              <>
                <Download size={20} />
                Remove
              </>
            ) : (
              <>
                <Search size={20} />
                Detect
              </>
            )}
          </button>

          {/* Results Section */}
          {result && (
            <section className="results-section">
              <h3>
                {result.type === 'embed' ? '✅ Watermarks Embedded' : 
                 result.type === 'extract' ? '✅ Watermarks Removed' :
                 '✅ Detection Complete'}
              </h3>

              {result.type === 'embed' && (
                <>
                  <div className="results-grid">
                    {result.data.map((item, idx) => (
                      <div key={idx} className="result-card">
                        <h4>{item.fileName}</h4>
                        <div className="result-image">
                          <img src={`data:image/png;base64,${item.image}`} alt="Watermarked" />
                        </div>
                        <button
                          className="download-button"
                          onClick={() => downloadEmbeddedImage(item.image, item.fileName)}
                        >
                          <Download size={18} />
                          Download
                        </button>
                      </div>
                    ))}
                  </div>
                  <div className="result-actions">
                    <button className="download-button secondary-button" onClick={handleEmbedAnother}>
                      <Upload size={18} />
                      Process Another Batch
                    </button>
                  </div>
                </>
              )}

              {result.type === 'extract' && (
                <>
                  <div className="results-list">
                    {result.data.map((item, idx) => (
                      <div key={idx} className="result-card">
                        <h4>{item.fileName}</h4>
                        <div className="extraction-results">
                          <div className="result-item">
                            <strong>Extracted Payload:</strong>
                            <code>{item.extracted_payload}</code>
                          </div>
                          <div className="result-item">
                            <strong>Bit Error Rate (BER):</strong>
                            <span>{(item.bit_error_rate * 100).toFixed(2)}%</span>
                          </div>
                          <div className="result-item">
                            <strong>Confidence:</strong>
                            <span>{(item.confidence * 100).toFixed(2)}%</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}

              {result.type === 'detect' && (
                <>
                  <div className="results-list">
                    {result.data.map((item, idx) => (
                      <div key={idx} className="result-card">
                        <h4>{item.fileName}</h4>
                        <div className="detection-results">
                          <div className="result-item">
                            <strong>Watermark Detected:</strong>
                            <span className={item.watermark_detected ? 'status-positive' : 'status-negative'}>
                              {item.watermark_detected ? 'YES' : 'NO'}
                            </span>
                          </div>
                          <div className="result-item">
                            <strong>Detection Confidence:</strong>
                            <span>{(item.detection_probability * 100).toFixed(2)}%</span>
                          </div>
                          <div className="result-item">
                            <strong>Bit Error Rate (BER):</strong>
                            <span>{(item.bit_error_rate * 100).toFixed(2)}%</span>
                          </div>
                          <div className="result-item">
                            <strong>Optimal Delta Found:</strong>
                            <span>{item.parameters.delta}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </section>
          )}
        </div>
      </main>

      {/* Confirmation Dialog */}
      {showConfirmDialog && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <AlertCircle size={24} className="modal-icon" />
              <h2>Switch Tab?</h2>
            </div>
            <p className="modal-message">
              You are about to switch to a different tab. Your current results will be cleared.
            </p>
            <div className="modal-actions">
              <button 
                className="modal-button cancel-button"
                onClick={handleCancelTabChange}
              >
                Cancel
              </button>
              <button 
                className="modal-button confirm-button"
                onClick={handleConfirmTabChange}
              >
                Proceed & Clear
              </button>
            </div>
          </div>
        </div>
      )}

      <footer className="footer">
        <p>DWT-QIM Watermarking | Research Implementation</p>
      </footer>
    </div>
  );
}
