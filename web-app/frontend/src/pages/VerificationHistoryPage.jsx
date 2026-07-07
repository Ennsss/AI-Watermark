import React, { useEffect, useState } from 'react';
import {
  Eye,
  Download,
  CheckCircle,
  AlertCircle,
  XCircle,
  Sparkles,
  Clock3,
  FileText,
  Image as ImageIcon,
  ArrowLeft,
} from 'lucide-react/dist/cjs/lucide-react';
import axios from 'axios';
import '../styles/VerificationHistoryPage.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const getResultIcon = (status) => {
  switch (status) {
    case 'match':
      return <CheckCircle size={18} className="icon-success" />;
    case 'partial':
      return <AlertCircle size={18} className="icon-warning" />;
    default:
      return <XCircle size={18} className="icon-error" />;
  }
};

const getResultLabel = (status) => {
  switch (status) {
    case 'match':
      return 'Verified Match';
    case 'partial':
      return 'Partial Detection';
    case 'no_match':
      return 'No Valid Watermark';
    default:
      return status;
  }
};

export default function VerificationHistoryPage() {
  const [verifications, setVerifications] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedVerification, setSelectedVerification] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    fetchVerifications();
  }, []);

  const fetchVerifications = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/api/verifications`);
      setVerifications(response.data.data);
      setError(null);
    } catch (err) {
      setError('Failed to load verification history');
      // eslint-disable-next-line no-console
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetails = async (verificationId) => {
    try {
      setDetailLoading(true);
      const response = await axios.get(`${API_BASE_URL}/api/verifications/${verificationId}`);
      setSelectedVerification(response.data.data);
    } catch (err) {
      setError('Failed to load verification details');
    } finally {
      setDetailLoading(false);
    }
  };

  const handleDownloadReport = async (verificationId) => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/api/verifications/${verificationId}/report.csv`,
        { responseType: 'blob' }
      );
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `verification_${verificationId}.csv`);
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
    } catch (err) {
      setError('Failed to download report');
    }
  };

  if (loading) return <div className="loading">Loading verification history...</div>;

  if (selectedVerification) {
    return (
      <div className="history-page">
        <button className="back-link" onClick={() => setSelectedVerification(null)}>
          <ArrowLeft size={18} /> Back to History
        </button>

        <section className="verification-detail-card">
          <div className="detail-hero">
            <div>
              <p className="page-kicker">
                <Sparkles size={14} /> Verification record
              </p>
              <h1>Verification Details</h1>
              <p className="page-subtitle">
                A complete record of this watermark check, including the result, timing, and linked artwork.
              </p>
            </div>
            <div className={`result-pill ${selectedVerification.result_status}`}>
              {getResultIcon(selectedVerification.result_status)}
              <span>{getResultLabel(selectedVerification.result_status)}</span>
            </div>
          </div>

          <div className="verification-detail-grid">
            <div className="info-panel">
              <h2>
                <FileText size={18} /> Record Summary
              </h2>
              <div className="detail-stack">
                <div className="detail-item">
                  <span className="label">Verification ID</span>
                  <span className="value">{selectedVerification.verification_id}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Artwork ID</span>
                  <span className="value">{selectedVerification.artwork_id}</span>
                </div>
                {selectedVerification.artwork_title && (
                  <div className="detail-item">
                    <span className="label">Artwork Title</span>
                    <span className="value">{selectedVerification.artwork_title}</span>
                  </div>
                )}
                {selectedVerification.artwork_creator && (
                  <div className="detail-item">
                    <span className="label">Creator</span>
                    <span className="value">{selectedVerification.artwork_creator}</span>
                  </div>
                )}
              </div>
            </div>

            <div className="info-panel">
              <h2>
                <Clock3 size={18} /> Verification Metrics
              </h2>
              <div className="detail-stack">
                <div className="detail-item">
                  <span className="label">Suspected File</span>
                  <span className="value">{selectedVerification.suspected_filename}</span>
                </div>
                <div className="detail-item">
                  <span className="label">Verification Date</span>
                  <span className="value">
                    {new Date(selectedVerification.verification_date).toLocaleString()}
                  </span>
                </div>
                <div className="detail-item">
                  <span className="label">BER</span>
                  <span className="value">
                    {selectedVerification.ber !== null ? selectedVerification.ber.toFixed(4) : 'N/A'}
                  </span>
                </div>
                <div className="detail-item">
                  <span className="label">Processing Time</span>
                  <span className="value">
                    {selectedVerification.processing_time_ms.toFixed(2)}ms
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="detail-actions">
            <button
              className="btn btn-primary"
              onClick={() => handleDownloadReport(selectedVerification.verification_id)}
            >
              <Download size={16} /> Download Technical Report
            </button>
            <button className="btn btn-outline" onClick={() => setSelectedVerification(null)}>
              Back to History
            </button>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="history-page">
      <div className="history-header">
        <div>
          <p className="page-kicker">
            <Sparkles size={14} /> Activity log
          </p>
          <h1>Verification History</h1>
          <p className="page-subtitle">
            Review verification events as individual records, each with its own summary and quick actions.
          </p>
        </div>
      </div>

      {error && <div className="error-message">{error}</div>}

      {verifications.length === 0 ? (
        <div className="empty-state">
          <p>No verifications yet. Start by registering an artwork and then verifying images.</p>
        </div>
      ) : (
        <div className="history-grid">
          {verifications.map((ver) => (
            <article key={ver.verification_id} className={`history-card ${ver.result_status}`}>
              <div className="history-card-top">
                <div className="history-card-title-block">
                  <p className="card-kicker">{ver.verification_id}</p>
                  <h2 className="scrollable-filename" title={ver.suspected_filename}>
                    {ver.suspected_filename}
                  </h2>
                </div>
                <div className={`result-pill ${ver.result_status}`}>
                  {getResultIcon(ver.result_status)}
                  <span>{getResultLabel(ver.result_status)}</span>
                </div>
              </div>

              <div className="history-meta-grid">
                <div className="meta-chip">
                  <ImageIcon size={14} />
                  <div>
                    <span className="meta-label">Artwork</span>
                    <span className="meta-value">{ver.artwork_id}</span>
                  </div>
                </div>

                <div className="meta-chip">
                  <Clock3 size={14} />
                  <div>
                    <span className="meta-label">Date</span>
                    <span className="meta-value">
                      {new Date(ver.verification_date).toLocaleDateString()}
                    </span>
                  </div>
                </div>

                <div className="meta-chip">
                  <FileText size={14} />
                  <div>
                    <span className="meta-label">BER</span>
                    <span className="meta-value">
                      {ver.ber !== null ? ver.ber.toFixed(4) : 'N/A'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="history-card-footer">
                <button
                  className="btn btn-small btn-outline"
                  onClick={() => handleViewDetails(ver.verification_id)}
                  title="View details"
                  disabled={detailLoading}
                >
                  <Eye size={16} />
                  View details
                </button>
                <button
                  className="btn btn-small btn-outline"
                  onClick={() => handleDownloadReport(ver.verification_id)}
                  title="Download report"
                >
                  <Download size={16} />
                  Download
                </button>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
