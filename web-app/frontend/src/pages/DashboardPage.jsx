import React, { useState, useEffect } from 'react';
import { Activity } from 'lucide-react/dist/cjs/lucide-react';
import axios from 'axios';
import '../styles/Dashboard.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function DashboardPage() {
  const [stats, setStats] = useState(null);
  const [activity, setActivity] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const [statsRes, activityRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/dashboard/summary`),
        axios.get(`${API_BASE_URL}/api/dashboard/recent-activity`),
      ]);

      setStats(statsRes.data.data);
      setActivity(activityRes.data.data);
      setError(null);
    } catch (err) {
      setError('Failed to load dashboard data');
      // eslint-disable-next-line no-console
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <div className="loading">Loading dashboard...</div>;

  return (
    <div className="dashboard-page">
      <h1>Dashboard</h1>

      {error && <div className="error-message">{error}</div>}

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-value">{stats?.total_artworks || 0}</div>
          <div className="stat-label">Registered Artworks</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.watermarked_artworks || 0}</div>
          <div className="stat-label">Watermarked</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.total_verifications || 0}</div>
          <div className="stat-label">Verifications</div>
        </div>
        <div className="stat-card highlight">
          <div className="stat-value">{stats?.matches || 0}</div>
          <div className="stat-label">Verified Matches</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.partials || 0}</div>
          <div className="stat-label">Partial Detections</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{stats?.no_matches || 0}</div>
          <div className="stat-label">No Valid Watermark</div>
        </div>
      </div>

      <div className="recent-activity-section">
        <h2>
          <Activity size={20} /> Recent Activity
        </h2>
        <div className="activity-list">
          {activity.length === 0 ? (
            <p className="empty-state">No recent activity</p>
          ) : (
            activity.map((event, idx) => (
              <div key={idx} className="activity-item">
                <div className="activity-type">{event.type.replace(/_/g, ' ').toUpperCase()}</div>
                <div className="activity-text">{event.text}</div>
                <div className="activity-time">
                  {new Date(event.timestamp).toLocaleString()}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
