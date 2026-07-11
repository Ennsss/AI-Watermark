import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import {
  Home,
  Image,
  Plus,
  CheckCircle,
  History,
  Menu,
  X,
} from 'lucide-react/dist/cjs/lucide-react';

import DashboardPage from './pages/DashboardPage';
import ArtworksPage from './pages/ArtworksPage';
import ArtworkDetailPage from './pages/ArtworkDetailPage';
import RegisterArtworkPage from './pages/RegisterArtworkPage';
import VerifyImagePage from './pages/VerifyImagePage';
import VerificationHistoryPage from './pages/VerificationHistoryPage';
import artifactLogo from './artifact-logo.png';

import './App.css';

function AppLayout({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  return (
    <div className="app-layout">
      <button
        className="mobile-menu-toggle"
        onClick={() => setSidebarOpen(!sidebarOpen)}
      >
        {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
      </button>

      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <img className="artifact-logo" src={artifactLogo} alt="Artifact" />
          <p className="brand-byline">by ChickenScratch Co.</p>
        </div>

        <nav className="sidebar-nav">
          <Link
            to="/"
            className={`nav-link ${isActive('/') ? 'active' : ''}`}
            onClick={() => setSidebarOpen(false)}
          >
            <Home size={20} /> Dashboard
          </Link>
          <Link
            to="/artworks"
            className={`nav-link ${isActive('/artworks') ? 'active' : ''}`}
            onClick={() => setSidebarOpen(false)}
          >
            <Image size={20} /> My Artworks
          </Link>
          <Link
            to="/register"
            className={`nav-link ${isActive('/register') ? 'active' : ''}`}
            onClick={() => setSidebarOpen(false)}
          >
            <Plus size={20} /> Register Artwork
          </Link>
          <Link
            to="/verify"
            className={`nav-link ${isActive('/verify') ? 'active' : ''}`}
            onClick={() => setSidebarOpen(false)}
          >
            <CheckCircle size={20} /> Verify Image
          </Link>
          <Link
            to="/history"
            className={`nav-link ${isActive('/history') ? 'active' : ''}`}
            onClick={() => setSidebarOpen(false)}
          >
            <History size={20} /> Verification History
          </Link>
        </nav>

        <div className="sidebar-footer">
          <div className="disclaimer-notice">
            <p>
              This system provides technical provenance support through registry records and
              watermark verification. A successful result does not constitute legal proof of
              ownership.
            </p>
          </div>
        </div>
      </aside>

      <main className="main-content" onClick={() => setSidebarOpen(false)}>
        {children}
      </main>
    </div>
  );
}

function AppPages() {
  return (
    <Routes>
      <Route path="/" element={<DashboardPage />} />
      <Route path="/artworks" element={<ArtworksPage />} />
      <Route path="/artworks/:artworkId" element={<ArtworkDetailPage />} />
      <Route path="/register" element={<RegisterArtworkPage />} />
      <Route path="/verify" element={<VerifyImagePage />} />
      <Route path="/history" element={<VerificationHistoryPage />} />
    </Routes>
  );
}

export default function App() {
  return (
    <Router>
      <AppLayout>
        <AppPages />
      </AppLayout>
    </Router>
  );
}
