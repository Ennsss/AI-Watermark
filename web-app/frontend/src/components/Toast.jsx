import React, { useEffect } from 'react';
import { CheckCircle, X } from 'lucide-react/dist/cjs/lucide-react';
import './Toast.css';

export default function Toast({ message, onDismiss }) {
  useEffect(() => {
    if (!message) return undefined;
    const timer = window.setTimeout(onDismiss, 4500);
    return () => window.clearTimeout(timer);
  }, [message, onDismiss]);

  if (!message) return null;
  return (
    <div className="success-toast" role="status" aria-live="polite">
      <CheckCircle size={20} />
      <span>{message}</span>
      <button type="button" onClick={onDismiss} aria-label="Dismiss notification"><X size={17} /></button>
    </div>
  );
}
