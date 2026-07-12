import React from 'react';
import { Search, X } from 'lucide-react/dist/cjs/lucide-react';
import './SearchInput.css';

export default function SearchInput({ value, onChange, placeholder, label }) {
  return (
    <div className="record-search" role="search">
      <Search size={18} aria-hidden="true" />
      <input type="search" value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} aria-label={label} />
      {value && <button type="button" onClick={() => onChange('')} aria-label="Clear search"><X size={17} /></button>}
    </div>
  );
}
