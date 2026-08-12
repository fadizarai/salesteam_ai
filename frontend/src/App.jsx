import React, { useState, useEffect } from 'react';
import {
  Search,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Info,
  Mic,
  X,
  Loader2,
  Sparkles,
} from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:8000/api';

const COMPANIES = ['LSAT', 'NEWTECH', 'ONETEL'];



const DEFAULT_PRESET_CLIENTS = [
  { id: 'CLT091206', label: 'CLT091206' },
  { id: 'CLT011712', label: 'CLT011712' },
  { id: 'CLT070730', label: 'CLT070730' },
  { id: 'CLT100521', label: 'CLT100521' },
  { id: 'CLT009160', label: 'CLT009160' },
  { id: 'CLT001977', label: 'CLT001977' },
];

const AI_TOGGLES = [
  { key: 'use_order_history', label: 'Historique des commandes' },
  { key: 'use_seasonality', label: 'Saisonnalité' },
  { key: 'use_localisation', label: 'Localisation GPS' },
  { key: 'use_category', label: 'Catégorie produit' },
];

export default function App() {
  const [company, setCompany] = useState('LSAT');
  const [selectedClient, setSelectedClient] = useState('CLT091206');
  const [searchQuery, setSearchQuery] = useState('');
  const [availableClients, setAvailableClients] = useState(DEFAULT_PRESET_CLIENTS);
  const [nbSuggestions, setNbSuggestions] = useState(5);
  const [aiToggles, setAiToggles] = useState({
    use_order_history: true,
    use_seasonality: true,
    use_localisation: true,
    use_category: true,
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [quantities, setQuantities] = useState({});
  const [itemStatuses, setItemStatuses] = useState({});
  const [selectedCategory, setSelectedCategory] = useState('ALL');
  const [selectedItemForModal, setSelectedItemForModal] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [chatOpen, setChatOpen] = useState(false);

  // ── Load Client List from API on Mount ──
  useEffect(() => {
    const loadAvailableClients = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/clients?limit=50`);
        if (res.ok) {
          const data = await res.json();
          if (data.clients && data.clients.length > 0) {
            setAvailableClients(
              data.clients.map((c) => ({
                id: c.code_client,
                label: c.code_client,
              }))
            );
            setSelectedClient(data.clients[0].code_client);
          }
        }
      } catch (err) {
        console.warn('Could not load client list, using presets:', err);
      }
    };
    loadAvailableClients();
  }, []);

  // ── API Call ──
  const fetchRecommendation = async (clientId) => {
    const activeId = clientId || selectedClient;
    if (!activeId) return;

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          client_id: activeId,
          commercial_id: `COMMERCIAL_${company}`,
          company: company,
          config: {
            use_order_history: aiToggles.use_order_history,
            use_seasonality: aiToggles.use_seasonality,
            use_localisation: aiToggles.use_localisation,
            use_category: aiToggles.use_category,
            nb_suggestions: parseInt(nbSuggestions),
          },
        }),
      });

      if (!response.ok) {
        throw new Error(`Erreur serveur (${response.status})`);
      }

      const data = await response.json();
      setRecommendation(data);

      const initialQty = {};
      const initialStatus = {};
      data.suggestions.forEach((item) => {
        initialQty[item.code_article] = item.quantite_suggeree;
        initialStatus[item.code_article] = 'pending';
      });
      setQuantities(initialQty);
      setItemStatuses(initialStatus);
    } catch (err) {
      console.error(err);
      setError(err.message || "Impossible de contacter l'API.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecommendation(selectedClient);
  }, [selectedClient, nbSuggestions]);

  // ── Handlers ──
  const handleSelectClientChip = (clientId) => {
    setSelectedClient(clientId);
    setSearchQuery('');
    fetchRecommendation(clientId);
  };

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    const targetCode = searchQuery.trim() ? searchQuery.trim().toUpperCase() : selectedClient;
    if (targetCode) {
      setSelectedClient(targetCode);
      fetchRecommendation(targetCode);
    }
  };

  const handleGenerateClick = () => {
    const targetCode = searchQuery.trim() ? searchQuery.trim().toUpperCase() : selectedClient;
    if (targetCode) {
      setSelectedClient(targetCode);
      fetchRecommendation(targetCode);
    }
  };

  const toggleStatus = (codeArticle, newStatus) => {
    setItemStatuses((prev) => ({
      ...prev,
      [codeArticle]: prev[codeArticle] === newStatus ? 'pending' : newStatus,
    }));
  };

  const handleQtyChange = (codeArticle, val) => {
    const num = parseInt(val) || 1;
    setQuantities((prev) => ({ ...prev, [codeArticle]: num }));
  };

  const handleToggle = (key) => {
    setAiToggles((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // ── Filtered suggestions ──
  const filteredSuggestions = recommendation?.suggestions || [];

  // ── Filtered client chips ──
  const filteredClients = searchQuery
    ? availableClients.filter((c) =>
        c.id.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : availableClients.slice(0, 10);

  return (
    <>
      {/* ─── NAVBAR ─── */}
      <nav className="navbar">
        <div className="navbar-brand">
          <div className="navbar-logo">
            <Sparkles size={20} />
          </div>
          <span className="navbar-title">SalesTeam AI</span>
        </div>
        <div className="company-pills">
          {COMPANIES.map((c) => (
            <button
              key={c}
              className={`company-pill ${company === c ? 'active' : ''}`}
              onClick={() => setCompany(c)}
            >
              {c}
            </button>
          ))}
        </div>
      </nav>

      {/* ─── MAIN ─── */}
      <div className="main-container">


        {/* ── Panels Row ── */}
        <div className="panels-row">
          {/* Client Selection */}
          <div className="panel-card">
            <div className="section-header">
              <div className="section-title">Sélection du Client</div>
            </div>
            <form onSubmit={handleSearchSubmit}>
              <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
                <div className="search-input-wrapper" style={{ flex: 1, marginBottom: 0 }}>
                  <Search size={18} className="search-icon" />
                  <input
                    type="text"
                    className="search-input"
                    placeholder="Saisir un code client (ex: CLT091206)..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                </div>
                <button type="submit" className="btn-primary" style={{ padding: '0 16px', borderRadius: '8px' }}>
                  Rechercher
                </button>
              </div>
            </form>

            <div className="client-chips-scroll">
              {filteredClients.map((preset) => (
                <button
                  key={preset.id}
                  className={`client-chip ${selectedClient === preset.id ? 'active' : ''}`}
                  onClick={() => handleSelectClientChip(preset.id)}
                >
                  {preset.id}
                </button>
              ))}
              {filteredClients.length === 0 && searchQuery && (
                <div style={{ fontSize: '0.82rem', color: '#6b7280', padding: '6px 0' }}>
                  Aucune suggestion directe dans les puces. Appuyez sur <strong>Rechercher</strong> pour tester <code>{searchQuery.toUpperCase()}</code>.
                </div>
              )}
            </div>
          </div>

          {/* AI Configuration */}
          <div className="panel-card">
            <div className="section-header">
              <div className="section-title">Configuration IA</div>
            </div>
            <div className="config-slider-row">
              <span className="config-slider-label">Nombre de suggestions</span>
              <span className="config-slider-value">{nbSuggestions}</span>
            </div>
            <input
              type="range"
              min="3"
              max="15"
              value={nbSuggestions}
              onChange={(e) => setNbSuggestions(e.target.value)}
              className="config-slider"
            />
            {AI_TOGGLES.map((toggle) => (
              <div className="toggle-row" key={toggle.key}>
                <span className="toggle-label">{toggle.label}</span>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={aiToggles[toggle.key]}
                    onChange={() => handleToggle(toggle.key)}
                  />
                  <span className="toggle-track"></span>
                </label>
              </div>
            ))}
          </div>
        </div>

        {/* ── Generate Button ── */}
        <button
          className="generate-btn"
          onClick={handleGenerateClick}
          disabled={loading}
        >
          {loading ? (
            <>
              <Loader2 size={20} style={{ animation: 'spin 1s linear infinite' }} />
              Calcul en cours...
            </>
          ) : (
            <>
              <Sparkles size={20} />
              Générer le Projet de Commande ({selectedClient})
            </>
          )}
        </button>

        {/* ── Results ── */}
        <section>
          <div className="results-header">
            <div>
              <div className="section-title">
                Projet de Commande — <strong>{selectedClient}</strong>
              </div>
              <div className="section-subtitle">
                {filteredSuggestions.length} article(s) recommandé(s)
              </div>
            </div>
            <button
              className="btn-refresh"
              onClick={() => fetchRecommendation(selectedClient)}
              title="Actualiser"
            >
              <RefreshCw size={16} />
            </button>
          </div>

          {/* Loading State */}
          {loading && (
            <div className="state-message">
              <div className="spinner"></div>
              <p>Calcul des probabilités par XGBoost pour {selectedClient}...</p>
            </div>
          )}

          {/* Error State */}
          {error && (
            <div className="state-message" style={{ color: '#dc2626' }}>
              <XCircle size={40} style={{ margin: '0 auto 12px' }} />
              <p>{error}</p>
              <button
                className="btn-retry"
                onClick={() => fetchRecommendation(selectedClient)}
              >
                Réessayer
              </button>
            </div>
          )}

          {/* Empty State */}
          {!loading && !error && filteredSuggestions.length === 0 && (
            <div className="state-message">
              <Info size={40} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
              <p>Aucun produit recommandé pour ce client / cette catégorie.</p>
            </div>
          )}

          {/* Results Grid */}
          {!loading && !error && filteredSuggestions.length > 0 && (
            <div className="results-grid">
              {filteredSuggestions.map((item) => {
                const probPct = (item.score_confiance * 100).toFixed(1);
                const isHigh = item.score_confiance >= 0.80;
                const status = itemStatuses[item.code_article] || 'pending';

                return (
                  <div
                    className="result-card"
                    key={item.code_article}
                    onClick={() => setSelectedItemForModal(item)}
                    style={{ cursor: 'pointer' }}
                  >
                    <div className="result-card-header">
                      <div>
                        <div className="result-article-name">
                          {item.designation}
                        </div>
                        <div className="result-article-code">
                          {item.code_article} • {item.categorie}
                        </div>
                      </div>
                      <button
                        className="btn-ai-analysis-trigger"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedItemForModal(item);
                        }}
                      >
                        <Sparkles size={14} />
                        <span>Analyse IA</span>
                      </button>
                    </div>

                    {/* Probability bar */}
                    <div className="result-proba-row">
                      <div className="proba-bar-bg">
                        <div
                          className="proba-bar-fill proba-fill-blue"
                          style={{ width: `${probPct}%` }}
                        ></div>
                      </div>
                      <span className="proba-text">{probPct}%</span>
                    </div>

                    {/* Actions */}
                    <div
                      className="result-actions"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <input
                        type="number"
                        min="1"
                        max="1000"
                        value={
                          quantities[item.code_article] || item.quantite_suggeree
                        }
                        onChange={(e) =>
                          handleQtyChange(item.code_article, e.target.value)
                        }
                        className="qty-input"
                      />
                      <button
                        className={`btn-accept ${status === 'accepted' ? 'active' : ''}`}
                        onClick={() =>
                          toggleStatus(item.code_article, 'accepted')
                        }
                      >
                        <CheckCircle2 size={14} />
                        Accepter
                      </button>
                      <button
                        className={`btn-reject ${status === 'rejected' ? 'active' : ''}`}
                        onClick={() =>
                          toggleStatus(item.code_article, 'rejected')
                        }
                      >
                        <XCircle size={14} />
                        Rejeter
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </div>

      {/* ─── PREMIUM LLM EXPLANATION MODAL ─── */}
      {selectedItemForModal && (
        <div
          className="modal-overlay"
          onClick={() => setSelectedItemForModal(null)}
        >
          <div
            className="modal-card"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <div className="modal-title-row">
                <div className="modal-icon-badge">
                  <Sparkles size={20} />
                </div>
                <div>
                  <h3 className="modal-title">Analyse & Justification IA</h3>
                  <p className="modal-subtitle">SalesTeam Intelligence • Llama 3.3</p>
                </div>
              </div>
              <button
                className="modal-close-btn"
                onClick={() => setSelectedItemForModal(null)}
              >
                <X size={20} />
              </button>
            </div>

            <div className="modal-body">
              <div className="modal-product-box">
                <div className="modal-product-name">
                  {selectedItemForModal.designation}
                </div>
                <div className="modal-product-code">
                  Code: <code>{selectedItemForModal.code_article}</code> • Catégorie: <strong>{selectedItemForModal.categorie}</strong>
                </div>
              </div>

              <div className="modal-stats-row">
                <div className="modal-stat-card">
                  <div className="stat-label">Probabilité de Réachat</div>
                  <div className="stat-value">
                    {(selectedItemForModal.score_confiance * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="modal-stat-card">
                  <div className="stat-label">Quantité Recommandée</div>
                  <div className="stat-value highlight">
                    {selectedItemForModal.quantite_suggeree} <span style={{ fontSize: '0.85rem', color: '#6b7280' }}>unités</span>
                  </div>
                </div>
              </div>

              <div className="modal-llm-quote-box">
                <div className="llm-quote-header">
                  <Sparkles size={16} color="#1a56e8" />
                  <span>Explication IA du Recommandation</span>
                </div>
                <p className="llm-quote-text">
                  "{selectedItemForModal.explication}"
                </p>
              </div>
            </div>

            <div className="modal-footer">
              <button
                className="btn-primary"
                onClick={() => setSelectedItemForModal(null)}
                style={{ width: '100%', padding: '12px' }}
              >
                Fermer l'analyse
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ─── CHATBOT FAB ─── */}
      <button className="chatbot-fab" onClick={() => setChatOpen(!chatOpen)}>
        {chatOpen ? <X size={24} /> : <Mic size={24} />}
      </button>

      {chatOpen && (
        <div className="chatbot-panel">
          <div className="chatbot-header">
            <span className="chatbot-title">🎤 Assistant Vocal IA</span>
            <button className="chatbot-close" onClick={() => setChatOpen(false)}>
              ✕
            </button>
          </div>
          <div className="chatbot-body">
            <div>
              <Mic size={40} style={{ color: '#d1d5db', marginBottom: '16px' }} />
              <p>Fonctionnalité vocale à venir...</p>
              <p style={{ fontSize: '0.78rem', marginTop: '8px', color: '#9ca3af' }}>
                L'assistant vocal vous permettra d'interagir avec l'IA en langage naturel.
              </p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
