import { useEffect, useState } from 'react'
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import { Search, MessageSquare, BarChart3, Send, ThumbsUp, ThumbsDown, FileText, MapPin, LogOut } from 'lucide-react'
import Login from './components/Login'
import Signup from './components/Signup'
import { apiFetch } from './api'
import './index.css'

function MainApp({ user, onLogout }) {
  const [activeTab, setActiveTab] = useState('ask')
  const [question, setQuestion] = useState('')
  const [isAsking, setIsAsking] = useState(false)
  const [answer, setAnswer] = useState(null)
  const [feedback, setFeedback] = useState(null)
  const [selectedCounty, setSelectedCounty] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [isSearching, setIsSearching] = useState(false)
  const [searchResults, setSearchResults] = useState(null)
  const [searchWarning, setSearchWarning] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [topicResults, setTopicResults] = useState(null)
  const [counties, setCounties] = useState([])

  useEffect(() => {
    let isMounted = true

    async function loadCounties() {
      try {
        const data = await apiFetch('/api/counties')
        if (isMounted) {
          setCounties(data.counties || [])
        }
      } catch (error) {
        console.error('County list load failed', error)
      }
    }

    loadCounties()

    return () => {
      isMounted = false
    }
  }, [])

  const handleAsk = async () => {
    if (!question.trim()) return
    setIsAsking(true)
    setAnswer({ text: "", confidence: "evaluating...", sources: [] })
    setFeedback(null)
    
    try {
      const token = localStorage.getItem('token');
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || ''}/api/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          question: question,
          county: selectedCounty === 'all' ? null : selectedCounty,
          top_k: 8
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorData = { detail: errorText || "Unknown error occurred" };
        try {
          errorData = JSON.parse(errorText);
        } catch {
          // Keep text fallback for non-JSON server/proxy errors.
        }
        setIsAsking(false);
        setAnswer({ text: errorData.detail || "Error occurred", confidence: "error", sources: [] });
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let currentText = "";
      let currentMetadata = { sources: [], confidence: "medium" };
      let pending = "";

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          pending += decoder.decode(value, { stream: true });
          const lines = pending.split('\n');
          pending = lines.pop() || "";
          
          for (const line of lines) {
            try {
              const data = JSON.parse(line);
              if (data.type === 'metadata') {
                currentMetadata = data;
                setAnswer({ text: currentText, confidence: currentMetadata.confidence, sources: currentMetadata.sources || [] });
              } else if (data.type === 'token') {
                currentText += data.content;
                setAnswer({ text: currentText, confidence: currentMetadata.confidence, sources: currentMetadata.sources || [] });
              }
            } catch (e) {
              console.error("Error parsing JSON chunk", e, line);
            }
          }
        }
      }
      if (pending.trim()) {
        try {
          const data = JSON.parse(pending);
          if (data.type === 'metadata') {
            currentMetadata = data;
            setAnswer({ text: currentText, confidence: currentMetadata.confidence, sources: currentMetadata.sources || [] });
          } else if (data.type === 'token') {
            currentText += data.content;
            setAnswer({ text: currentText, confidence: currentMetadata.confidence, sources: currentMetadata.sources || [] });
          }
        } catch (e) {
          console.error("Error parsing final JSON chunk", e, pending);
        }
      }
      setIsAsking(false);
    } catch (error) {
      console.error(error);
      setIsAsking(false);
      setAnswer({ text: "Failed to connect to the server.", confidence: "error", sources: [] });
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    setIsSearching(true)
    setSearchResults(null)
    setSearchWarning('')

    try {
      const token = localStorage.getItem('token')
      const data = await apiFetch('/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          query: searchQuery,
          county: selectedCounty === 'all' ? null : selectedCounty,
          top_k: 8
        })
      })
      setSearchResults(data.results || [])
      setSearchWarning(data.warning || '')
    } catch (error) {
      setSearchResults({ error: error.message })
    } finally {
      setIsSearching(false)
    }
  }

  const handleAnalyzeTopics = async () => {
    if (selectedCounty === 'all') {
      setTopicResults({ error: 'Select a county before running topic analysis.' })
      return
    }
    setIsAnalyzing(true)
    setTopicResults(null)

    try {
      const token = localStorage.getItem('token')
      const data = await apiFetch('/api/topics/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          county: selectedCounty,
          top_k: 3
        })
      })
      setTopicResults(data)
    } catch (error) {
      setTopicResults({ error: error.message })
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleFeedback = async (rating) => {
    setFeedback(rating)
    if (!answer) return
    try {
      const token = localStorage.getItem('token')
      await apiFetch('/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          question,
          answer: answer.text,
          rating,
          sources: answer.sources || []
        })
      })
    } catch (error) {
      console.error('Feedback save failed', error)
    }
  }

  return (
    <div className="animate-fade-in">
      <div style={{ position: 'absolute', top: '1rem', right: '1rem', zIndex: 50, display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <span style={{ color: 'var(--text-muted)' }}>{user?.name || user?.email}</span>
        <button onClick={onLogout} className="icon-btn" title="Logout" style={{ background: 'rgba(255,255,255,0.1)' }}>
          <LogOut size={18} />
        </button>
      </div>

      <div className="hero-section">
        <div className="hero-content">
          <div className="stat-pill" style={{ display: 'inline-flex', marginBottom: '1rem', background: 'rgba(255,255,255,0.1)' }}>
            County comprehensive plans
          </div>
          <h1 className="hero-title">Florida Policy Navigator</h1>
          <p className="hero-subtitle">Explore Florida county plans with semantic search and cited answers.</p>
          <div className="stat-pills">
            <span className="stat-pill"><strong>67</strong> Counties</span>
            <span className="stat-pill"><strong>270+</strong> Plans</span>
            <span className="stat-pill"><strong>Cited</strong> Answers</span>
          </div>
        </div>
        <div className="animate-float" style={{ opacity: 0.8 }}>
          <MapPin size={120} color="var(--sky-light)" strokeWidth={1} />
        </div>
      </div>

      <div className="tabs-container">
        <button 
          className={`tab-button ${activeTab === 'ask' ? 'active' : ''}`}
          onClick={() => setActiveTab('ask')}
        >
          <MessageSquare size={20} /> Ask
        </button>
        <button 
          className={`tab-button ${activeTab === 'search' ? 'active' : ''}`}
          onClick={() => setActiveTab('search')}
        >
          <Search size={20} /> Search
        </button>
        <button 
          className={`tab-button ${activeTab === 'topics' ? 'active' : ''}`}
          onClick={() => setActiveTab('topics')}
        >
          <BarChart3 size={20} /> Topic analysis
        </button>
      </div>

      <div className="chat-container">
        <div className="chat-main">
          {activeTab === 'ask' && (
            <div className="glass-panel">
              <h2 style={{ marginTop: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                👉 Ask About Florida Policies
              </h2>
              <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>
                Answers use agent-style search: sub-questions are generated, then multiple retrievals are merged before the model writes a cited answer.
              </p>

              <div className="input-group">
                <textarea 
                  className="text-area"
                  placeholder="e.g., Does Collier County require wildlife surveys before development?"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                />
              </div>

              <button 
                className="primary-button" 
                onClick={handleAsk}
                disabled={isAsking}
                style={{ opacity: isAsking ? 0.7 : 1 }}
              >
                {isAsking ? 'Thinking...' : 'Get Answer'} <Send size={18} />
              </button>

              {answer && (
                <div className="response-card animate-fade-in glass-panel" style={{ marginTop: '2rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
                    <h3 style={{ margin: 0 }}>Answer</h3>
                    {answer.confidence !== 'error' && (
                      <span style={{ 
                        fontSize: '0.8rem', 
                        background: 'rgba(74, 222, 128, 0.2)', 
                        color: '#4ade80',
                        padding: '0.2rem 0.6rem',
                        borderRadius: '999px',
                        fontWeight: 'bold'
                      }}>
                        {answer.confidence.toUpperCase()} CONFIDENCE
                      </span>
                    )}
                  </div>
                  
                  <p style={{ fontSize: '1.1rem', lineHeight: '1.6' }}>{answer.text}</p>

                  {answer.sources?.length > 0 && (
                    <>
                      <h4 style={{ color: 'var(--sky-light)', marginTop: '2rem' }}>Sources</h4>
                      {answer.sources.map((source, idx) => (
                        <div key={idx} className="source-card">
                          <div className="source-title">
                            <FileText size={16} /> {source.citation || source.county || 'Document'}
                          </div>
                          <div style={{ fontStyle: 'italic', color: 'var(--text-muted)' }}>
                            "{source.text ? source.text.substring(0, 200) + '...' : (source.excerpt || 'No excerpt')}"
                          </div>
                        </div>
                      ))}
                    </>
                  )}

                  {answer.confidence !== 'error' && (
                    <div className="feedback-section">
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem', fontWeight: 600 }}>Was this answer helpful?</span>
                      <button 
                        className={`icon-btn ${feedback === 'up' ? 'active-up' : ''}`}
                        onClick={() => handleFeedback('up')}
                      >
                        <ThumbsUp size={18} />
                      </button>
                      <button 
                        className={`icon-btn ${feedback === 'down' ? 'active-down' : ''}`}
                        onClick={() => handleFeedback('down')}
                      >
                        <ThumbsDown size={18} />
                      </button>
                      </div>
                  )}
                </div>
              )}
            </div>
          )}

          {activeTab === 'search' && (
            <div className="glass-panel">
              <h2>🔍 Semantic Search</h2>
              <p style={{ color: 'var(--text-muted)' }}>Find relevant policy excerpts without generating a full answer.</p>
              <input 
                className="text-area" 
                style={{ minHeight: '60px', padding: '1rem' }}
                placeholder="e.g., wildlife corridor habitat connectivity" 
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <br/><br/>
              <button className="primary-button" onClick={handleSearch} disabled={isSearching}>
                {isSearching ? 'Searching...' : 'Search'}
              </button>

              {searchResults && (
                <div style={{ marginTop: '2rem' }}>
                  {searchResults.error ? (
                    <div style={{ color: '#f87171' }}>{searchResults.error}</div>
                  ) : (
                    <>
                      {searchWarning && (
                        <div style={{ color: '#fbbf24', marginBottom: '1rem' }}>{searchWarning}</div>
                      )}
                      <h3>Found {searchResults.length} excerpts</h3>
                      {searchResults.map((source, idx) => (
                        <div key={`${source.citation}-${idx}`} className="source-card">
                          <div className="source-title">
                            <FileText size={16} /> {source.citation}
                          </div>
                          <div style={{ color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                            Distance: {Number(source.distance).toFixed(3)}
                          </div>
                          <div>{source.text}</div>
                        </div>
                      ))}
                    </>
                  )}
                </div>
              )}
            </div>
          )}

          {activeTab === 'topics' && (
            <div className="glass-panel">
              <h2 style={{ marginTop: 0 }}>📊 Topic Analysis</h2>
              <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>
                Quick scan of retrieval strength per environmental topic for the selected county.
              </p>
              <button 
                className="primary-button"
                onClick={handleAnalyzeTopics}
                disabled={isAnalyzing}
                style={{ opacity: isAnalyzing ? 0.7 : 1 }}
              >
                {isAnalyzing ? 'Analyzing...' : 'Analyze Topics'}
              </button>

              {topicResults && (
                <div style={{ marginTop: '2rem' }}>
                  {topicResults.error ? (
                    <div style={{ color: '#f87171' }}>{topicResults.error}</div>
                  ) : (
                    <>
                      <h3 style={{ marginBottom: '1.5rem' }}>Results for {topicResults.county}</h3>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        {topicResults.results.map((result) => (
                          <div key={result.topic_key} className="source-card animate-fade-in" style={{ borderLeft: `4px solid ${result.has_evidence ? '#4ade80' : '#f87171'}` }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                              <strong style={{ fontSize: '1.1rem', color: 'white' }}>{result.topic}</strong>
                              <span style={{ fontSize: '0.8rem', padding: '0.2rem 0.5rem', borderRadius: '4px', background: result.has_evidence ? 'rgba(74, 222, 128, 0.2)' : 'rgba(248, 113, 113, 0.2)', color: result.has_evidence ? '#4ade80' : '#f87171' }}>
                                {result.has_evidence ? 'EVIDENCE LIKELY' : 'NO CLEAR MATCH'}
                              </span>
                            </div>
                            <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '0.95rem' }}>{result.description}</p>
                            {result.sources?.slice(0, 2).map((source, idx) => (
                              <div key={`${result.topic_key}-${idx}`} style={{ marginTop: '1rem', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                                <strong style={{ color: 'var(--sky-light)' }}>{source.citation}</strong>
                                <div>{source.text.substring(0, 300)}...</div>
                              </div>
                            ))}
                          </div>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="chat-sidebar">
          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <div className="input-group">
              <span className="input-label">Search Scope</span>
              <select value={selectedCounty} onChange={(e) => setSelectedCounty(e.target.value)}>
                <option value="all">All Counties</option>
                {counties.map((county) => (
                  <option key={county} value={county}>{county}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function App() {
  // Manage user session
  const [user, setUser] = useState(() => {
    const savedUser = localStorage.getItem('user')
    const savedToken = localStorage.getItem('token')
    if (savedUser && savedToken) {
      return JSON.parse(savedUser)
    }
    return null
  })
  
  const navigate = useNavigate()

  const handleLogin = (token, userData) => {
    localStorage.setItem('token', token)
    localStorage.setItem('user', JSON.stringify(userData))
    setUser(userData)
  }

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    setUser(null)
    navigate('/login')
  }

  return (
    <Routes>
      <Route 
        path="/login" 
        element={user ? <Navigate to="/" /> : <Login onLogin={handleLogin} />} 
      />
      <Route 
        path="/signup" 
        element={user ? <Navigate to="/" /> : <Signup onLogin={handleLogin} />} 
      />
      <Route 
        path="/" 
        element={
          user ? (
            <MainApp user={user} onLogout={handleLogout} />
          ) : (
            <Navigate to="/login" />
          )
        } 
      />
    </Routes>
  )
}

export default App
