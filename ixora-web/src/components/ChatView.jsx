import { useState, useRef, useEffect, useCallback } from 'react'
import { useAuth } from '../context/AuthContext'
import { useChat } from '../context/ChatContext'
import { toast } from './Toast'

const API = 'http://localhost:8000'

const DOMAIN_CFG = {
  bio: { label:'Biomedical',    model:'BioGPT',        color:'#6B8F71' },
  cs:  { label:'Comp. Science', model:'Qwen2.5-Coder', color:'#8A7650' },
  gen: { label:'General',       model:'Mistral-Large', color:'#8E977D' },
}

const DOMAIN_MAP = { bio:'biomed', cs:'cs', gen:'general' }

// ─── right-panel tabs ────────────────────────────────────────────────────────
const PANEL_TABS = ['papers', 'trace', 'causal', 'optimization']

// ─── XML Response Parser ──────────────────────────────────────────────────────
// Parses the XML-tagged responses from the backend and renders them as
// structured React elements. Without this, browser silently hides
// content inside unknown tags like <explanation>, <hypothesis>, etc.
function parseXmlResponse(text) {
  if (!text) return null

  const hasXml = /<(explanation|enthusiasm|hypothesis|clarify|followup|analysis)[\s>]/.test(text)

  // ── Plain text (fast path) ───────────────────────────────────────────────
  if (!hasXml) {
    return <PlainText text={text} />
  }

  // ── XML-structured response ──────────────────────────────────────────────
  const sections = []

  // Helper: extract a tag's content
  const extract = (tag) => {
    const re = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, 'i')
    const m = text.match(re)
    return m ? m[1].trim() : null
  }

  // 1. Enthusiasm / intro
  const enthusiasm = extract('enthusiasm')
  if (enthusiasm) {
    sections.push(
      <div key="enthusiasm" style={xmlCss.enthusiasm}>
        <span style={xmlCss.enthusiasmIcon}>✦</span>
        <PlainText text={enthusiasm} />
      </div>
    )
  }

  // 2. Clarify
  const clarify = extract('clarify')
  if (clarify) {
    sections.push(
      <XmlSection key="clarify" icon="❓" label="Clarifications" color="#8A7650">
        <PlainText text={clarify} />
      </XmlSection>
    )
  }

  // 3. Explanation (main body — most important)
  const explanation = extract('explanation')
  if (explanation) {
    sections.push(
      <XmlSection key="explanation" icon="□" label="Analysis" color="#4A6741" accent>
        <ExplanationBody text={explanation} />
      </XmlSection>
    )
  }

  // 4. Hypothesis
  const hypothesis = extract('hypothesis')
  if (hypothesis) {
    sections.push(
      <XmlSection key="hypothesis" icon="⬡" label="Hypothesis" color="#5B7A9D">
        <HypothesisBody text={hypothesis} />
      </XmlSection>
    )
  }

  // 5. Followup questions
  const followup = extract('followup')
  if (followup) {
    sections.push(
      <XmlSection key="followup" icon="→" label="Follow-up Questions" color="#8E977D">
        <PlainText text={followup} />
      </XmlSection>
    )
  }

  // If nothing matched (malformed XML), fall back to plain
  if (sections.length === 0) {
    return <PlainText text={text} />
  }

  return <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>{sections}</div>
}

// ── Renders plain text: **bold**, newlines, numbered lists ───────────────────
function PlainText({ text }) {
  if (!text) return null
  const lines = text.split('\n')
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
      {lines.map((line, i) => {
        if (!line.trim()) return <div key={i} style={{ height: '0.4rem' }} />
        // Bold markdown
        const parts = line.split(/\*\*(.*?)\*\*/g)
        return (
          <div key={i} style={{ lineHeight: 1.75 }}>
            {parts.map((part, j) =>
              j % 2 === 1
                ? <strong key={j}>{part}</strong>
                : part
            )}
          </div>
        )
      })}
    </div>
  )
}

// ── Renders the <explanation> block, splitting on **Bold Headings** ──────────
function ExplanationBody({ text }) {
  if (!text) return null

  // Split on **Heading** that appears at the start of a line
  const chunks = text.split(/(?=\n\*\*[^*]+\*\*)/)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {chunks.map((chunk, i) => {
        const headingMatch = chunk.match(/^\n?\*\*([^*]+)\*\*\n?(.*)$/s)
        if (headingMatch) {
          return (
            <div key={i}>
              <div style={xmlCss.sectionHeading}>{headingMatch[1]}</div>
              <PlainText text={headingMatch[2].trim()} />
            </div>
          )
        }
        return <PlainText key={i} text={chunk.trim()} />
      })}
    </div>
  )
}

// ── Renders the <hypothesis> block — parses H0/H1/Expected Effect etc. ───────
function HypothesisBody({ text }) {
  if (!text) return null

  const fields = [
    'H0 (Null Hypothesis)',
    'H1 (Alternative Hypothesis)',
    'Expected Effect',
    'Scientific Rationale',
    'Measurable Outcome',
    'Confounding Variables',
    'Statistical Test',
  ]

  const rows = []
  let remaining = text

  fields.forEach((field) => {
    const re = new RegExp(`\\*\\*${field.replace(/[()]/g, '\\$&')}:\\*\\*\\s*([\\s\\S]*?)(?=\\*\\*|$)`, 'i')
    const m = remaining.match(re)
    if (m) {
      rows.push({ label: field, value: m[1].trim() })
    }
  })

  if (rows.length === 0) {
    // Fallback: plain text
    return <PlainText text={text} />
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
      {rows.map(({ label, value }) => (
        <div key={label} style={xmlCss.hypRow}>
          <div style={xmlCss.hypLabel}>{label}</div>
          <div style={xmlCss.hypValue}>{value}</div>
        </div>
      ))}
    </div>
  )
}

// ── Wrapper for each named XML section ───────────────────────────────────────
function XmlSection({ icon, label, color, accent, children }) {
  const [open, setOpen] = useState(true)
  return (
    <div style={{
      border: `1px solid ${color}22`,
      borderRadius: 10,
      overflow: 'hidden',
      background: accent ? `${color}06` : 'transparent',
    }}>
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: '0.5rem',
          padding: '0.5rem 0.75rem',
          background: `${color}10`,
          cursor: 'pointer',
          userSelect: 'none',
        }}
        onClick={() => setOpen(v => !v)}
      >
        <span style={{ fontSize: '0.7rem', color }}>{icon}</span>
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: '0.65rem',
          fontWeight: 700, letterSpacing: '0.12em',
          textTransform: 'uppercase', color,
          flex: 1,
        }}>{label}</span>
        <span style={{ fontSize: '0.6rem', color, opacity: 0.6 }}>{open ? '▲' : '▼'}</span>
      </div>
      {open && (
        <div style={{ padding: '0.85rem 1rem', fontSize: '0.91rem', lineHeight: 1.78, color: 'var(--ink)' }}>
          {children}
        </div>
      )}
    </div>
  )
}

// ── Styles for XML rendering ─────────────────────────────────────────────────
const xmlCss = {
  enthusiasm: {
    display: 'flex', gap: '0.5rem', alignItems: 'flex-start',
    padding: '0.6rem 0.85rem',
    background: 'rgba(107,143,113,0.08)',
    borderLeft: '3px solid #6B8F71',
    borderRadius: '0 8px 8px 0',
    fontStyle: 'italic',
    color: 'var(--bark-dark)',
    fontSize: '0.91rem',
    lineHeight: 1.7,
  },
  enthusiasmIcon: {
    color: '#6B8F71', fontSize: '0.8rem', flexShrink: 0, marginTop: 3,
  },
  sectionHeading: {
    fontFamily: 'var(--font-sans)',
    fontWeight: 700,
    fontSize: '0.85rem',
    color: 'var(--ink)',
    marginBottom: '0.35rem',
    paddingBottom: '0.25rem',
    borderBottom: '1px solid rgba(138,118,80,0.15)',
  },
  hypRow: {
    display: 'grid',
    gridTemplateColumns: '160px 1fr',
    gap: '0.5rem',
    padding: '0.45rem 0',
    borderBottom: '1px solid rgba(138,118,80,0.08)',
  },
  hypLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.63rem',
    fontWeight: 700,
    color: '#5B7A9D',
    letterSpacing: '0.04em',
    paddingTop: 2,
  },
  hypValue: {
    fontSize: '0.87rem',
    color: 'var(--ink)',
    lineHeight: 1.65,
  },
}

// ─────────────────────────────────────────────────────────────────────────────

export default function ChatView({ chatId, domain, onBack }) {
  const { user, authFetch } = useAuth()
  const { history, appendMessage, updateChat, toggleBookmark } = useChat()

  const chat       = history.find(c => c.id === chatId)
  const cfg        = DOMAIN_CFG[domain] || DOMAIN_CFG.gen
  const initials   = user ? (user.first_name?.[0]||'')+(user.last_name?.[0]||'') : 'U'
  const bookmarked = chat?.bookmarked || false

  const [messages,    setMessages]    = useState(chat?.msgs || [])
  const [input,       setInput]       = useState('')
  const [loading,     setLoading]     = useState(false)
  const [chatTitle,   setChatTitle]   = useState(chat?.title || '')

  // Right panel
  const [panelTab,    setPanelTab]    = useState(null)
  const [papers,      setPapers]      = useState([])
  const [papersLoading, setPapersLoading] = useState(false)
  const [activePaper, setActivePaper] = useState(null)

  const [msgMeta,  setMsgMeta]  = useState({})

  // Causal analysis
  const [causalData,    setCausalData]    = useState(null)
  const [causalLoading, setCausalLoading] = useState(false)

  // Bayesian optimisation
  const [optData,    setOptData]    = useState(null)
  const [optPolling, setOptPolling] = useState(false)

  const [feedback, setFeedback] = useState({})

  const sessionIdRef = useRef(chatId)
  const scrollRef    = useRef(null)
  const inputRef     = useRef(null)
  const pollTimer    = useRef(null)

  // ── Greeting ────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!chat?.msgs?.length && messages.length === 0) {
      setMessages([{
        id: Date.now(), role:'ai',
        text: `Hello! I'm IXORA's **${cfg.label}** specialist, powered by ${cfg.model}.\n\nAsk me anything — experimental design, literature review, hypothesis generation. Every reasoning step is traced and visible in the panel.`,
        sources:[],
      }])
    }
  }, [chatId])

  // ── Pending prompt from suggestion chips ───────────────────────────────────
  useEffect(() => {
    const pending = sessionStorage.getItem('ixora_pending_prompt')
    if (pending) {
      sessionStorage.removeItem('ixora_pending_prompt')
      setTimeout(() => sendMessageWithText(pending), 120)
    }
  }, [chatId])

  // ── Auto-scroll ─────────────────────────────────────────────────────────────
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages, loading])

  // ── Cleanup polling on unmount ───────────────────────────────────────────────
  useEffect(() => () => clearInterval(pollTimer.current), [])

  // ── Send message ─────────────────────────────────────────────────────────────
  const sendMessageWithText = async (text) => {
    if (!text.trim() || loading) return
    setInput('')
    if (inputRef.current) inputRef.current.style.height = 'auto'

    const userMsg = { id: Date.now(), role:'user', text }
    setMessages(prev => [...prev, userMsg])
    if (!chatTitle) { const t = text.slice(0,50); setChatTitle(t); updateChat(chatId,{title:t}) }
    appendMessage(chatId, userMsg)
    setLoading(true)

    try {
      const res = await authFetch('/chat', {
        method:'POST',
        body: JSON.stringify({
          message:    text,
          session_id: sessionIdRef.current,
          domain:     DOMAIN_MAP[domain] || 'biomed',
        }),
      })
      if (!res.ok) {
        const err = await res.json().catch(()=>({}))
        throw new Error(err.detail?.message || err.detail || `Server error ${res.status}`)
      }
      const data = await res.json()

      const aiMsgId = Date.now() + 1
      const sources = (data.papers||[]).map(p=>p.title||'Source').filter(Boolean)

      const aiMsg = {
        id: aiMsgId, role:'ai',
        text: data.response || 'No response received.',
        sources,
        confidence: data.confidence != null ? Math.round(data.confidence * 100) : undefined,
        sessionId: data.session_id,
        queryHash: data.query_hash,
        usedPipeline: data.used_full_pipeline,
      }
      setMessages(prev => [...prev, aiMsg])
      appendMessage(chatId, aiMsg)

      setMsgMeta(prev => ({
        ...prev,
        [aiMsgId]: {
          trace:        data.trace        || [],
          parameters:   data.parameters   || {},
          intent:       data.intent,
          domain:       data.domain,
          processingTime: data.processing_time_seconds,
          rewardScore:  data.reward_score,
          usedPipeline: data.used_full_pipeline,
          optimizationNote: data.optimization_note,
        }
      }))

      sessionIdRef.current = data.session_id || sessionIdRef.current

      if (data.optimization_note) startOptimizationPolling(data.session_id)

      fetchPapers(text)

    } catch(e) {
      console.error('Chat error:', e)
      toast(e.message || 'Failed to reach backend', 'error')
      setMessages(prev => [...prev, {
        id: Date.now()+1, role:'ai',
        text:`⚠️ **Backend error:** ${e.message}\n\nMake sure the server is running at \`${API}\`.`,
        sources:[],
      }])
    } finally {
      setLoading(false)
    }
  }

  const sendMessage = () => sendMessageWithText(input.trim())
  const handleKey = (e) => { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage()} }
  const autoResize = (el) => { el.style.height='auto'; el.style.height=Math.min(el.scrollHeight,120)+'px' }

  // ── arXiv papers ─────────────────────────────────────────────────────────────
  const fetchPapers = async (query) => {
    setPapersLoading(true)
    try {
      const res = await fetch(`${API}/arxiv`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ query, max_papers: 80 }),
      })
      const data = await res.json()
      setPapers(data.links || [])
    } catch(e) {
      console.warn('arXiv fetch failed:', e)
    } finally {
      setPapersLoading(false)
    }
  }

  // ── Causal analysis ──────────────────────────────────────────────────────────
  const runCausalAnalysis = async () => {
    setCausalLoading(true)
    setCausalData(null)
    setPanelTab('causal')
    const lastAiMsg = [...messages].reverse().find(m=>m.role==='ai')
    const query = lastAiMsg?.text?.slice(0,200) || 'research query'
    try {
      const res = await fetch(`${API}/causal`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          query,
          session_id: sessionIdRef.current,
          include_links: true,
          domain: DOMAIN_MAP[domain] || 'biomed',
        }),
      })
      const data = await res.json()
      if (data.status === 'error' || data.status === 'timeout') {
        toast(data.error || 'Causal analysis failed', 'error')
        setCausalData({ error: data.error })
      } else {
        setCausalData(data)
        toast('Causal analysis complete ✓', 'success')
      }
    } catch(e) {
      toast('Causal analysis failed', 'error')
      setCausalData({ error: e.message })
    } finally {
      setCausalLoading(false)
    }
  }

  // ── Bayesian optimisation polling ────────────────────────────────────────────
  const startOptimizationPolling = (sid) => {
    clearInterval(pollTimer.current)
    setOptPolling(true)
    setOptData(null)
    let attempts = 0
    pollTimer.current = setInterval(async () => {
      attempts++
      try {
        const res = await fetch(`${API}/optimization/${sid}`)
        const data = await res.json()
        if (data.status === 'completed' || data.status === 'failed' || data.status === 'timeout') {
          setOptData(data)
          setOptPolling(false)
          clearInterval(pollTimer.current)
          toast(`Optimisation ${data.status} ✓`, data.status==='completed' ? 'success' : 'error')
        } else if (attempts >= 20) {
          setOptPolling(false)
          clearInterval(pollTimer.current)
        }
      } catch(e) {
        if (attempts >= 20) { setOptPolling(false); clearInterval(pollTimer.current) }
      }
    }, 3000)
  }

  const manualOptimizationCheck = async () => {
    setPanelTab('optimization')
    setOptPolling(true)
    try {
      const res = await fetch(`${API}/optimization/${sessionIdRef.current}`)
      const data = await res.json()
      setOptData(data)
      if (data.status === 'running' || data.status === 'not_started') startOptimizationPolling(sessionIdRef.current)
    } catch(e) { toast('Could not fetch optimisation status', 'error') }
    finally { setOptPolling(false) }
  }

  // ── Feedback ─────────────────────────────────────────────────────────────────
  const sendFeedback = async (msgId, pref) => {
    const msg = messages.find(m=>m.id===msgId)
    if (!msg) return
    setFeedback(prev=>({...prev,[msgId]:pref}))
    try {
      await fetch(`${API}/feedback`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          session_id: msg.sessionId || sessionIdRef.current,
          preference: pref,
          response: msg.text,
          query_hash: msg.queryHash || 'unknown',
        }),
      })
      toast(pref==='good' ? '👍 Feedback recorded' : '👎 Feedback recorded')
    } catch(e) { console.warn('Feedback failed:', e) }
  }

  const lastMetaEntry = Object.values(msgMeta).filter(m=>m.trace?.length>0).slice(-1)[0] || null

  const togglePanel = (tab) => setPanelTab(prev => prev===tab ? null : tab)

  const exportChat = () => {
    if (!messages.length) { toast('Nothing to export yet'); return }
    const text = messages.map(m=>`[${m.role.toUpperCase()}]\n${m.text}`).join('\n\n---\n\n')
    const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(new Blob([text],{type:'text/plain'})), download:`${chatTitle||'ixora-chat'}.txt` })
    a.click(); toast('Exported as .txt')
  }

  const handleBookmark = () => { toggleBookmark(chatId); toast(bookmarked?'Bookmark removed':'★ Chat bookmarked') }

  const panelOpen = panelTab !== null

  return (
    <div style={css.wrap}>
      {/* ── Topbar ── */}
      <div style={css.topbar}>
        <button style={css.backBtn} className="tb-btn" onClick={onBack}>← Back</button>
        <div style={css.domainBadge}>
          <span style={{width:6,height:6,borderRadius:'50%',background:cfg.color,display:'block'}}/>
          {cfg.label}
        </div>
        <input
          style={css.titleInput}
          value={chatTitle}
          onChange={e=>{setChatTitle(e.target.value);updateChat(chatId,{title:e.target.value})}}
          placeholder="Untitled conversation…"
        />
        <div style={css.topbarActions}>
          <button style={css.topBtn(panelTab==='papers')}       onClick={()=>togglePanel('papers')}>📄 Papers {papers.length>0&&`(${papers.length})`}</button>
          <button style={css.topBtn(panelTab==='trace')}        onClick={()=>togglePanel('trace')}>🔍 Trace {lastMetaEntry&&`(${lastMetaEntry.trace.length})`}</button>
          <button style={css.topBtn(panelTab==='causal')}       onClick={()=>{runCausalAnalysis()}}>🔬 Causal</button>
          <button style={css.topBtn(panelTab==='optimization')} onClick={manualOptimizationCheck}>
            {optPolling ? '⏳ Optimising…' : '⚙️ Optimisation'}
          </button>
          <button style={css.topBtn(bookmarked)} onClick={handleBookmark}>{bookmarked?'★':'☆'}</button>
          <button style={css.topBtn(false)} onClick={exportChat}>↓</button>
        </div>
      </div>

      {/* ── Body ── */}
      <div style={css.body}>

        {/* Messages pane */}
        <div style={css.messagesPane}>
          <div style={css.messagesScroll} ref={scrollRef}>
            <div style={css.divider}><span style={css.divLine}/><span>Start of conversation</span><span style={css.divLine}/></div>

            {messages.map(msg => (
              <Message
                key={msg.id}
                msg={msg}
                initials={initials}
                modelName={cfg.model}
                meta={msgMeta[msg.id]}
                feedbackState={feedback[msg.id]}
                onFeedback={(pref)=>sendFeedback(msg.id,pref)}
                onShowTrace={()=>{ setPanelTab('trace') }}
              />
            ))}
            {loading && <TypingIndicator />}
          </div>

          {/* Input */}
          <div style={css.inputArea}>
            <div style={css.inputWrap} className="input-wrap">
              <textarea
                ref={inputRef}
                style={css.textarea}
                placeholder={`Ask IXORA (${cfg.label})…`}
                value={input}
                onChange={e=>{setInput(e.target.value);autoResize(e.target)}}
                onKeyDown={handleKey}
                rows={1}
              />
              <div style={{display:'flex',alignItems:'center',gap:'0.4rem',flexShrink:0}}>
                <button style={css.sendBtn} onClick={sendMessage} disabled={loading||!input.trim()}>→</button>
              </div>
            </div>
            <div style={css.inputFooter}>
              <span>↵ send · shift+↵ newline</span>
              {lastMetaEntry && (
                <span style={{color:'var(--bark)'}}>
                  ⏱ {lastMetaEntry.processingTime}s ·{' '}
                  {lastMetaEntry.intent && <span>{lastMetaEntry.intent}</span>}
                  {lastMetaEntry.usedPipeline && <span> · full pipeline</span>}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Right panel */}
        <div style={{...css.panel, width: panelOpen ? (panelTab==='papers' && activePaper ? 680 : 440) : 0}}>
          {panelOpen && (
            <div style={css.panelInner}>

              <div style={css.panelTabBar}>
                {PANEL_TABS.map(t => (
                  <button key={t} style={css.panelTabBtn(panelTab===t)} onClick={()=>setPanelTab(t)}>
                    { t==='papers'       ? '📄 Papers'
                    : t==='trace'        ? '🔍 Trace'
                    : t==='causal'       ? '🔬 Causal'
                    :                     '⚙️ Optim.' }
                  </button>
                ))}
                <button style={css.panelClose} onClick={()=>setPanelTab(null)}>✕</button>
              </div>

              <div style={css.panelScroll}>
                {panelTab==='papers' && (
                  <PapersPanel
                    papers={papers}
                    loading={papersLoading}
                    activePaper={activePaper}
                    onSelect={setActivePaper}
                    onBack={()=>setActivePaper(null)}
                  />
                )}
                {panelTab==='trace' && (
                  <TracePanel meta={lastMetaEntry} />
                )}
                {panelTab==='causal' && (
                  <CausalPanel data={causalData} loading={causalLoading} />
                )}
                {panelTab==='optimization' && (
                  <OptimizationPanel data={optData} polling={optPolling} />
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      <style>{`
        .input-wrap:focus-within{border-color:var(--bark)!important;box-shadow:0 0 0 3px rgba(138,118,80,.1)!important;}
        .tb-btn:hover{background:var(--border)!important;color:var(--ink)!important;}
        @keyframes msgIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
        @keyframes typingBounce{0%,80%,100%{transform:scale(0)}40%{transform:scale(1)}}
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes pulse2{0%,100%{opacity:.4}50%{opacity:1}}
      `}</style>
    </div>
  )
}

// ─── Message ─────────────────────────────────────────────────────────────────
function Message({ msg, initials, modelName, meta, feedbackState, onFeedback, onShowTrace }) {
  const isUser = msg.role === 'user'
  const [showMeta, setShowMeta] = useState(false)

  // For user messages: plain inline HTML (no XML tags)
  // For AI messages: use the XML-aware parser
  const userFormatted = isUser
    ? (msg.text || '')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br/>')
    : null

  return (
    <div style={{...css.msg, flexDirection:isUser?'row-reverse':'row'}}>
      <div style={{
        ...css.avatar,
        background: isUser ? 'linear-gradient(135deg,var(--bark),var(--sage))' : 'var(--parchment)',
        border: isUser ? 'none' : '1px solid var(--border)',
        color: isUser ? 'var(--parchment-light)' : 'var(--bark)',
      }}>
        {isUser ? initials.toUpperCase() : 'IX'}
      </div>

      <div style={{maxWidth:'72%'}}>
        <div style={{
          ...css.bubble,
          background: isUser ? 'var(--bark-deeper)' : 'white',
          color: isUser ? 'var(--parchment-light)' : 'var(--ink)',
          border: isUser ? 'none' : '1px solid var(--border)',
          borderBottomRightRadius: isUser ? 4 : 14,
          borderBottomLeftRadius:  isUser ? 14 : 4,
          // AI bubbles need a bit more padding for the structured sections
          padding: isUser ? '0.95rem 1.1rem' : '1rem 1.1rem',
        }}>
          {isUser
            ? <span dangerouslySetInnerHTML={{ __html: userFormatted }} />
            : parseXmlResponse(msg.text)
          }

          {/* Sources */}
          {msg.sources?.length > 0 && (
            <div style={css.sources}>
              {msg.sources.map((s,i) => <span key={i} style={css.sourceChip}>📄 {s}</span>)}
            </div>
          )}

          {/* Confidence bar */}
          {msg.confidence != null && !isUser && (
            <div style={css.confBar}>
              <span>Confidence</span>
              <div style={{flex:1,height:3,background:'var(--border)',borderRadius:2,overflow:'hidden'}}>
                <div style={{height:'100%',width:`${msg.confidence}%`,background:'linear-gradient(90deg,var(--sage),var(--bark))',borderRadius:2}}/>
              </div>
              <span>{msg.confidence}%</span>
            </div>
          )}
        </div>

        {/* Message footer */}
        <div style={{...css.msgMeta, justifyContent:isUser?'flex-end':'flex-start'}}>
          <span>{isUser?'You':`IXORA · ${modelName}`} · {new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}</span>

          {!isUser && meta && (
            <>
              {meta.intent && <Tag>{meta.intent}</Tag>}
              {meta.usedPipeline && <Tag green>pipeline</Tag>}
              {meta.processingTime && <span style={{opacity:.6}}>{meta.processingTime}s</span>}
              {meta.trace?.length>0 && (
                <button style={css.metaBtn} onClick={()=>{setShowMeta(v=>!v);onShowTrace()}}>
                  {showMeta?'hide trace':'show trace ↗'}
                </button>
              )}
            </>
          )}

          {!isUser && (
            <div style={{display:'flex',gap:3,marginLeft:'auto'}}>
              <button style={{...css.fbBtn, color: feedbackState==='good'?'var(--sage)':undefined}} onClick={()=>onFeedback('good')}>👍</button>
              <button style={{...css.fbBtn, color: feedbackState==='bad' ?'#B5614A':undefined}} onClick={()=>onFeedback('bad')}>👎</button>
            </div>
          )}
        </div>

        {/* Inline mini-trace */}
        {!isUser && meta?.trace?.length>0 && showMeta && (
          <div style={css.inlineTrace}>
            {meta.trace.map((step,i)=>(
              <div key={i} style={css.traceStepMini}>
                <span style={css.traceNum}>{i+1}</span>
                <span style={{fontFamily:'var(--font-mono)',fontSize:'0.6rem',color:'var(--muted)'}}>
                  {step.step || step.agent || 'step'}
                </span>
                <span style={{marginLeft:'auto',fontFamily:'var(--font-mono)',fontSize:'0.58rem',color:'var(--bark-dark)',opacity:.7}}>
                  {step.duration!=null ? `${step.duration}s` : ''}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function Tag({children,green}) {
  return (
    <span style={{fontFamily:'var(--font-mono)',fontSize:'0.52rem',padding:'2px 5px',borderRadius:4,
      background:green?'rgba(107,143,113,.15)':'rgba(138,118,80,.1)',
      color:green?'#6B8F71':'var(--bark-dark)',border:`1px solid ${green?'rgba(107,143,113,.2)':'rgba(138,118,80,.15)'}`}}>
      {children}
    </span>
  )
}

// ─── Typing indicator ────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div style={{display:'flex',gap:'0.75rem',marginBottom:'1.5rem'}}>
      <div style={{...css.avatar,background:'var(--parchment)',border:'1px solid var(--border)',color:'var(--bark)'}}>IX</div>
      <div style={{...css.bubble,background:'white',border:'1px solid var(--border)',borderBottomLeftRadius:4}}>
        <div style={{display:'flex',gap:4,padding:'0.2rem 0',alignItems:'center'}}>
          {[0,0.2,0.4].map((d,i)=>(
            <div key={i} style={{width:6,height:6,background:'var(--muted-light)',borderRadius:'50%',animation:`typingBounce 1.2s ${d}s ease infinite`}}/>
          ))}
        </div>
      </div>
    </div>
  )
}

// ─── Papers panel ────────────────────────────────────────────────────────────
function PapersPanel({ papers, loading, activePaper, onSelect, onBack }) {
  if (activePaper) {
    return (
      <div style={{ display:'flex', flexDirection:'column', height:'100%' }}>
        <div style={{ display:'flex', alignItems:'center', gap:'0.5rem', padding:'0.5rem 0 0.75rem', borderBottom:'1px solid var(--border)', marginBottom:'0.75rem', flexShrink:0 }}>
          <button style={css.panelClose} onClick={onBack}>← Back to list</button>
          <span style={{ fontSize:'0.75rem', fontWeight:600, color:'var(--ink)', flex:1, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{activePaper.title}</span>
        </div>
        <div style={{ display:'flex', gap:'0.75rem', flex:1, overflow:'hidden', minHeight:0 }}>
          <div style={{ width:200, flexShrink:0, overflowY:'auto', display:'flex', flexDirection:'column', gap:'0.5rem', paddingRight:'0.5rem', borderRight:'1px solid var(--border)' }}>
            <div style={{ fontFamily:'var(--font-mono)', fontSize:'0.6rem', letterSpacing:'0.12em', textTransform:'uppercase', color:'var(--muted)', marginBottom:'0.25rem' }}>All Papers</div>
            {papers.map((p, i) => (
              <div
                key={i}
                style={{
                  padding:'0.5rem 0.6rem', borderRadius:7, cursor:'pointer', transition:'all .18s',
                  border:`1px solid ${activePaper===p ? 'var(--bark)' : 'var(--border)'}`,
                  background: activePaper===p ? 'rgba(138,118,80,.08)' : 'var(--parchment-light)',
                }}
                onClick={() => onSelect({...p, pdfUrl: p.pdf_url})}
              >
                <div style={{ fontSize:'0.72rem', fontWeight:600, color:'var(--ink)', lineHeight:1.35, marginBottom:'0.2rem' }}>{p.title || p.id}</div>
                {p.year && <div style={{ fontFamily:'var(--font-mono)', fontSize:'0.58rem', color:'var(--bark)' }}>{p.year}</div>}
                {p.authors && <div style={{ fontSize:'0.62rem', color:'var(--muted)', marginTop:2, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{p.authors}</div>}
              </div>
            ))}
          </div>
          <div style={{ flex:1, display:'flex', flexDirection:'column', overflow:'hidden' }}>
            <div style={{ marginBottom:'0.6rem', flexShrink:0 }}>
              {activePaper.authors && <div style={{ fontSize:'0.68rem', color:'var(--muted)', marginBottom:'0.2rem' }}>{activePaper.authors}</div>}
              {activePaper.summary && <div style={{ fontSize:'0.7rem', color:'var(--ink-mid)', lineHeight:1.55, marginBottom:'0.5rem' }}>{activePaper.summary}</div>}
              <div style={{ display:'flex', gap:'0.4rem' }}>
                {activePaper.url && <a href={activePaper.url} target="_blank" rel="noreferrer" style={css.paperBtn}>Open on arXiv ↗</a>}
              </div>
            </div>
            {activePaper.pdfUrl ? (
              <iframe
                src={activePaper.pdfUrl}
                style={{ flex:1, border:'1px solid var(--border)', borderRadius:8, minHeight:0 }}
                title={activePaper.title}
              />
            ) : (
              <div style={{ flex:1, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', border:'1px dashed var(--border)', borderRadius:8, gap:'0.5rem' }}>
                <div style={{ fontSize:'2rem', opacity:.3 }}>📄</div>
                <div style={{ fontFamily:'var(--font-mono)', fontSize:'0.68rem', color:'var(--muted)' }}>No PDF available — view on arXiv</div>
                {activePaper.url && (
                  <a href={activePaper.url} target="_blank" rel="noreferrer" style={css.paperBtn}>Open on arXiv ↗</a>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div>
      <div style={css.panelSectionTitle}>Research Papers · arXiv</div>
      {loading && <Spinner label="Fetching papers…"/>}
      {!loading && papers.length===0 && (
        <Empty icon="📚" text="Send a message to fetch relevant papers"/>
      )}
      {papers.map((p,i)=>(
        <div key={i} style={css.paperCard} className="paper-card">
          <div style={{fontSize:'0.8rem',fontWeight:600,color:'var(--ink)',lineHeight:1.4,marginBottom:'0.3rem'}}>{p.title||p.id}</div>
          {p.authors && <div style={{fontSize:'0.7rem',color:'var(--muted)',marginBottom:'0.25rem'}}>{p.authors}</div>}
          {p.year   && <div style={{fontFamily:'var(--font-mono)',fontSize:'0.65rem',color:'var(--bark)',marginBottom:'0.5rem'}}>{p.year}{p.journal?` · ${p.journal}`:''}</div>}
          {p.summary && <div style={{fontSize:'0.72rem',color:'var(--muted)',lineHeight:1.6,marginBottom:'0.6rem'}}>{p.summary.slice(0,200)}…</div>}
          <div style={{display:'flex',gap:'0.4rem',flexWrap:'wrap'}}>
            {p.url && <a href={p.url} target="_blank" rel="noreferrer" style={css.paperBtn}>arXiv →</a>}
            {p.pdf_url && <button style={css.paperBtn} onClick={()=>onSelect({...p, pdfUrl:p.pdf_url})}>Open PDF</button>}
            <button style={css.paperBtn} onClick={()=>{navigator.clipboard.writeText(p.title||'');toast('Title copied')}}>Cite</button>
          </div>
        </div>
      ))}
      <style>{`.paper-card:hover{border-color:var(--bark)!important;background:var(--parchment)!important;}`}</style>
    </div>
  )
}

// ─── Trace panel ─────────────────────────────────────────────────────────────
function TracePanel({ meta }) {
  if (!meta) return <Empty icon="🔍" text="Send a research query to see the reasoning trace"/>

  const { trace=[], parameters={}, intent, domain, processingTime, rewardScore, usedPipeline } = meta

  return (
    <div>
      <div style={css.panelSectionTitle}>Reasoning Trace</div>
      <div style={css.traceMetaRow}>
        {intent      && <MetaChip label="Intent"   value={intent}/>}
        {domain      && <MetaChip label="Domain"   value={domain}/>}
        {processingTime && <MetaChip label="Time"  value={`${processingTime}s`}/>}
        {rewardScore!=null && <MetaChip label="Reward" value={rewardScore.toFixed(3)}/>}
        <MetaChip label="Pipeline" value={usedPipeline ? 'Full' : 'Fast'}/>
      </div>

      {trace.length>0 ? (
        <>
          <div style={{fontFamily:'var(--font-mono)',fontSize:'0.58rem',letterSpacing:'0.12em',textTransform:'uppercase',color:'var(--muted-light)',marginBottom:'0.5rem'}}>
            {trace.length} steps
          </div>
          {trace.map((step,i)=>(
            <TraceStep key={i} step={step} index={i}/>
          ))}
        </>
      ) : (
        <Empty icon="📋" text="No detailed trace available for this response (fast path used)"/>
      )}

      {Object.keys(parameters).length>0 && (
        <>
          <div style={{...css.panelSectionTitle,marginTop:'1.25rem'}}>Extracted Parameters</div>
          {Object.entries(parameters).map(([k,v])=>(
            <div key={k} style={css.paramRow}>
              <span style={css.paramKey}>{k.replace(/_/g,' ')}</span>
              <span style={css.paramVal}>{typeof v==='object' ? (v.value??JSON.stringify(v)) : v}</span>
              {typeof v==='object' && v.unit && <span style={css.paramUnit}>{v.unit}</span>}
              {typeof v==='object' && v.confidence && (
                <span style={css.paramConf}>{Math.round(v.confidence*100)}%</span>
              )}
            </div>
          ))}
        </>
      )}
    </div>
  )
}

function TraceStep({ step, index }) {
  const [open, setOpen] = useState(false)
  const name    = step.step || step.agent || step.name || `Step ${index+1}`
  const summary = step.summary || step.result || step.output || ''
  const hasMeta = step.duration!=null || step.confidence!=null || step.method

  return (
    <div style={css.traceStep}>
      <div style={css.traceStepHeader} onClick={()=>setOpen(v=>!v)}>
        <div style={css.traceIdx}>{index+1}</div>
        <div style={{flex:1}}>
          <div style={{fontSize:'0.75rem',fontWeight:600,color:'var(--ink)',textTransform:'capitalize'}}>{name}</div>
          {summary && !open && <div style={{fontSize:'0.65rem',color:'var(--muted)',marginTop:2,lineHeight:1.4}}>{String(summary).slice(0,80)}{String(summary).length>80?'…':''}</div>}
        </div>
        {hasMeta && (
          <div style={{display:'flex',gap:'0.4rem',alignItems:'center',flexShrink:0}}>
            {step.duration!=null && <span style={css.traceBadge}>{step.duration}s</span>}
            {step.method          && <span style={css.traceBadge}>{step.method}</span>}
            {step.confidence!=null && <span style={{...css.traceBadge,background:'rgba(107,143,113,.15)',color:'#6B8F71'}}>{Math.round(step.confidence*100)}%</span>}
          </div>
        )}
        <span style={{fontSize:'0.6rem',color:'var(--muted-light)',marginLeft:'0.4rem'}}>{open?'▲':'▼'}</span>
      </div>
      {open && (
        <div style={css.traceStepBody}>
          {summary && <div style={{fontSize:'0.72rem',color:'var(--ink-mid)',lineHeight:1.65,marginBottom:'0.5rem'}}>{String(summary)}</div>}
          {Object.entries(step).filter(([k])=>!['step','agent','name','summary','result','output','duration','confidence','method'].includes(k)).map(([k,v])=>(
            <div key={k} style={{display:'flex',gap:'0.5rem',marginBottom:'0.25rem'}}>
              <span style={{fontFamily:'var(--font-mono)',fontSize:'0.6rem',color:'var(--muted)',minWidth:100,flexShrink:0}}>{k}</span>
              <span style={{fontFamily:'var(--font-mono)',fontSize:'0.6rem',color:'var(--ink-mid)',wordBreak:'break-all'}}>{typeof v==='object'?JSON.stringify(v,null,1):String(v)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Causal analysis panel ───────────────────────────────────────────────────
function CausalPanel({ data, loading }) {
  if (loading) return <Spinner label="Running causal analysis…"/>
  if (!data)   return <Empty icon="🔬" text="Click 'Causal' in the toolbar to run analysis"/>
  if (data.error) return (
    <div style={{padding:'1rem',background:'rgba(181,97,74,.08)',borderRadius:8,border:'1px solid rgba(181,97,74,.2)'}}>
      <div style={{fontSize:'0.72rem',color:'var(--error)'}}>{data.error}</div>
    </div>
  )

  const cr = data.causal_results || {}
  const links = data.arxiv_links || []
  const paramsAnalyzed = data.parameters_analyzed || []

  return (
    <div>
      <div style={css.panelSectionTitle}>Causal Analysis</div>

      {paramsAnalyzed.length>0 && (
        <div style={{display:'flex',flexWrap:'wrap',gap:4,marginBottom:'1rem'}}>
          {paramsAnalyzed.map(p=>(
            <span key={p} style={{fontFamily:'var(--font-mono)',fontSize:'0.58rem',padding:'2px 6px',borderRadius:4,background:'rgba(138,118,80,.1)',color:'var(--bark)'}}>{p}</span>
          ))}
        </div>
      )}

      {cr.causal_effects && Object.keys(cr.causal_effects).length>0 && (
        <>
          <div style={css.subHeading}>Causal Effects</div>
          {Object.entries(cr.causal_effects).map(([k,v])=>(
            <div key={k} style={css.causalRow}>
              <div style={{fontSize:'0.72rem',fontWeight:600,color:'var(--ink)',marginBottom:2}}>{k.replace(/_/g,' ')}</div>
              <div style={{fontSize:'0.68rem',color:'var(--muted)',lineHeight:1.5}}>{typeof v==='object'?JSON.stringify(v,null,2):String(v)}</div>
            </div>
          ))}
        </>
      )}

      {cr.statistics && (
        <>
          <div style={css.subHeading}>Statistics</div>
          <div style={css.codeBlock}>{JSON.stringify(cr.statistics,null,2)}</div>
        </>
      )}

      {!cr.causal_effects && !cr.statistics && Object.keys(cr).length>0 && (
        <>
          <div style={css.subHeading}>Analysis Result</div>
          <div style={css.codeBlock}>{JSON.stringify(cr,null,2)}</div>
        </>
      )}

      {links.length>0 && (
        <>
          <div style={{...css.subHeading,marginTop:'1.25rem'}}>Related Papers</div>
          {links.map((p,i)=>(
            <div key={i} style={{marginBottom:'0.5rem'}}>
              <a href={p.url||p.arxiv_url||'#'} target="_blank" rel="noreferrer"
                 style={{fontSize:'0.7rem',color:'var(--bark)',textDecoration:'underline',lineHeight:1.4}}>
                {p.title||p.id}
              </a>
            </div>
          ))}
        </>
      )}
    </div>
  )
}

// ─── Optimisation panel ──────────────────────────────────────────────────────
function OptimizationPanel({ data, polling }) {
  if (polling && !data) return <Spinner label="Waiting for Bayesian optimisation…"/>
  if (!data) return <Empty icon="⚙️" text="Optimisation runs automatically after a research query with numeric parameters"/>

  const status = data.status || 'unknown'
  const result = data.result || {}
  const optimal = result.optimal_parameters || result.best_parameters || {}
  const statusColor = status==='completed'?'#6B8F71':status==='failed'||status==='timeout'?'#B5614A':'var(--bark)'

  return (
    <div>
      <div style={css.panelSectionTitle}>Bayesian Optimisation</div>

      <div style={{display:'flex',alignItems:'center',gap:'0.5rem',marginBottom:'1rem'}}>
        <div style={{width:8,height:8,borderRadius:'50%',background:statusColor,flexShrink:0,
          animation: polling?'pulse2 1.5s infinite':undefined}}/>
        <span style={{fontFamily:'var(--font-mono)',fontSize:'0.7rem',fontWeight:600,color:statusColor,textTransform:'uppercase'}}>
          {polling?'Running…':status}
        </span>
        {data.timestamp && <span style={{fontFamily:'var(--font-mono)',fontSize:'0.58rem',color:'var(--muted)',marginLeft:'auto'}}>{new Date(data.timestamp).toLocaleTimeString()}</span>}
      </div>

      {data.error && (
        <div style={{padding:'0.75rem',background:'rgba(181,97,74,.08)',borderRadius:8,border:'1px solid rgba(181,97,74,.15)',fontSize:'0.7rem',color:'var(--error)',marginBottom:'1rem'}}>
          {data.error}
        </div>
      )}

      {Object.keys(optimal).length>0 && (
        <>
          <div style={css.subHeading}>Optimal Parameters</div>
          {Object.entries(optimal).map(([k,v])=>(
            <div key={k} style={css.paramRow}>
              <span style={css.paramKey}>{k.replace(/_/g,' ')}</span>
              <span style={{...css.paramVal,color:'var(--sage)',fontWeight:700}}>{typeof v==='object'?JSON.stringify(v):String(v)}</span>
            </div>
          ))}
        </>
      )}

      {result.best_value!=null && (
        <div style={{marginTop:'0.75rem'}}>
          <div style={css.subHeading}>Best Score</div>
          <div style={{fontFamily:'var(--font-mono)',fontSize:'1.1rem',fontWeight:700,color:'var(--sage)'}}>{Number(result.best_value).toFixed(4)}</div>
        </div>
      )}

      {result.iterations!=null && (
        <div style={{fontFamily:'var(--font-mono)',fontSize:'0.62rem',color:'var(--muted)',marginTop:'0.5rem'}}>
          {result.iterations} iterations · {result.method||'Bayesian'}
        </div>
      )}

      {Object.keys(result).length>0 && (
        <>
          <div style={{...css.subHeading,marginTop:'1rem'}}>Full Result</div>
          <div style={css.codeBlock}>{JSON.stringify(result,null,2)}</div>
        </>
      )}
    </div>
  )
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
function MetaChip({label,value}) {
  return (
    <div style={{background:'rgba(138,118,80,.07)',border:'1px solid rgba(138,118,80,.12)',borderRadius:6,padding:'0.3rem 0.6rem',flexShrink:0}}>
      <div style={{fontFamily:'var(--font-mono)',fontSize:'0.5rem',color:'var(--muted-light)',marginBottom:1}}>{label}</div>
      <div style={{fontFamily:'var(--font-mono)',fontSize:'0.65rem',fontWeight:600,color:'var(--ink-mid)'}}>{value}</div>
    </div>
  )
}

function Spinner({label}) {
  return (
    <div style={{display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',padding:'2.5rem',gap:'0.75rem'}}>
      <div style={{width:22,height:22,border:'2px solid var(--border)',borderTopColor:'var(--bark)',borderRadius:'50%',animation:'spin .9s linear infinite'}}/>
      <div style={{fontFamily:'var(--font-mono)',fontSize:'0.65rem',color:'var(--muted)'}}>{label}</div>
    </div>
  )
}

function Empty({icon,text}) {
  return (
    <div style={{textAlign:'center',padding:'2.5rem 1rem'}}>
      <div style={{fontSize:'2rem',marginBottom:'0.75rem',opacity:.4}}>{icon}</div>
      <div style={{fontFamily:'var(--font-mono)',fontSize:'0.65rem',color:'var(--muted-light)',lineHeight:1.6}}>{text}</div>
    </div>
  )
}

// ─── Styles ──────────────────────────────────────────────────────────────────
const css = {
  wrap: { flex:1, display:'flex', flexDirection:'column', overflow:'hidden', background:'var(--parchment-light)' },
  topbar: { display:'flex', alignItems:'center', padding:'0.75rem 1.4rem', borderBottom:'1px solid var(--border)', background:'rgba(255,255,255,.6)', backdropFilter:'blur(12px)', gap:'0.6rem', flexShrink:0, flexWrap:'wrap' },
  backBtn: { background:'none', border:'none', cursor:'pointer', color:'var(--muted)', fontSize:'0.88rem', display:'flex', alignItems:'center', gap:'0.4rem', padding:'0.3rem 0.6rem', borderRadius:6, transition:'all .2s', fontFamily:'var(--font-sans)' },
  domainBadge: { display:'flex', alignItems:'center', gap:'0.4rem', background:'var(--parchment)', border:'1px solid var(--border)', borderRadius:20, padding:'0.28rem 0.7rem', fontSize:'0.76rem', fontWeight:600, color:'var(--bark-dark)', flexShrink:0 },
  titleInput: { flex:1, background:'none', border:'none', fontFamily:'var(--font-sans)', fontSize:'0.92rem', fontWeight:600, color:'var(--ink)', outline:'none', minWidth:80 },
  topbarActions: { display:'flex', alignItems:'center', gap:'0.35rem', flexWrap:'wrap' },
  topBtn: (active) => ({ background: active?'rgba(138,118,80,.1)':'none', border:`1px solid ${active?'var(--bark)':'var(--border)'}`, borderRadius:7, padding:'0.32rem 0.7rem', fontSize:'0.74rem', color:active?'var(--bark)':'var(--muted)', cursor:'pointer', transition:'all .2s', fontFamily:'var(--font-sans)', fontWeight:500, whiteSpace:'nowrap' }),

  body: { flex:1, display:'flex', overflow:'hidden' },
  messagesPane: { flex:1, display:'flex', flexDirection:'column', overflow:'hidden', minWidth:0 },
  messagesScroll: { flex:1, overflowY:'auto', padding:'2rem 2.5rem' },
  divider: { display:'flex', alignItems:'center', gap:'0.75rem', textAlign:'center', fontFamily:'var(--font-mono)', fontSize:'0.65rem', color:'var(--muted-light)', letterSpacing:'0.1em', margin:'0 0 1.5rem' },
  divLine: { flex:1, height:1, background:'var(--border)' },
  msg: { display:'flex', gap:'0.85rem', marginBottom:'1.8rem', animation:'msgIn .3s ease forwards' },
  avatar: { width:34, height:34, borderRadius:'50%', display:'flex', alignItems:'center', justifyContent:'center', fontSize:'0.68rem', fontWeight:700, flexShrink:0, marginTop:2 },
  bubble: { borderRadius:14, padding:'0.95rem 1.1rem', fontSize:'0.92rem', lineHeight:1.8 },
  sources: { display:'flex', flexWrap:'wrap', gap:'0.4rem', marginTop:'0.7rem' },
  sourceChip: { background:'var(--parchment)', border:'1px solid var(--border)', borderRadius:6, padding:'0.28rem 0.6rem', fontSize:'0.7rem', color:'var(--bark-dark)' },
  confBar: { display:'flex', alignItems:'center', gap:'0.5rem', marginTop:'0.6rem', fontFamily:'var(--font-mono)', fontSize:'0.65rem', color:'var(--muted)' },
  msgMeta: { fontFamily:'var(--font-mono)', fontSize:'0.62rem', color:'var(--muted-light)', marginTop:'0.35rem', display:'flex', alignItems:'center', gap:'0.4rem', flexWrap:'wrap' },
  metaBtn: { background:'none', border:'none', cursor:'pointer', fontFamily:'var(--font-mono)', fontSize:'0.62rem', color:'var(--bark)', padding:0 },
  fbBtn: { background:'none', border:'none', cursor:'pointer', fontSize:'0.82rem', padding:'1px 3px', borderRadius:4, transition:'all .15s', opacity:.5 },
  inlineTrace: { marginTop:'0.4rem', background:'rgba(138,118,80,.04)', border:'1px solid rgba(138,118,80,.1)', borderRadius:8, padding:'0.5rem', display:'flex', flexDirection:'column', gap:2 },
  traceStepMini: { display:'flex', alignItems:'center', gap:'0.5rem', padding:'0.25rem 0' },
  traceNum: { width:18, height:18, borderRadius:'50%', background:'rgba(138,118,80,.15)', display:'flex', alignItems:'center', justifyContent:'center', fontSize:'0.55rem', fontFamily:'var(--font-mono)', color:'var(--bark)', flexShrink:0 },

  inputArea: { padding:'0.9rem 2rem 1.1rem', background:'rgba(255,255,255,.6)', backdropFilter:'blur(12px)', borderTop:'1px solid var(--border)', flexShrink:0 },
  inputWrap: { background:'white', border:'1.5px solid var(--border)', borderRadius:14, display:'flex', alignItems:'flex-end', gap:'0.5rem', padding:'0.7rem 0.7rem 0.7rem 1.1rem', transition:'all .2s' },
  textarea: { flex:1, border:'none', outline:'none', fontFamily:'var(--font-sans)', fontSize:'0.92rem', color:'var(--ink)', background:'transparent', resize:'none', maxHeight:120, lineHeight:1.65 },
  sendBtn: { width:36, height:36, background:'var(--bark-deeper)', border:'none', borderRadius:8, color:'var(--parchment-light)', fontSize:'0.88rem', cursor:'pointer', transition:'all .2s', display:'flex', alignItems:'center', justifyContent:'center' },
  inputFooter: { marginTop:'0.4rem', display:'flex', alignItems:'center', gap:'1rem', fontFamily:'var(--font-mono)', fontSize:'0.63rem', color:'var(--muted-light)' },

  panel: { borderLeft:'1px solid var(--border)', overflow:'hidden', transition:'width .32s cubic-bezier(.4,0,.2,1)', flexShrink:0, background:'white' },
  panelInner: { width:'100%', height:'100%', display:'flex', flexDirection:'column', overflow:'hidden' },
  panelTabBar: { display:'flex', gap:2, padding:'0.5rem 0.75rem 0', borderBottom:'1px solid var(--border)', background:'var(--parchment)', flexShrink:0 },
  panelTabBtn: (active) => ({ flex:1, padding:'0.42rem 0', border:'none', borderRadius:'6px 6px 0 0', fontFamily:'var(--font-sans)', fontSize:'0.72rem', fontWeight:600, cursor:'pointer', transition:'all .2s', background:active?'white':'transparent', color:active?'var(--ink)':'var(--muted)', borderBottom:active?'none':`1px solid var(--border)` }),
  panelClose: { background:'none', border:'none', cursor:'pointer', color:'var(--muted)', fontSize:'0.85rem', padding:'3px 6px', borderRadius:5, transition:'all .2s', fontFamily:'var(--font-sans)', marginLeft:'auto' },
  panelScroll: { flex:1, overflowY:'auto', padding:'1rem', scrollbarWidth:'thin', scrollbarColor:'rgba(138,118,80,.2) transparent' },
  panelSectionTitle: { fontFamily:'var(--font-mono)', fontSize:'0.68rem', letterSpacing:'0.15em', textTransform:'uppercase', color:'var(--bark)', marginBottom:'0.75rem', paddingBottom:'0.4rem', borderBottom:'1px solid var(--border)' },
  subHeading: { fontFamily:'var(--font-mono)', fontSize:'0.64rem', letterSpacing:'0.1em', textTransform:'uppercase', color:'var(--muted)', marginBottom:'0.4rem', marginTop:'0.6rem' },

  traceMetaRow: { display:'flex', gap:'0.5rem', flexWrap:'wrap', marginBottom:'1rem' },
  traceStep: { border:'1px solid var(--border)', borderRadius:9, marginBottom:'0.5rem', overflow:'hidden' },
  traceStepHeader: { display:'flex', alignItems:'center', gap:'0.6rem', padding:'0.65rem 0.75rem', cursor:'pointer', background:'var(--parchment-light)', transition:'background .15s' },
  traceIdx: { width:24, height:24, borderRadius:'50%', background:'rgba(138,118,80,.15)', display:'flex', alignItems:'center', justifyContent:'center', fontSize:'0.65rem', fontFamily:'var(--font-mono)', color:'var(--bark)', fontWeight:700, flexShrink:0 },
  traceBadge: { fontFamily:'var(--font-mono)', fontSize:'0.6rem', padding:'2px 6px', borderRadius:4, background:'rgba(138,118,80,.1)', color:'var(--bark-dark)' },
  traceStepBody: { padding:'0.65rem 0.75rem 0.8rem', borderTop:'1px solid var(--border)', background:'white' },

  paramRow: { display:'flex', alignItems:'center', gap:'0.5rem', padding:'0.45rem 0', borderBottom:'1px solid rgba(138,118,80,.07)', flexWrap:'wrap' },
  paramKey: { fontFamily:'var(--font-mono)', fontSize:'0.68rem', color:'var(--muted)', minWidth:110, flexShrink:0 },
  paramVal: { fontFamily:'var(--font-mono)', fontSize:'0.74rem', fontWeight:600, color:'var(--ink-mid)' },
  paramUnit: { fontFamily:'var(--font-mono)', fontSize:'0.63rem', color:'var(--bark)', opacity:.8 },
  paramConf: { fontFamily:'var(--font-mono)', fontSize:'0.6rem', padding:'1px 4px', borderRadius:3, background:'rgba(107,143,113,.12)', color:'#6B8F71', marginLeft:'auto' },

  causalRow: { background:'var(--parchment-light)', border:'1px solid var(--border)', borderRadius:8, padding:'0.65rem', marginBottom:'0.5rem' },

  codeBlock: { fontFamily:'var(--font-mono)', fontSize:'0.65rem', lineHeight:1.65, color:'var(--ink-mid)', background:'rgba(138,118,80,.04)', border:'1px solid rgba(138,118,80,.1)', borderRadius:8, padding:'0.75rem', whiteSpace:'pre-wrap', wordBreak:'break-all', maxHeight:280, overflowY:'auto' },

  paperCard: { background:'var(--parchment-light)', border:'1px solid var(--border)', borderRadius:9, padding:'0.85rem', marginBottom:'0.6rem', transition:'all .2s' },
  paperBtn: { fontSize:'0.67rem', padding:'4px 10px', borderRadius:4, border:'1px solid var(--border)', background:'white', color:'var(--bark-dark)', cursor:'pointer', transition:'all .2s', fontFamily:'var(--font-mono)', textDecoration:'none' },
}