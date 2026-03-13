import { useState, useRef, useEffect } from 'react'
import { useAuth } from '../context/AuthContext'
import { useChat } from '../context/ChatContext'
import { toast } from './Toast'

const DOMAIN_CFG = {
  bio: { label:'Biomedical',    model:'BioGPT',        color:'#6B8F71' },
  cs:  { label:'Comp. Science', model:'Qwen2.5-Coder', color:'#8A7650' },
  gen: { label:'General',       model:'Mistral-Large', color:'#8E977D' },
}

const DEMO_PAPERS = [
  { id:'p1', title:'CRISPR-Cas9 Mechanism in Mammalian Genome Editing', authors:'Zhang et al. · 2024 · Nature Biotech', url:'' },
  { id:'p2', title:'Bayesian Optimization Strategies for Accelerated Drug Discovery', authors:'Liu & Park · 2024 · Cell Systems', url:'' },
  { id:'p3', title:'Causal Inference Methods for Observational Clinical Trial Data', authors:'Hernán et al. · 2023 · NEJM', url:'' },
  { id:'p4', title:'Attention Is All You Need: Revisited for Scientific Language Models', authors:'Vaswani et al. · 2023 · arXiv', url:'' },
]

const AI_RESPONSES = {
  bio: [
    `Based on the **IMRAD framework**, here is my structured analysis:\n\n**Introduction:** Your query touches a well-studied area with several open questions. The causal mechanism involves multiple confounding variables requiring careful isolation.\n\n**Recommended Methods:** DoWhy-based causal analysis + Bayesian optimization of key parameters (pH: 7.0, temp: 37°C, nutrient concentration).\n\n**Hypothesis (BioGPT):** The proposed mechanism is consistent with oxidative stress pathways documented in 12 peer-reviewed sources.\n\nConfidence: 91% across retrieved literature.`,
    `Excellent research question. Let me trace my reasoning:\n\n**Step 1 — Parameter extraction** identified 3 key experimental variables\n**Step 2 — Literature retrieval** pulled 7 relevant papers from PubMed and bioRxiv\n**Step 3 — BioGPT hypothesis generation** with SHAP feature importance scores\n\nThe evidence strongly suggests a significant correlation (p < 0.001). I recommend a controlled trial with n ≥ 30 to confirm causal direction.`,
  ],
  cs: [
    `**Architecture comparison complete.** After benchmarking across 3 datasets:\n\nThe transformer with sparse attention achieves **23% lower latency** at comparable accuracy.\n\n**Bayesian-optimized hyperparameters:**\n- Learning rate: 3e-4\n- Batch size: 128\n- Warmup steps: 1000\n- Weight decay: 0.01\n\nSHAP analysis identifies attention head count as most impactful (importance score: 0.43). Full benchmark results surfaced in arXiv panel.`,
    `Code analysis complete.\n\n**Complexity:** O(n log n) average, O(n²) worst case\n**Memory footprint:** O(n) auxiliary space\n**Bottleneck identified:** Inner loop — consider memoization to reduce redundant computations\n\nI also found 2 edge cases your current implementation doesn't handle. Would you like me to generate unit tests covering those paths?`,
  ],
  gen: [
    `Great question spanning multiple fields. Here's a cross-domain synthesis:\n\nCurrent scientific consensus (2024) supports three competing explanatory frameworks, each with strong empirical backing. Recent work by Park et al. (2024) reconciles these using a unified **information-theoretic model** — I've surfaced this paper in the panel.\n\nKey insight: the apparent contradiction between Framework A and B dissolves when you account for measurement scale differences. Would you like me to dig into any specific framework?`,
  ],
}

export default function ChatView({ chatId, domain, onBack }) {
  const { user } = useAuth()
  const { history, appendMessage, updateChat, toggleBookmark } = useChat()

  const chat = history.find(c => c.id === chatId)
  const [messages, setMessages] = useState(chat?.msgs || [])
  const [input, setInput]       = useState('')
  const [loading, setLoading]   = useState(false)
  const [pdfOpen, setPdfOpen]   = useState(false)
  const [activePaper, setActivePaper] = useState(null)
  const [chatTitle, setChatTitle]     = useState(chat?.title || '')
  const scrollRef = useRef(null)
  const inputRef  = useRef(null)
  const cfg = DOMAIN_CFG[domain] || DOMAIN_CFG.gen
  const initials = user ? (user.first_name?.[0] || '') + (user.last_name?.[0] || '') : 'U'
  const bookmarked = chat?.bookmarked || false

  // Seed greeting if new chat
  useEffect(() => {
    if (!chat?.msgs?.length && messages.length === 0) {
      const greeting = {
        id: Date.now(),
        role: 'ai',
        text: `Hello! I'm IXORA's **${cfg.label}** specialist, powered by ${cfg.model}. Ask me anything — experimental design, literature review, hypothesis generation. Every reasoning step is traced and open to your inspection.`,
        sources: [],
      }
      setMessages([greeting])
    }
  }, [chatId])

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages, loading])

  const sendMessage = async () => {
    if (!input.trim() || loading) return
    const text = input.trim()
    setInput('')
    inputRef.current.style.height = 'auto'

    const userMsg = { id: Date.now(), role: 'user', text }
    setMessages(prev => [...prev, userMsg])

    // Update title if first real message
    if (!chatTitle) {
      const title = text.slice(0, 50)
      setChatTitle(title)
      updateChat(chatId, { title })
    }

    appendMessage(chatId, userMsg)
    setLoading(true)

    await new Promise(r => setTimeout(r, 1000 + Math.random() * 800))

    const pool = AI_RESPONSES[domain] || AI_RESPONSES.gen
    const aiText = pool[Math.floor(Math.random() * pool.length)]
    const sources = domain === 'bio'
      ? ['PubMed · 2024', 'bioRxiv · 2024', 'Nature · 2023']
      : domain === 'cs'
      ? ['arXiv · 2024', 'NeurIPS · 2023', 'ICML · 2024']
      : ['arXiv · 2024', 'Science · 2024']

    const aiMsg = { id: Date.now() + 1, role: 'ai', text: aiText, sources, confidence: 88 + Math.floor(Math.random() * 8) }
    setMessages(prev => [...prev, aiMsg])
    appendMessage(chatId, aiMsg)
    setLoading(false)
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }
  }

  const autoResize = (el) => {
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 120) + 'px'
  }

  const exportChat = () => {
    if (!messages.length) { toast('Nothing to export yet'); return }
    const text = messages.map(m => `[${m.role.toUpperCase()}]\n${m.text}`).join('\n\n---\n\n')
    const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(new Blob([text], {type:'text/plain'})), download: `${chatTitle || 'ixora-chat'}.txt` })
    a.click()
    toast('Exported as .txt')
  }

  const handleBookmark = () => {
    toggleBookmark(chatId)
    toast(bookmarked ? 'Bookmark removed' : '★ Chat bookmarked')
  }

  return (
    <div style={css.wrap}>
      {/* Topbar */}
      <div style={css.topbar}>
        <button style={css.backBtn} onClick={onBack}>← Back</button>
        <div style={css.domainBadge}>
          <span style={{ width:6, height:6, borderRadius:'50%', background:cfg.color, display:'block' }}/>
          {cfg.label}
        </div>
        <input
          style={css.titleInput}
          value={chatTitle}
          onChange={e => { setChatTitle(e.target.value); updateChat(chatId, { title: e.target.value }) }}
          placeholder="Untitled conversation…"
        />
        <div style={css.topbarActions}>
          <button style={css.topBtn(pdfOpen)} onClick={() => setPdfOpen(v => !v)}>📄 Papers</button>
          <button style={css.topBtn(bookmarked)} onClick={handleBookmark}>{bookmarked ? '★' : '☆'} Bookmark</button>
          <button style={css.topBtn(false)} onClick={exportChat}>↓ Export</button>
        </div>
      </div>

      {/* Body */}
      <div style={css.body}>
        {/* Messages */}
        <div style={css.messagesPane}>
          <div style={css.messagesScroll} ref={scrollRef}>
            <div style={css.divider}>Start of conversation</div>
            {messages.map(msg => (
              <Message key={msg.id} msg={msg} initials={initials} modelName={cfg.model} />
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
                onChange={e => { setInput(e.target.value); autoResize(e.target) }}
                onKeyDown={handleKey}
                rows={1}
              />
              <div style={{ display:'flex', alignItems:'center', gap:'0.4rem', flexShrink:0 }}>
                <button style={css.attachBtn} onClick={() => toast('File attachment coming soon')}>📎</button>
                <button style={css.sendBtn} onClick={sendMessage} disabled={loading || !input.trim()}>→</button>
              </div>
            </div>
            <div style={css.inputFooter}>
              <span>↵ send · shift+↵ newline</span>
              <button style={css.arxivBtn} onClick={() => setPdfOpen(v => !v)}>📄 arXiv papers</button>
            </div>
          </div>
        </div>

        {/* PDF Panel */}
        <div style={{ ...css.pdfPanel, width: pdfOpen ? '44%' : 0 }}>
          {pdfOpen && (
            activePaper
              ? <PdfEmbed paper={activePaper} onBack={() => setActivePaper(null)} onClose={() => { setPdfOpen(false); setActivePaper(null) }} />
              : <PapersList papers={DEMO_PAPERS} onSelect={p => setActivePaper(p)} onClose={() => setPdfOpen(false)} />
          )}
        </div>
      </div>

      <style>{`
        .input-wrap:focus-within{border-color:var(--bark)!important;box-shadow:0 0 0 3px rgba(138,118,80,.1)!important;}
        .back-btn:hover{background:var(--border)!important;color:var(--ink)!important;}
      `}</style>
    </div>
  )
}

function Message({ msg, initials, modelName }) {
  const isUser = msg.role === 'user'
  const formatted = (msg.text || '')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>')

  return (
    <div style={{ ...css.msg, flexDirection: isUser ? 'row-reverse' : 'row' }}>
      <div style={{ ...css.avatar, background: isUser ? 'linear-gradient(135deg,var(--bark),var(--sage))' : 'var(--parchment)', border: isUser ? 'none' : '1px solid var(--border)', color: isUser ? 'var(--parchment-light)' : 'var(--bark)' }}>
        {isUser ? initials.toUpperCase() : 'IX'}
      </div>
      <div>
        <div style={{ ...css.bubble, background: isUser ? 'var(--bark-deeper)' : 'white', color: isUser ? 'var(--parchment-light)' : 'var(--ink)', border: isUser ? 'none' : '1px solid var(--border)', borderBottomRightRadius: isUser ? 4 : 14, borderBottomLeftRadius: isUser ? 14 : 4 }}>
          <span dangerouslySetInnerHTML={{ __html: formatted }} />
          {msg.sources?.length > 0 && (
            <div style={css.sources}>
              {msg.sources.map((s, i) => (
                <span key={i} style={css.sourceChip}>📄 {s}</span>
              ))}
            </div>
          )}
          {msg.confidence && !isUser && (
            <div style={css.confBar}>
              <span>Confidence</span>
              <div style={{ flex:1, height:3, background:'var(--border)', borderRadius:2, overflow:'hidden' }}>
                <div style={{ height:'100%', width:`${msg.confidence}%`, background:'linear-gradient(90deg,var(--sage),var(--bark))', borderRadius:2 }}/>
              </div>
              <span>{msg.confidence}%</span>
            </div>
          )}
        </div>
        <div style={{ ...css.msgMeta, justifyContent: isUser ? 'flex-end' : 'flex-start' }}>
          {isUser ? 'You' : `IXORA · ${modelName}`} · {new Date().toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' })}
        </div>
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div style={{ display:'flex', gap:'0.75rem', marginBottom:'1.5rem' }}>
      <div style={{ ...css.avatar, background:'var(--parchment)', border:'1px solid var(--border)', color:'var(--bark)' }}>IX</div>
      <div style={{ ...css.bubble, background:'white', border:'1px solid var(--border)', borderBottomLeftRadius:4 }}>
        <div style={{ display:'flex', gap:4, padding:'0.2rem 0', alignItems:'center' }}>
          {[0,0.2,0.4].map((d,i) => (
            <div key={i} style={{ width:6, height:6, background:'var(--muted-light)', borderRadius:'50%', animation:`typingBounce 1.2s ${d}s ease infinite` }}/>
          ))}
        </div>
      </div>
    </div>
  )
}

function PapersList({ papers, onSelect, onClose }) {
  return (
    <>
      <div style={css.panelTopbar}>
        <span>📑</span>
        <span style={{ flex:1, fontSize:'0.78rem', fontWeight:600, color:'var(--ink)' }}>Research Papers</span>
        <button style={css.closeBtn} onClick={onClose}>✕</button>
      </div>
      <div style={{ flex:1, overflowY:'auto', padding:'0.75rem', display:'flex', flexDirection:'column', gap:'0.5rem' }}>
        {papers.map(p => (
          <div key={p.id} style={css.paperItem} className="paper-item">
            <div style={{ fontSize:'0.72rem', fontWeight:600, color:'var(--ink)', lineHeight:1.4, marginBottom:'0.3rem' }}>{p.title}</div>
            <div style={{ fontSize:'0.63rem', color:'var(--muted)', marginBottom:'0.4rem' }}>{p.authors}</div>
            <div style={{ display:'flex', gap:'0.4rem' }}>
              <button style={css.piBtn} onClick={() => onSelect(p)}>Open PDF</button>
              <button style={css.piBtn} onClick={() => toast('Citation copied')}>Cite</button>
            </div>
          </div>
        ))}
      </div>
      <style>{`.paper-item:hover{border-color:var(--bark)!important;background:var(--parchment)!important;}`}</style>
    </>
  )
}

function PdfEmbed({ paper, onBack, onClose }) {
  const [zoom, setZoom] = useState(1)
  // Use a placeholder PDF since we don't have real URLs
  const src = paper.url || 'about:blank'
  return (
    <>
      <div style={css.panelTopbar}>
        <button style={css.closeBtn} onClick={onBack}>←</button>
        <span style={{ flex:1, fontSize:'0.72rem', fontWeight:600, color:'var(--ink)', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{paper.title}</span>
        <button style={css.closeBtn} onClick={onClose}>✕</button>
      </div>
      <div style={{ display:'flex', alignItems:'center', gap:'0.5rem', padding:'0.5rem 1rem', borderBottom:'1px solid var(--border)', background:'white', flexShrink:0 }}>
        <button style={css.piBtn} onClick={() => setZoom(z => Math.max(0.5, z - 0.2))}>−</button>
        <span style={{ fontFamily:'var(--font-mono)', fontSize:'0.62rem', color:'var(--muted)', minWidth:40, textAlign:'center' }}>{Math.round(zoom*100)}%</span>
        <button style={css.piBtn} onClick={() => setZoom(z => Math.min(2, z + 0.2))}>+</button>
      </div>
      <div style={{ flex:1, overflow:'hidden', background:'#f5f5f5', display:'flex', alignItems:'center', justifyContent:'center' }}>
        {src === 'about:blank' ? (
          <div style={{ textAlign:'center', color:'var(--muted-light)', fontFamily:'var(--font-mono)', fontSize:'0.7rem' }}>
            <div style={{ fontSize:'2.5rem', marginBottom:'1rem', opacity:0.4 }}>📄</div>
            <div>PDF viewer</div>
            <div style={{ marginTop:'0.4rem', opacity:0.6 }}>Connect arXiv integration<br/>to load papers inline</div>
          </div>
        ) : (
          <iframe src={src} style={{ width:`${100/zoom}%`, height:`${100/zoom}%`, border:'none', transform:`scale(${zoom})`, transformOrigin:'top left' }} title={paper.title}/>
        )}
      </div>
    </>
  )
}

const css = {
  wrap: { flex:1, display:'flex', flexDirection:'column', overflow:'hidden', background:'var(--parchment-light)' },
  topbar: { display:'flex', alignItems:'center', padding:'0.8rem 1.5rem', borderBottom:'1px solid var(--border)', background:'rgba(255,255,255,.6)', backdropFilter:'blur(12px)', gap:'0.8rem', flexShrink:0 },
  backBtn: { background:'none', border:'none', cursor:'none', color:'var(--muted)', fontSize:'0.8rem', display:'flex', alignItems:'center', gap:'0.4rem', padding:'0.3rem 0.6rem', borderRadius:6, transition:'all .2s', fontFamily:'var(--font-sans)' },
  domainBadge: { display:'flex', alignItems:'center', gap:'0.4rem', background:'var(--parchment)', border:'1px solid var(--border)', borderRadius:20, padding:'0.28rem 0.7rem', fontSize:'0.7rem', fontWeight:600, color:'var(--bark-dark)', flexShrink:0 },
  titleInput: { flex:1, background:'none', border:'none', fontFamily:'var(--font-sans)', fontSize:'0.85rem', fontWeight:600, color:'var(--ink)', outline:'none', minWidth:0 },
  topbarActions: { display:'flex', alignItems:'center', gap:'0.4rem', marginLeft:'auto' },
  topBtn: (active) => ({ background: active ? 'rgba(138,118,80,.08)' : 'none', border:`1px solid ${active ? 'var(--bark)' : 'var(--border)'}`, borderRadius:7, padding:'0.32rem 0.65rem', fontSize:'0.7rem', color: active ? 'var(--bark)' : 'var(--muted)', cursor:'none', transition:'all .2s', display:'flex', alignItems:'center', gap:'0.3rem', fontFamily:'var(--font-sans)', fontWeight:500 }),
  body: { flex:1, display:'flex', overflow:'hidden' },
  messagesPane: { flex:1, display:'flex', flexDirection:'column', overflow:'hidden', minWidth:0 },
  messagesScroll: { flex:1, overflowY:'auto', padding:'2rem 2.5rem' },
  divider: { textAlign:'center', fontFamily:'var(--font-mono)', fontSize:'0.58rem', color:'var(--muted-light)', letterSpacing:'0.1em', margin:'0 0 1.5rem', display:'flex', alignItems:'center', gap:'0.75rem' },
  msg: { display:'flex', gap:'0.75rem', marginBottom:'1.6rem', animation:'msgIn .3s ease forwards' },
  avatar: { width:32, height:32, borderRadius:'50%', display:'flex', alignItems:'center', justifyContent:'center', fontSize:'0.62rem', fontWeight:700, flexShrink:0, marginTop:2 },
  bubble: { maxWidth:'70%', borderRadius:14, padding:'0.88rem 1.1rem', fontSize:'0.85rem', lineHeight:1.75 },
  msgMeta: { fontFamily:'var(--font-mono)', fontSize:'0.55rem', color:'var(--muted-light)', marginTop:'0.35rem', display:'flex', alignItems:'center', gap:'0.4rem' },
  sources: { display:'flex', flexWrap:'wrap', gap:'0.4rem', marginTop:'0.7rem' },
  sourceChip: { background:'var(--parchment)', border:'1px solid var(--border)', borderRadius:6, padding:'0.28rem 0.6rem', fontSize:'0.64rem', color:'var(--bark-dark)' },
  confBar: { display:'flex', alignItems:'center', gap:'0.5rem', marginTop:'0.6rem', fontFamily:'var(--font-mono)', fontSize:'0.58rem', color:'var(--muted)' },
  inputArea: { padding:'0.9rem 2rem 1.1rem', background:'rgba(255,255,255,.6)', backdropFilter:'blur(12px)', borderTop:'1px solid var(--border)', flexShrink:0 },
  inputWrap: { background:'white', border:'1.5px solid var(--border)', borderRadius:14, display:'flex', alignItems:'flex-end', gap:'0.5rem', padding:'0.7rem 0.7rem 0.7rem 1.1rem', transition:'all .2s' },
  textarea: { flex:1, border:'none', outline:'none', fontFamily:'var(--font-sans)', fontSize:'0.85rem', color:'var(--ink)', background:'transparent', resize:'none', maxHeight:120, lineHeight:1.6 },
  attachBtn: { width:34, height:34, background:'transparent', border:'none', cursor:'none', color:'var(--muted)', fontSize:'1rem', borderRadius:8, transition:'all .2s', display:'flex', alignItems:'center', justifyContent:'center' },
  sendBtn: { width:34, height:34, background:'var(--bark-deeper)', border:'none', borderRadius:8, color:'var(--parchment-light)', fontSize:'0.8rem', cursor:'none', transition:'all .2s', display:'flex', alignItems:'center', justifyContent:'center' },
  inputFooter: { marginTop:'0.4rem', display:'flex', alignItems:'center', gap:'1rem', fontFamily:'var(--font-mono)', fontSize:'0.58rem', color:'var(--muted-light)' },
  arxivBtn: { background:'none', border:'none', cursor:'none', fontFamily:'var(--font-mono)', fontSize:'0.6rem', color:'var(--bark)', display:'flex', alignItems:'center', gap:'0.35rem', padding:'2px 6px', borderRadius:5, transition:'opacity .2s' },
  pdfPanel: { borderLeft:'1px solid var(--border)', overflow:'hidden', transition:'width .35s cubic-bezier(.4,0,.2,1)', display:'flex', flexDirection:'column', background:'white', flexShrink:0 },
  panelTopbar: { display:'flex', alignItems:'center', gap:'0.6rem', padding:'0.75rem 1rem', borderBottom:'1px solid var(--border)', background:'var(--parchment)', flexShrink:0 },
  closeBtn: { background:'none', border:'none', cursor:'none', color:'var(--muted)', fontSize:'0.85rem', padding:'3px 6px', borderRadius:5, transition:'all .2s', fontFamily:'var(--font-sans)' },
  paperItem: { background:'var(--parchment-light)', border:'1px solid var(--border)', borderRadius:8, padding:'0.72rem', cursor:'none', transition:'all .2s' },
  piBtn: { fontSize:'0.6rem', padding:'2px 8px', borderRadius:4, border:'1px solid var(--border)', background:'white', color:'var(--bark-dark)', cursor:'none', transition:'all .2s', fontFamily:'var(--font-mono)' },
}
