const DOMAIN_CFG = {
  bio: { label:'Biomedical Research',  model:'BioGPT · IMRAD · DoWhy',        emoji:'🧬', cls:'bio', desc:'Hypothesis generation, causal analysis, and experimental design with clinical precision.' },
  cs:  { label:'Computer Science',     model:'Qwen2.5-Coder · SHAP · Bench',  emoji:'💻', cls:'cs',  desc:'Hyperparameter optimization, algorithm analysis, and ML experiment design.' },
  gen: { label:'General Research',     model:'Mistral-Large · Universal',      emoji:'✦',  cls:'gen', desc:'Cross-domain queries, physics, mathematics, and open-ended scientific exploration.' },
}

const SUGGESTIONS = [
  { text:'CRISPR in gene therapy',            domain:'bio', prompt:'Explain the role of CRISPR-Cas9 in gene therapy' },
  { text:'Transformer architectures',         domain:'cs',  prompt:'Compare transformer architectures for NLP tasks' },
  { text:'Antibiotic resistance study',       domain:'bio', prompt:'Design an experiment to test antibiotic resistance mechanisms' },
  { text:'Diffusion models survey',           domain:'cs',  prompt:'What are the latest papers on diffusion models?' },
  { text:'Bayesian drug optimization',        domain:'bio', prompt:'Bayesian optimization for drug dosing parameters' },
  { text:'Attention mechanisms intuition',    domain:'cs',  prompt:'Explain attention mechanisms with clear intuition' },
]

function getGreeting() {
  const h = new Date().getHours()
  return h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening'
}

export default function HomeView({ user, domain, onStartChat, onSendSuggestion }) {
  const firstName = user?.first_name || 'Researcher'

  return (
    <div style={css.page}>
      <div style={css.bg}/>
      <div style={css.inner}>

        <div style={css.greeting}>
          <span style={css.greetLine}/>
          {getGreeting()}, {firstName}
          <span style={css.greetLine}/>
        </div>

        <h1 style={css.title}>
          Where would you like to<br/><em style={css.em}>begin today?</em>
        </h1>
        <p style={css.sub}>
          Select a research domain, or jump into a conversation. IXORA adapts its full reasoning pipeline to your field of inquiry.
        </p>

        {/* Domain cards */}
        <div style={css.cards}>
          {Object.entries(DOMAIN_CFG).map(([key, cfg]) => (
            <DomainCard key={key} id={key} cfg={cfg} active={domain === key} onClick={() => onStartChat(key)} />
          ))}
        </div>

        {/* Suggestions */}
        <div style={css.sugLabel}>— or try a prompt —</div>
        <div style={css.chips}>
          {SUGGESTIONS.map((s, i) => (
            <button key={i} style={css.chip} className="chip-btn"
              onClick={() => onSendSuggestion(s.domain, s.prompt)}>
              {s.text}
            </button>
          ))}
        </div>
      </div>

      <style>{`
        .chip-btn:hover{background:var(--warm-tan)!important;border-color:var(--bark)!important;}
        .domain-card:hover{transform:translateY(-3px);box-shadow:var(--shadow-lg);border-color:var(--border-strong)!important;}
        .domain-card:hover .dc-top-bar{opacity:1!important;}
      `}</style>
    </div>
  )
}

function DomainCard({ id, cfg, active, onClick }) {
  const topColor = id==='bio'?'linear-gradient(90deg,#6B8F71,#8E977D)':id==='cs'?'linear-gradient(90deg,#8A7650,#DBCEA5)':'linear-gradient(90deg,#8E977D,#DBCEA5)'
  const iconBg   = id==='bio'?'rgba(107,143,113,.15)':id==='cs'?'rgba(138,118,80,.15)':'rgba(142,151,125,.15)'
  return (
    <div className="domain-card" style={{ ...css.card, borderColor: active ? 'var(--bark)' : 'var(--border)', boxShadow: active ? '0 0 0 3px rgba(138,118,80,.12),var(--shadow)' : 'none' }} onClick={onClick}>
      <div className="dc-top-bar" style={{ position:'absolute', top:0, left:0, right:0, height:3, borderRadius:'14px 14px 0 0', background:topColor, opacity: active ? 1 : 0, transition:'opacity .25s' }}/>
      <div style={{ width:42, height:42, borderRadius:10, background:iconBg, display:'flex', alignItems:'center', justifyContent:'center', fontSize:'1.2rem', marginBottom:'1rem' }}>
        {cfg.emoji}
      </div>
      <div style={{ fontWeight:700, fontSize:'0.92rem', color:'var(--ink)', marginBottom:'0.4rem' }}>{cfg.label}</div>
      <div style={{ fontSize:'0.8rem', color:'var(--muted)', lineHeight:1.6, marginBottom:'0.9rem' }}>{cfg.desc}</div>
      <div style={{ fontFamily:'var(--font-mono)', fontSize:'0.63rem', color:'var(--bark)' }}>{cfg.model}</div>
    </div>
  )
}

const css = {
  page: { flex:1, display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', padding:'3rem 4rem', overflowY:'auto', position:'relative', background:'var(--parchment-light)' },
  bg: { position:'absolute', inset:0, background:'radial-gradient(ellipse 60% 50% at 60% 30%,rgba(219,206,165,.5) 0%,transparent 60%),radial-gradient(ellipse 40% 40% at 20% 70%,rgba(142,151,125,.1) 0%,transparent 50%)', pointerEvents:'none' },
  inner: { position:'relative', zIndex:1, maxWidth:780, width:'100%', textAlign:'center' },
  greeting: { fontFamily:'var(--font-mono)', fontSize:'0.72rem', letterSpacing:'0.18em', textTransform:'uppercase', color:'var(--bark)', marginBottom:'1.2rem', display:'flex', alignItems:'center', justifyContent:'center', gap:'0.6rem', animation:'fadeUp .5s both' },
  greetLine: { display:'block', width:24, height:1, background:'var(--bark)', opacity:0.5 },
  title: { fontFamily:'var(--font-serif)', fontSize:'clamp(2.4rem,4vw,3.6rem)', fontWeight:300, lineHeight:1.1, color:'var(--ink)', marginBottom:'1rem', animation:'fadeUp .6s .08s both' },
  em: { fontStyle:'italic', color:'var(--bark)' },
  sub: { fontSize:'0.96rem', color:'var(--muted)', lineHeight:1.8, marginBottom:'2.5rem', maxWidth:500, marginLeft:'auto', marginRight:'auto', animation:'fadeUp .6s .15s both' },
  cards: { display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:'1rem', marginBottom:'2.5rem', animation:'fadeUp .6s .22s both' },
  card: { background:'white', border:'1px solid var(--border)', borderRadius:14, padding:'1.5rem 1.3rem', cursor:'none', transition:'all .25s', textAlign:'left', position:'relative', overflow:'hidden' },
  sugLabel: { fontFamily:'var(--font-mono)', fontSize:'0.67rem', letterSpacing:'0.15em', textTransform:'uppercase', color:'var(--muted)', marginBottom:'0.75rem', animation:'fadeUp .6s .28s both' },
  chips: { display:'flex', flexWrap:'wrap', gap:'0.5rem', justifyContent:'center', animation:'fadeUp .6s .34s both' },
  chip: { background:'white', border:'1px solid var(--border)', borderRadius:20, padding:'0.5rem 1.1rem', fontSize:'0.82rem', color:'var(--ink-mid)', cursor:'none', transition:'all .2s', fontFamily:'var(--font-sans)' },
}