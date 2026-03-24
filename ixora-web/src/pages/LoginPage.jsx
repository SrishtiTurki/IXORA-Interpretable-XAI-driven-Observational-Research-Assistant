import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { toast } from '../components/Toast'

const s = {
  page: { display:'grid', gridTemplateColumns:'1fr 1fr', height:'100vh', fontFamily:'var(--font-sans)', cursor:'none' },
  left: { background:'linear-gradient(160deg,#2d3322 0%,#111710 100%)', padding:'3rem 4rem', display:'flex', flexDirection:'column', position:'relative', overflow:'hidden' },
  leftBefore: { position:'absolute',inset:0, background:'radial-gradient(ellipse 60% 50% at 60% 50%,rgba(134,171,137,.07) 0%,transparent 60%)', pointerEvents:'none' },
  grid: { position:'absolute',inset:0, backgroundImage:'linear-gradient(rgba(162,139,85,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(162,139,85,.04) 1px,transparent 1px)', backgroundSize:'50px 50px' },
  orb1: { position:'absolute', width:300,height:300, background:'rgba(134,171,137,.06)', borderRadius:'50%', filter:'blur(80px)', top:-100,right:-100, animation:'orbFloat1 8s ease-in-out infinite' },
  orb2: { position:'absolute', width:200,height:200, background:'rgba(162,139,85,.05)', borderRadius:'50%', filter:'blur(80px)', bottom:100,left:-50, animation:'orbFloat2 10s ease-in-out infinite' },
  logo: { fontFamily:'var(--font-serif)', fontSize:'2.2rem', fontWeight:300, color:'var(--parchment-light)', letterSpacing:'0.12em', display:'flex', alignItems:'center', gap:'0.5rem', position:'relative', zIndex:2, textDecoration:'none' },
  logoDot: { width:8,height:8, background:'var(--warm-tan)', borderRadius:'50%', animation:'pulse 2.5s infinite', flexShrink:0 },
  leftContent: { flex:1, display:'flex', flexDirection:'column', justifyContent:'center', position:'relative', zIndex:2 },
  leftTag: { fontFamily:'var(--font-mono)', fontSize:'0.6rem', letterSpacing:'0.2em', textTransform:'uppercase', color:'var(--bark)', display:'flex', alignItems:'center', gap:'0.6rem', marginBottom:'2rem' },
  leftTagLine: { display:'block', width:20, height:1, background:'var(--bark)' },
  leftTitle: { fontFamily:'var(--font-serif)', fontSize:'clamp(2rem,3.2vw,3.2rem)', fontWeight:300, lineHeight:1.1, color:'var(--parchment-light)', marginBottom:'1.5rem' },
  leftDesc: { fontSize:'0.85rem', lineHeight:1.9, color:'rgba(254,250,224,.45)', maxWidth:380, marginBottom:'2.5rem' },
  termCard: { background:'rgba(255,255,255,.03)', border:'1px solid rgba(162,139,85,.15)', borderRadius:10, padding:'1.5rem', maxWidth:380, position:'relative' },
  termBar: { position:'absolute',top:0,left:0,right:0,height:2, background:'linear-gradient(90deg,var(--sage),var(--bark))', borderRadius:'10px 10px 0 0' },
  termDots: { display:'flex',gap:6,marginBottom:'1rem' },
  termDot: (c) => ({ width:8,height:8,borderRadius:'50%',background:c }),
  termLine: { fontFamily:'var(--font-mono)', fontSize:'0.62rem', color:'rgba(254,250,224,.5)', lineHeight:2, paddingLeft:'0.5rem' },
  termPrompt: { color:'var(--bark)', marginRight:'0.4rem' },
  termCmd: { color:'#CBE2B5' },
  termOut: { color:'rgba(254,250,224,.35)', fontStyle:'italic' },
  termCursor: { display:'inline-block',width:6,height:12,background:'var(--bark)',verticalAlign:'middle',animation:'blink 1s infinite',marginLeft:2 },
  leftFooter: { position:'relative',zIndex:2, fontFamily:'var(--font-mono)', fontSize:'0.55rem', color:'rgba(254,250,224,.2)', letterSpacing:'0.1em' },

  right: { background:'var(--parchment)', display:'flex', alignItems:'center', justifyContent:'center', padding:'3rem', position:'relative', overflow:'hidden' },
  rightBg: { position:'absolute',inset:0, background:'radial-gradient(ellipse 70% 50% at 80% 20%,rgba(231,251,230,.6) 0%,transparent 60%),radial-gradient(ellipse 50% 40% at 20% 80%,rgba(203,226,181,.3) 0%,transparent 50%)' },
  wrap: { width:'100%', maxWidth:420, position:'relative', zIndex:2 },

  formTag: { fontFamily:'var(--font-mono)', fontSize:'0.6rem', letterSpacing:'0.2em', textTransform:'uppercase', color:'var(--bark)', display:'block', marginBottom:'0.6rem' },
  formTitle: { fontFamily:'var(--font-serif)', fontSize:'2.2rem', fontWeight:300, color:'var(--ink)', lineHeight:1.15, marginBottom:'2rem' },
  formTitleEm: { fontStyle:'italic', color:'var(--bark)' },

  tabs: { display:'flex', gap:0, marginBottom:'2rem', background:'rgba(138,118,80,.08)', borderRadius:9, padding:3 },
  tab: (active) => ({ flex:1, padding:'0.55rem', border:'none', borderRadius:7, fontFamily:'var(--font-sans)', fontSize:'0.72rem', fontWeight:600, letterSpacing:'0.08em', cursor:'none', transition:'all .2s',
    background: active ? 'var(--bark-deeper)' : 'transparent',
    color: active ? 'var(--parchment-light)' : 'var(--muted)',
  }),

  field: { marginBottom:'1.2rem' },
  label: { display:'block', fontFamily:'var(--font-mono)', fontSize:'0.58rem', letterSpacing:'0.12em', textTransform:'uppercase', color:'var(--muted)', marginBottom:'0.45rem' },
  inputWrap: { position:'relative' },
  input: (err) => ({ width:'100%', padding:'0.75rem 1rem', background:'white', border:`1.5px solid ${err ? 'var(--error)' : 'var(--border)'}`, borderRadius:9, fontFamily:'var(--font-sans)', fontSize:'0.85rem', color:'var(--ink)', outline:'none', transition:'border .2s,box-shadow .2s' }),
  inputMsg: (err) => ({ fontSize:'0.65rem', marginTop:'0.3rem', color: err ? 'var(--error)' : 'var(--success)' }),

  pwRow: { display:'grid', gridTemplateColumns:'1fr 1fr', gap:'0.75rem' },

  btn: { width:'100%', padding:'0.85rem', background:'var(--bark-deeper)', color:'var(--parchment-light)', border:'none', borderRadius:9, fontFamily:'var(--font-sans)', fontSize:'0.82rem', fontWeight:600, letterSpacing:'0.06em', cursor:'none', transition:'all .2s', marginBottom:'1rem', display:'flex', alignItems:'center', justifyContent:'center', gap:'0.5rem' },
  googleBtn: { width:'100%', padding:'0.75rem', background:'white', color:'var(--ink)', border:'1.5px solid var(--border)', borderRadius:9, fontFamily:'var(--font-sans)', fontSize:'0.8rem', fontWeight:500, cursor:'none', transition:'all .2s', display:'flex', alignItems:'center', justifyContent:'center', gap:'0.6rem', marginBottom:'1.2rem' },
  divider: { display:'flex', alignItems:'center', gap:'0.75rem', margin:'1.2rem 0', color:'var(--muted-light)', fontSize:'0.7rem', fontFamily:'var(--font-mono)' },
  divLine: { flex:1, height:1, background:'var(--border)' },
  switchLink: { textAlign:'center', fontSize:'0.75rem', color:'var(--muted)', marginTop:'0.5rem' },
  switchBtn: { background:'none', border:'none', color:'var(--bark)', fontWeight:600, cursor:'none', fontSize:'0.75rem', fontFamily:'var(--font-sans)' },
  errAlert: { background:'rgba(181,97,74,.1)', border:'1px solid rgba(181,97,74,.3)', borderRadius:8, padding:'0.75rem 1rem', fontSize:'0.78rem', color:'var(--error)', marginBottom:'1rem' },
  successOverlay: { position:'fixed',inset:0, background:'var(--parchment)', display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', zIndex:9999, animation:'fadeIn .4s ease' },
  successIcon: { fontSize:'3rem', marginBottom:'1rem', animation:'fadeUp .5s .1s both' },
  successTitle: { fontFamily:'var(--font-serif)', fontSize:'2.5rem', fontWeight:300, color:'var(--ink)', animation:'fadeUp .5s .2s both' },
  successSub: { color:'var(--muted)', fontSize:'0.85rem', marginTop:'0.5rem', animation:'fadeUp .5s .3s both' },
}

export default function LoginPage() {
  const { login, register, googleAuth, isAuthed } = useAuth()
  const navigate = useNavigate()
  const [panel, setPanel]       = useState('login')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')
  const [success, setSuccess]   = useState(null)

  // Login fields
  const [loginEmail, setLoginEmail]   = useState('')
  const [loginPw, setLoginPw]         = useState('')
  const [loginErrs, setLoginErrs]     = useState({})

  // Register fields
  const [regFirst, setRegFirst]       = useState('')
  const [regLast, setRegLast]         = useState('')
  const [regEmail, setRegEmail]       = useState('')
  const [regPw, setRegPw]             = useState('')
  const [regConfirm, setRegConfirm]   = useState('')
  const [regErrs, setRegErrs]         = useState({})

  // Forgot
  const [forgotEmail, setForgotEmail] = useState('')

  useEffect(() => {
    if (isAuthed) navigate('/', { replace: true })
  }, [isAuthed, navigate])

  const handleLogin = async () => {
    const errs = {}
    if (!loginEmail || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(loginEmail)) errs.email = 'Valid email required'
    if (!loginPw || loginPw.length < 8) errs.pw = 'Minimum 8 characters'
    setLoginErrs(errs)
    if (Object.keys(errs).length) return
    setLoading(true); setError('')
    try {
      const data = await login(loginEmail, loginPw)
      setSuccess({ title: `Welcome back, ${data.user.first_name}!`, sub: 'Taking you to your workspace…' })
      setTimeout(() => navigate('/'), 1600)
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  const handleRegister = async () => {
    const errs = {}
    if (!regFirst.trim()) errs.first = 'Required'
    if (!regLast.trim())  errs.last  = 'Required'
    if (!regEmail || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(regEmail)) errs.email = 'Valid email required'
    if (!regPw || regPw.length < 8) errs.pw = 'Minimum 8 characters'
    if (regPw !== regConfirm) errs.confirm = 'Passwords do not match'
    setRegErrs(errs)
    if (Object.keys(errs).length) return
    setLoading(true); setError('')
    try {
      const data = await register(regFirst, regLast, regEmail, regPw)
      setSuccess({ title: `Welcome, ${data.user.first_name}!`, sub: 'Your workspace is ready…' })
      setTimeout(() => navigate('/'), 1600)
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  const handleForgot = async () => {
    if (!forgotEmail || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(forgotEmail)) { setError('Valid email required'); return }
    setError(''); toast('If that email exists, a reset link has been sent.', 'success')
  }

  return (
    <>
      <style>{`
        @keyframes orbFloat1{0%,100%{transform:translate(0,0)}50%{transform:translate(-20px,20px)}}
        @keyframes orbFloat2{0%,100%{transform:translate(0,0)}50%{transform:translate(20px,-20px)}}
        .auth-input:focus{border-color:var(--bark)!important;box-shadow:0 0 0 3px rgba(138,118,80,.1);}
        .auth-btn:hover{background:var(--bark)!important;}
        .google-btn:hover{border-color:var(--bark)!important;background:var(--parchment)!important;}
      `}</style>

      {success && (
        <div style={s.successOverlay}>
          <div style={s.successIcon}>✓</div>
          <div style={s.successTitle}>{success.title}</div>
          <div style={s.successSub}>{success.sub}</div>
        </div>
      )}

      <div style={s.page}>
        {/* LEFT */}
        <div style={s.left}>
          <div style={s.leftBefore}/>
          <div style={s.grid}/>
          <div style={s.orb1}/>
          <div style={s.orb2}/>

          <a href="/" style={s.logo}>
            <span style={s.logoDot}/>
            IXORA
          </a>

          <div style={s.leftContent}>
            <div style={s.leftTag}>
              <span style={s.leftTagLine}/>
              Multi-Agent Research Intelligence
            </div>
            <h2 style={s.leftTitle}>
              Research that <em style={{fontStyle:'italic',color:'#CBE2B5'}}>thinks</em><br/>alongside you
            </h2>
            <p style={s.leftDesc}>
              Six specialized agents. Causal analysis. Bayesian optimization. Full reasoning traces — so you always know <em>why</em>, not just what.
            </p>

            <div style={s.termCard}>
              <div style={s.termBar}/>
              <div style={s.termDots}>
                <div style={s.termDot('#ff5f57')}/>
                <div style={s.termDot('#febc2e')}/>
                <div style={s.termDot('#28c840')}/>
              </div>
              <div style={s.termLine}>
                <span style={s.termPrompt}>›</span>
                <span style={s.termCmd}>ixora.pipeline.run(domain="biomed")</span>
              </div>
              <div style={s.termLine}><span style={s.termOut}>✓ Intent detected · Biomedical</span></div>
              <div style={s.termLine}><span style={s.termOut}>✓ Parameters extracted (conf: 94%)</span></div>
              <div style={s.termLine}><span style={s.termOut}>✓ 6 agents activated</span></div>
              <div style={s.termLine}>
                <span style={s.termPrompt}>›</span>
                <span style={s.termCursor}/>
              </div>
            </div>
          </div>

          <div style={s.leftFooter}>IXORA · v1.0 · PID-44</div>
        </div>

        {/* RIGHT */}
        <div style={s.right}>
          <div style={s.rightBg}/>
          <div style={s.wrap}>
            <span style={s.formTag}>
              {panel === 'login' ? 'Sign In' : panel === 'register' ? 'Create Account' : 'Reset Password'}
            </span>
            <h1 style={s.formTitle}>
              {panel === 'login'    && <><em style={s.formTitleEm}>Welcome</em> back</>}
              {panel === 'register' && <>Join <em style={s.formTitleEm}>IXORA</em></>}
              {panel === 'forgot'   && <>Reset your <em style={s.formTitleEm}>password</em></>}
            </h1>

            {panel !== 'forgot' && (
              <div style={s.tabs}>
                <button style={s.tab(panel==='login')}    className="auth-btn" onClick={()=>{setPanel('login');setError('')}}>Sign In</button>
                <button style={s.tab(panel==='register')} className="auth-btn" onClick={()=>{setPanel('register');setError('')}}>Register</button>
              </div>
            )}

            {error && <div style={s.errAlert}>{error}</div>}

            {/* GOOGLE */}
            {panel !== 'forgot' && (
              <>
                <button style={s.googleBtn} className="google-btn" onClick={googleAuth}>
                  <svg width="18" height="18" viewBox="0 0 48 48"><path fill="#4285F4" d="M47.5 24.5c0-1.6-.1-3.2-.4-4.7H24v9h13.1c-.6 3-2.3 5.5-4.9 7.2v6h7.9c4.6-4.3 7.4-10.6 7.4-17.5z"/><path fill="#34A853" d="M24 48c6.5 0 11.9-2.1 15.9-5.8l-7.9-6c-2.1 1.4-4.8 2.3-8 2.3-6.1 0-11.3-4.1-13.2-9.7H2.7v6.2C6.7 42.9 14.8 48 24 48z"/><path fill="#FBBC05" d="M10.8 28.8c-.5-1.4-.7-2.9-.7-4.4s.3-3 .7-4.4v-6.2H2.7C1 17.2 0 20.5 0 24s1 6.8 2.7 9.8l8.1-5z"/><path fill="#EA4335" d="M24 9.5c3.4 0 6.5 1.2 8.9 3.5l6.6-6.6C35.9 2.5 30.4 0 24 0 14.8 0 6.7 5.1 2.7 12.6l8.1 6.2C12.7 13.6 17.9 9.5 24 9.5z"/></svg>
                  {panel === 'register' ? 'Sign up with Google' : 'Continue with Google'}
                </button>
                <div style={s.divider}><div style={s.divLine}/><span>or</span><div style={s.divLine}/></div>
              </>
            )}

            {/* LOGIN FORM */}
            {panel === 'login' && (
              <>
                <div style={s.field}>
                  <label style={s.label}>Email</label>
                  <input className="auth-input" style={s.input(loginErrs.email)} type="email" placeholder="you@university.edu" value={loginEmail} onChange={e=>setLoginEmail(e.target.value)} onKeyDown={e=>e.key==='Enter'&&handleLogin()}/>
                  {loginErrs.email && <div style={s.inputMsg(true)}>{loginErrs.email}</div>}
                </div>
                <div style={s.field}>
                  <label style={s.label}>Password</label>
                  <input className="auth-input" style={s.input(loginErrs.pw)} type="password" placeholder="••••••••" value={loginPw} onChange={e=>setLoginPw(e.target.value)} onKeyDown={e=>e.key==='Enter'&&handleLogin()}/>
                  {loginErrs.pw && <div style={s.inputMsg(true)}>{loginErrs.pw}</div>}
                </div>
                <div style={{textAlign:'right',marginBottom:'1.2rem'}}>
                  <button style={s.switchBtn} onClick={()=>{setPanel('forgot');setError('')}}>Forgot password?</button>
                </div>
                <button style={s.btn} className="auth-btn" onClick={handleLogin} disabled={loading}>
                  {loading ? <span style={{display:'inline-block',width:16,height:16,border:'2px solid rgba(255,255,255,.3)',borderTopColor:'white',borderRadius:'50%',animation:'spin 0.8s linear infinite'}}/> : null}
                  {loading ? 'Signing in…' : 'Sign In to IXORA'}
                </button>
              </>
            )}

            {/* REGISTER FORM */}
            {panel === 'register' && (
              <>
                <div style={s.pwRow}>
                  <div style={s.field}>
                    <label style={s.label}>First name</label>
                    <input className="auth-input" style={s.input(regErrs.first)} placeholder="Ada" value={regFirst} onChange={e=>setRegFirst(e.target.value)}/>
                    {regErrs.first && <div style={s.inputMsg(true)}>{regErrs.first}</div>}
                  </div>
                  <div style={s.field}>
                    <label style={s.label}>Last name</label>
                    <input className="auth-input" style={s.input(regErrs.last)} placeholder="Lovelace" value={regLast} onChange={e=>setRegLast(e.target.value)}/>
                    {regErrs.last && <div style={s.inputMsg(true)}>{regErrs.last}</div>}
                  </div>
                </div>
                <div style={s.field}>
                  <label style={s.label}>Email</label>
                  <input className="auth-input" style={s.input(regErrs.email)} type="email" placeholder="you@university.edu" value={regEmail} onChange={e=>setRegEmail(e.target.value)}/>
                  {regErrs.email && <div style={s.inputMsg(true)}>{regErrs.email}</div>}
                </div>
                <div style={s.field}>
                  <label style={s.label}>Password</label>
                  <input className="auth-input" style={s.input(regErrs.pw)} type="password" placeholder="Min. 8 characters" value={regPw} onChange={e=>setRegPw(e.target.value)}/>
                  {regErrs.pw && <div style={s.inputMsg(true)}>{regErrs.pw}</div>}
                </div>
                <div style={s.field}>
                  <label style={s.label}>Confirm password</label>
                  <input className="auth-input" style={s.input(regErrs.confirm)} type="password" placeholder="Repeat password" value={regConfirm} onChange={e=>setRegConfirm(e.target.value)} onKeyDown={e=>e.key==='Enter'&&handleRegister()}/>
                  {regErrs.confirm && <div style={s.inputMsg(true)}>{regErrs.confirm}</div>}
                  {regConfirm && regPw === regConfirm && <div style={s.inputMsg(false)}>Passwords match ✓</div>}
                </div>
                <button style={s.btn} className="auth-btn" onClick={handleRegister} disabled={loading}>
                  {loading ? <span style={{display:'inline-block',width:16,height:16,border:'2px solid rgba(255,255,255,.3)',borderTopColor:'white',borderRadius:'50%',animation:'spin 0.8s linear infinite'}}/> : null}
                  {loading ? 'Creating account…' : 'Create Account'}
                </button>
              </>
            )}

            {/* FORGOT FORM */}
            {panel === 'forgot' && (
              <>
                <div style={s.field}>
                  <label style={s.label}>Email address</label>
                  <input className="auth-input" style={s.input(false)} type="email" placeholder="you@university.edu" value={forgotEmail} onChange={e=>setForgotEmail(e.target.value)} onKeyDown={e=>e.key==='Enter'&&handleForgot()}/>
                </div>
                <button style={s.btn} className="auth-btn" onClick={handleForgot}>Send Reset Link</button>
                <div style={s.switchLink}>
                  <button style={s.switchBtn} onClick={()=>{setPanel('login');setError('')}}>← Back to sign in</button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </>
  )
}