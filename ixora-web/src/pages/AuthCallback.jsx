import { useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'

export default function AuthCallback() {
  const [params] = useSearchParams()
  const navigate  = useNavigate()

  useEffect(() => {
    const code = params.get('code')
    if (!code) { navigate('/login'); return }

    fetch('http://localhost:8000/api/auth/google', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
    })
      .then(r => r.json())
      .then(data => {
        if (data.token) {
          localStorage.setItem('ixora_token', data.token)
          localStorage.setItem('ixora_user', JSON.stringify(data.user))
        }
        window.location.href = '/'
      })
      .catch(() => { window.location.href = '/login' })
  }, [])

  return (
    <div style={{ height:'100vh', display:'flex', alignItems:'center', justifyContent:'center', fontFamily:'var(--font-mono)', fontSize:'0.75rem', color:'var(--muted)', background:'var(--parchment)' }}>
      <div style={{ textAlign:'center' }}>
        <div style={{ width:24, height:24, border:'2px solid var(--bark)', borderTopColor:'transparent', borderRadius:'50%', animation:'spin .8s linear infinite', margin:'0 auto 1rem' }}/>
        Completing sign-in…
      </div>
    </div>
  )
}
