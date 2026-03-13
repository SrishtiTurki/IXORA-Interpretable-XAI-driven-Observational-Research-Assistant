import { useState, useEffect } from 'react'

// Simple event bus for toast
const listeners = []
export const toast = (msg, type = 'default') => listeners.forEach(fn => fn(msg, type))

export default function Toast() {
  const [visible, setVisible]   = useState(false)
  const [message, setMessage]   = useState('')
  const [type, setType]         = useState('default')
  let timer

  useEffect(() => {
    const handler = (msg, t) => {
      clearTimeout(timer)
      setMessage(msg)
      setType(t)
      setVisible(true)
      timer = setTimeout(() => setVisible(false), 2400)
    }
    listeners.push(handler)
    return () => {
      const i = listeners.indexOf(handler)
      if (i > -1) listeners.splice(i, 1)
    }
  }, [])

  const bg = type === 'error' ? 'var(--error)' : type === 'success' ? 'var(--success)' : 'var(--bark-deeper)'

  return (
    <div style={{
      position: 'fixed', bottom: '1.5rem', right: '1.5rem',
      background: bg, color: 'var(--parchment-light)',
      padding: '0.7rem 1.2rem', borderRadius: '10px',
      fontFamily: 'var(--font-sans)', fontSize: '0.78rem',
      boxShadow: 'var(--shadow-lg)', zIndex: 9000,
      opacity: visible ? 1 : 0,
      transform: visible ? 'translateY(0)' : 'translateY(10px)',
      transition: 'all 0.3s', pointerEvents: 'none',
    }}>
      {message}
    </div>
  )
}
