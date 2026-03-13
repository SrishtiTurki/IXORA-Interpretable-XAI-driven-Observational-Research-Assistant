import { createContext, useContext, useState, useCallback } from 'react'

const AuthContext = createContext(null)
const API = 'http://localhost:8000'

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem('ixora_user')) } catch { return null }
  })
  const [token, setToken] = useState(() => localStorage.getItem('ixora_token'))

  const saveSession = (tok, usr) => {
    localStorage.setItem('ixora_token', tok)
    localStorage.setItem('ixora_user', JSON.stringify(usr))
    setToken(tok)
    setUser(usr)
  }

  const login = useCallback(async (email, password) => {
    const res = await fetch(`${API}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data.detail?.message || 'Invalid credentials')
    saveSession(data.token, data.user)
    return data
  }, [])

  const register = useCallback(async (first_name, last_name, email, password) => {
    const res = await fetch(`${API}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ first_name, last_name, email, password }),
    })
    const data = await res.json()
    if (!res.ok) throw new Error(data.detail?.message || 'Registration failed')
    saveSession(data.token, data.user)
    return data
  }, [])

  const googleAuth = useCallback(() => {
    const clientId = '286470694523-2gahpdu7lt8r9270hs93tarsjda9762g.apps.googleusercontent.com'
    const redirectUri = encodeURIComponent('http://localhost:3000/auth/callback')
    const scope = encodeURIComponent('openid email profile')
    window.location.href = `https://accounts.google.com/o/oauth2/v2/auth?client_id=${clientId}&redirect_uri=${redirectUri}&response_type=code&scope=${scope}&access_type=offline&prompt=consent`
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem('ixora_token')
    localStorage.removeItem('ixora_user')
    setToken(null)
    setUser(null)
  }, [])

  const authFetch = useCallback(async (url, opts = {}) => {
  // Build headers — only add Authorization if token actually exists
  const headers = {
    'Content-Type': 'application/json',
    ...opts.headers,               // let caller override if needed
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const finalOpts = {
    ...opts,
    headers,
  };

  const res = await fetch(`${API}${url}`, finalOpts);

  // Auto-logout + error only when we actually sent a token
  // (prevents treating public 401s as session expiration)
  if (res.status === 401 && token) {
    logout();
    throw new Error('Session expired. Please sign in again.');
  }

  return res;
}, [token, logout]);
  return (
    <AuthContext.Provider value={{ user, token, login, register, googleAuth, logout, authFetch, isAuthed: !!token }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
