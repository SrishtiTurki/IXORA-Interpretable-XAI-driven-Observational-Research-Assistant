import { createContext, useContext, useState, useCallback } from 'react'

const ChatContext = createContext(null)

const DEMO_HISTORY = [
  { id: 'h1', title: 'CRISPR gene editing mechanisms',      domain: 'bio', time: '2h ago',    bookmarked: true,  msgs: [] },
  { id: 'h2', title: 'Transformer architecture comparison', domain: 'cs',  time: 'Yesterday', bookmarked: false, msgs: [] },
  { id: 'h3', title: 'Antibiotic resistance in E. coli',   domain: 'bio', time: '2d ago',    bookmarked: true,  msgs: [] },
  { id: 'h4', title: 'Bayesian optimization hyperparams',  domain: 'cs',  time: '3d ago',    bookmarked: false, msgs: [] },
  { id: 'h5', title: 'Quantum entanglement overview',      domain: 'gen', time: '5d ago',    bookmarked: false, msgs: [] },
]

export function ChatProvider({ children }) {
  const [history, setHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem('ixora_history')) || DEMO_HISTORY }
    catch { return DEMO_HISTORY }
  })

  const save = (h) => {
    setHistory(h)
    localStorage.setItem('ixora_history', JSON.stringify(h))
  }

  const addChat = useCallback((chat) => {
    save(h => [chat, ...h])
  }, [])

  const updateChat = useCallback((id, patch) => {
    setHistory(prev => {
      const next = prev.map(c => c.id === id ? { ...c, ...patch } : c)
      localStorage.setItem('ixora_history', JSON.stringify(next))
      return next
    })
  }, [])

  const toggleBookmark = useCallback((id) => {
    setHistory(prev => {
      const next = prev.map(c => c.id === id ? { ...c, bookmarked: !c.bookmarked } : c)
      localStorage.setItem('ixora_history', JSON.stringify(next))
      return next
    })
  }, [])

  const deleteChat = useCallback((id) => {
    setHistory(prev => {
      const next = prev.filter(c => c.id !== id)
      localStorage.setItem('ixora_history', JSON.stringify(next))
      return next
    })
  }, [])

  const appendMessage = useCallback((chatId, msg) => {
    setHistory(prev => {
      const next = prev.map(c => c.id === chatId ? { ...c, msgs: [...c.msgs, msg] } : c)
      localStorage.setItem('ixora_history', JSON.stringify(next))
      return next
    })
  }, [])

  return (
    <ChatContext.Provider value={{ history, addChat, updateChat, toggleBookmark, deleteChat, appendMessage }}>
      {children}
    </ChatContext.Provider>
  )
}

export const useChat = () => useContext(ChatContext)
