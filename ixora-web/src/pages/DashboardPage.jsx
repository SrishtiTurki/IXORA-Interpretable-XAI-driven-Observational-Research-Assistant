import { useState } from 'react'
import { useAuth } from '../context/AuthContext'
import { useChat } from '../context/ChatContext'
import Sidebar from '../components/Sidebar'
import HomeView from '../components/HomeView'
import ChatView from '../components/ChatView'

export default function DashboardPage() {
  const { user } = useAuth()
  const { history, addChat } = useChat()

  const [domain, setDomain]       = useState('bio')
  const [view, setView]           = useState('home')   // 'home' | 'chat'
  const [activeChatId, setActiveChatId] = useState(null)

  const openNewChat = (d = domain) => {
    const id = 'chat_' + Date.now()
    addChat({ id, title: '', domain: d, time: 'Just now', bookmarked: false, msgs: [] })
    setDomain(d)
    setActiveChatId(id)
    setView('chat')
  }

  const openExistingChat = (histItem) => {
    setDomain(histItem.domain)
    setActiveChatId(histItem.id)
    setView('chat')
  }

  const handleSendSuggestion = (d, prompt) => {
    openNewChat(d)
    // Pass prompt via sessionStorage so ChatView picks it up
    sessionStorage.setItem('ixora_pending_prompt', prompt)
  }

  const handleDomainChange = (d) => {
    setDomain(d)
    if (view === 'chat') {
      // Start a fresh chat in new domain
      openNewChat(d)
    }
  }

  return (
    <div style={{ display:'grid', gridTemplateColumns:'260px 1fr', height:'100vh', overflow:'hidden' }}>
      <Sidebar
        domain={domain}
        onDomainChange={handleDomainChange}
        activeChatId={activeChatId}
        onSelectChat={openExistingChat}
        onNewChat={() => openNewChat(domain)}
      />

      <div style={{ display:'flex', flexDirection:'column', overflow:'hidden' }}>
        {/* Breadcrumb */}
        <div style={css.breadcrumb}>
          <span style={css.bc1}>IXORA</span>
          <span style={css.bcSep}>›</span>
          <span style={css.bc2}>
            {view === 'home' ? 'Workspace' : domain === 'bio' ? 'Biomedical' : domain === 'cs' ? 'Comp. Science' : 'General'}
          </span>
        </div>

        {view === 'home' && (
          <HomeView
            user={user}
            domain={domain}
            onStartChat={openNewChat}
            onSendSuggestion={handleSendSuggestion}
          />
        )}

        {view === 'chat' && activeChatId && (
          <ChatView
            key={activeChatId}
            chatId={activeChatId}
            domain={domain}
            onBack={() => { setView('home'); setActiveChatId(null) }}
          />
        )}
      </div>
    </div>
  )
}

const css = {
  breadcrumb: { display:'flex', alignItems:'center', gap:'0.4rem', padding:'0 1.5rem', height:44, borderBottom:'1px solid var(--border)', background:'rgba(255,255,255,.5)', backdropFilter:'blur(10px)', flexShrink:0 },
  bc1: { fontFamily:'var(--font-mono)', fontSize:'0.6rem', color:'var(--muted)' },
  bcSep: { color:'var(--border-strong)', fontSize:'0.7rem' },
  bc2: { fontFamily:'var(--font-mono)', fontSize:'0.6rem', fontWeight:600, color:'var(--bark)' },
}
