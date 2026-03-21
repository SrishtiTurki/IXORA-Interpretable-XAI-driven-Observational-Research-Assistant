import { useState } from 'react'
import { useAuth } from '../context/AuthContext'
import { useChat } from '../context/ChatContext'
import { toast } from './Toast'

const DOMAIN_CFG = {
  bio: { label: 'Biomedical',      model: 'BioGPT',        emoji: '🧬', cls: 'bio' },
  cs:  { label: 'Comp. Science',   model: 'Qwen2.5',       emoji: '💻', cls: 'cs'  },
  gen: { label: 'General',         model: 'Mistral-Large', emoji: '✦',  cls: 'gen' },
}

export default function Sidebar({ domain, onDomainChange, activeChatId, onSelectChat, onNewChat }) {
  const { user, logout } = useAuth()
  const { history, toggleBookmark, deleteChat } = useChat()
  const [confirmDelete, setConfirmDelete] = useState(null)
  const [search, setSearch] = useState('')

  const initials = user ? (user.first_name?.[0] || '') + (user.last_name?.[0] || '') : '?'
  const displayName = user ? `${user.first_name} ${user.last_name || ''}`.trim() : 'Researcher'

  const filtered = history.filter(h =>
    !search || h.title.toLowerCase().includes(search.toLowerCase())
  )
  const bookmarked = filtered.filter(h => h.bookmarked)
  const recent     = filtered.filter(h => !h.bookmarked)

  const handleLogout = () => {
    logout()
    toast('Signed out')
  }

  return (
    <aside style={css.sidebar}>
      <div style={css.sidebarBg}/>

      {/* Delete confirm modal */}
      {confirmDelete && (
        <div style={css.deleteOverlay}>
          <div style={css.deleteModal}>
            <div style={css.deleteTitle}>Delete conversation?</div>
            <div style={css.deleteSubtitle}>This cannot be undone.</div>
            <div style={css.deleteActions}>
              <button style={css.deleteCancelBtn} onClick={() => setConfirmDelete(null)}>Cancel</button>
              <button style={css.deleteConfirmBtn} onClick={() => {
                deleteChat(confirmDelete)
                toast('Conversation deleted')
                setConfirmDelete(null)
              }}>Delete</button>
            </div>
          </div>
        </div>
      )}

      {/* Logo */}
      <div style={css.logo}>
        <span style={css.logoDot}/>
        <span style={css.logoText}>IXORA</span>
      </div>

      {/* Domain switcher */}
      <div style={css.sectionLabel}>Workspace</div>
      <div style={css.domainList}>
        {Object.entries(DOMAIN_CFG).map(([key, cfg]) => (
          <button
            key={key}
            style={css.domainPill(domain === key)}
            onClick={() => onDomainChange(key)}
          >
            <span style={css.dpIcon(key)}>{cfg.emoji}</span>
            <span style={css.dpLabel(domain === key)}>{cfg.label}</span>
            <span style={css.dpBadge(domain === key)}>{cfg.model}</span>
          </button>
        ))}
      </div>

      <button style={css.newChatBtn} onClick={onNewChat}>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        New Conversation
      </button>

      {/* Search */}
      <div style={css.searchWrap}>
        <input
          style={css.searchInput}
          placeholder="Search history…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
      </div>

      {/* History */}
      <div style={css.historyScroll}>
        {bookmarked.length > 0 && (
          <>
            <div style={css.groupLabel}>★ Bookmarked</div>
            {bookmarked.map(h => (
              <HistoryItem key={h.id} item={h} active={h.id === activeChatId}
                onSelect={() => onSelectChat(h)}
                onBookmark={() => { toggleBookmark(h.id); toast(h.bookmarked ? 'Bookmark removed' : '★ Bookmarked') }}
                onDelete={() => setConfirmDelete(h.id)}
              />
            ))}
          </>
        )}
        {recent.length > 0 && (
          <>
            <div style={css.groupLabel}>Recent</div>
            {recent.map(h => (
              <HistoryItem key={h.id} item={h} active={h.id === activeChatId}
                onSelect={() => onSelectChat(h)}
                onBookmark={() => { toggleBookmark(h.id); toast(h.bookmarked ? 'Bookmark removed' : '★ Bookmarked') }}
                onDelete={() => setConfirmDelete(h.id)}
              />
            ))}
          </>
        )}
        {filtered.length === 0 && (
          <div style={{ padding:'1rem 0.5rem', fontFamily:'var(--font-mono)', fontSize:'0.62rem', color:'rgba(219,206,165,.25)', textAlign:'center' }}>
            {search ? 'No results' : 'No conversations yet'}
          </div>
        )}
      </div>

      {/* User */}
      <div style={css.footer}>
        <div style={css.avatar}>{initials.toUpperCase()}</div>
        <div style={css.userInfo}>
          <div style={css.userName}>{displayName}</div>
          <div style={css.userPlan}>{user?.provider === 'google' ? 'Google · ' : ''}Free Plan</div>
        </div>
        <button style={css.logoutBtn} onClick={handleLogout} title="Sign out">⏻</button>
      </div>
    </aside>
  )
}

function HistoryItem({ item, active, onSelect, onBookmark, onDelete }) {
  const [hovered, setHovered] = useState(false)
  const cfg = { bio:'🧬', cs:'💻', gen:'✦' }
  return (
    <div
      style={css.histItem(active)}
      onClick={onSelect}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <span style={{ fontSize:'0.8rem', marginTop:2, flexShrink:0, opacity:0.6 }}>{cfg[item.domain] || '✦'}</span>
      <div style={{ flex:1, minWidth:0 }}>
        <div style={css.histTitle(active)}>{item.title}</div>
        <div style={css.histMeta}>
          <span style={css.domainTag(item.domain)}>{item.domain.toUpperCase()}</span>
          {item.time}
        </div>
      </div>
      <div style={{ display:'flex', alignItems:'center', gap:2, flexShrink:0 }}>
        <button
          style={{ background:'none', border:'none', cursor:'none', color:'var(--warm-tan)', fontSize:'0.85rem', padding:'2px 4px', opacity: item.bookmarked ? 1 : (hovered ? 0.5 : 0.2), transition:'opacity .2s' }}
          onClick={e => { e.stopPropagation(); onBookmark() }}
        >
          {item.bookmarked ? '★' : '☆'}
        </button>
        <button
          style={{ background:'none', border:'none', cursor:'none', color:'rgba(181,97,74,.7)', fontSize:'0.75rem', padding:'2px 4px', opacity: hovered ? 0.8 : 0, transition:'opacity .2s', lineHeight:1 }}
          onClick={e => { e.stopPropagation(); onDelete() }}
          title="Delete conversation"
        >
          ✕
        </button>
      </div>
    </div>
  )
}

const css = {
  sidebar: { background:'var(--bark-deeper)', display:'flex', flexDirection:'column', borderRight:'1px solid rgba(138,118,80,.2)', overflow:'hidden', position:'relative', width:260, flexShrink:0 },
  sidebarBg: { position:'absolute', inset:0, background:'radial-gradient(ellipse 100% 60% at 50% 0%,rgba(219,206,165,.04) 0%,transparent 60%)', pointerEvents:'none' },

  logo: { padding:'1.6rem 1.5rem 1.2rem', borderBottom:'1px solid rgba(219,206,165,.1)', display:'flex', alignItems:'center', gap:'0.5rem', flexShrink:0, position:'relative', zIndex:1 },
  logoDot: { display:'block', width:7, height:7, background:'var(--warm-tan)', borderRadius:'50%', animation:'pulse 2.5s infinite', flexShrink:0 },
  logoText: { fontFamily:'var(--font-serif)', fontSize:'1.7rem', fontWeight:300, letterSpacing:'0.12em', color:'var(--parchment-light)' },

  sectionLabel: { padding:'1rem 1.2rem 0.4rem', fontFamily:'var(--font-mono)', fontSize:'0.63rem', letterSpacing:'0.18em', textTransform:'uppercase', color:'rgba(219,206,165,.3)', flexShrink:0, position:'relative', zIndex:1 },

  domainList: { padding:'0.25rem 1rem', display:'flex', flexDirection:'column', gap:3, flexShrink:0, position:'relative', zIndex:1 },
  domainPill: (active) => ({ padding:'0.6rem 0.9rem', borderRadius:8, display:'flex', alignItems:'center', gap:'0.65rem', border:`1px solid ${active ? 'rgba(219,206,165,.2)' : 'transparent'}`, background: active ? 'rgba(219,206,165,.12)' : 'transparent', cursor:'none', transition:'all .2s', textAlign:'left', width:'100%' }),
  dpIcon: (k) => ({ width:28, height:28, borderRadius:7, display:'flex', alignItems:'center', justifyContent:'center', fontSize:'0.85rem', flexShrink:0, background: k==='bio'?'rgba(107,143,113,.25)':k==='cs'?'rgba(138,118,80,.25)':'rgba(142,151,125,.25)' }),
  dpLabel: (active) => ({ fontSize:'0.84rem', fontWeight:500, color: active ? 'var(--parchment-light)' : 'rgba(236,231,209,.65)', transition:'color .2s', fontFamily:'var(--font-sans)' }),
  dpBadge: (active) => ({ marginLeft:'auto', fontFamily:'var(--font-mono)', fontSize:'0.58rem', background:'rgba(219,206,165,.12)', color: active ? 'var(--warm-tan)' : 'rgba(219,206,165,.4)', padding:'2px 6px', borderRadius:4 }),

  newChatBtn: { margin:'0.5rem 1rem', padding:'0.65rem 1rem', borderRadius:8, border:'1px dashed rgba(219,206,165,.2)', background:'transparent', color:'rgba(219,206,165,.5)', fontFamily:'var(--font-sans)', fontSize:'0.8rem', display:'flex', alignItems:'center', gap:'0.5rem', cursor:'none', transition:'all .2s', flexShrink:0, position:'relative', zIndex:1 },

  searchWrap: { padding:'0.25rem 1rem 0.5rem', flexShrink:0, position:'relative', zIndex:1 },
  searchInput: { width:'100%', background:'rgba(219,206,165,.06)', border:'1px solid rgba(219,206,165,.1)', borderRadius:7, padding:'0.5rem 0.75rem', fontFamily:'var(--font-mono)', fontSize:'0.72rem', color:'rgba(236,231,209,.7)', outline:'none' },

  groupLabel: { fontFamily:'var(--font-mono)', fontSize:'0.6rem', letterSpacing:'0.14em', textTransform:'uppercase', color:'rgba(219,206,165,.25)', padding:'0.5rem 0.25rem 0.2rem' },

  historyScroll: { flex:1, overflowY:'auto', padding:'0.1rem 1rem 1rem', scrollbarWidth:'thin', scrollbarColor:'rgba(219,206,165,.15) transparent', position:'relative', zIndex:1 },
  histItem: (active) => ({ padding:'0.6rem 0.7rem', borderRadius:7, cursor:'none', transition:'all .2s', display:'flex', alignItems:'flex-start', gap:'0.45rem', border:`1px solid ${active ? 'rgba(219,206,165,.15)' : 'transparent'}`, background: active ? 'rgba(219,206,165,.1)' : 'transparent', marginBottom:2 }),
  histTitle: (active) => ({ fontSize:'0.8rem', fontWeight:500, color: active ? 'var(--parchment-light)' : 'rgba(236,231,209,.6)', whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis', lineHeight:1.3, fontFamily:'var(--font-sans)' }),
  histMeta: { fontFamily:'var(--font-mono)', fontSize:'0.6rem', color:'rgba(219,206,165,.3)', marginTop:3, display:'flex', alignItems:'center', gap:'0.4rem' },
  domainTag: (d) => ({ fontSize:'0.53rem', padding:'1px 4px', borderRadius:3, fontFamily:'var(--font-mono)', background: d==='bio'?'rgba(107,143,113,.2)':d==='cs'?'rgba(138,118,80,.2)':'rgba(142,151,125,.2)', color: d==='bio'?'#8FB898':d==='cs'?'#BFA878':'#A8B09A' }),

  footer: { padding:'1rem 1.2rem', borderTop:'1px solid rgba(219,206,165,.1)', display:'flex', alignItems:'center', gap:'0.7rem', flexShrink:0, position:'relative', zIndex:1 },
  avatar: { width:34, height:34, background:'linear-gradient(135deg,var(--bark),var(--sage))', borderRadius:'50%', display:'flex', alignItems:'center', justifyContent:'center', fontSize:'0.68rem', fontWeight:700, color:'var(--parchment-light)', flexShrink:0 },
  userInfo: { flex:1, minWidth:0 },
  userName: { fontSize:'0.8rem', fontWeight:600, color:'var(--parchment-light)', whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis', fontFamily:'var(--font-sans)' },
  userPlan: { fontFamily:'var(--font-mono)', fontSize:'0.6rem', color:'rgba(219,206,165,.35)' },
  logoutBtn: { background:'none', border:'none', cursor:'none', color:'rgba(219,206,165,.3)', fontSize:'1rem', transition:'color .2s', padding:4 },

  // Delete modal
  deleteOverlay: { position:'absolute', inset:0, background:'rgba(44,36,22,.6)', zIndex:100, display:'flex', alignItems:'center', justifyContent:'center', backdropFilter:'blur(4px)' },
  deleteModal: { background:'var(--bark-deeper)', border:'1px solid rgba(219,206,165,.2)', borderRadius:12, padding:'1.5rem', width:200, textAlign:'center' },
  deleteTitle: { fontSize:'0.88rem', fontWeight:600, color:'var(--parchment-light)', marginBottom:'0.4rem', fontFamily:'var(--font-sans)' },
  deleteSubtitle: { fontSize:'0.72rem', color:'rgba(219,206,165,.4)', marginBottom:'1.2rem', fontFamily:'var(--font-mono)' },
  deleteActions: { display:'flex', gap:'0.5rem' },
  deleteCancelBtn: { flex:1, padding:'0.5rem', borderRadius:7, border:'1px solid rgba(219,206,165,.2)', background:'transparent', color:'rgba(219,206,165,.6)', fontSize:'0.75rem', cursor:'none', fontFamily:'var(--font-sans)' },
  deleteConfirmBtn: { flex:1, padding:'0.5rem', borderRadius:7, border:'none', background:'rgba(181,97,74,.8)', color:'white', fontSize:'0.75rem', cursor:'none', fontWeight:600, fontFamily:'var(--font-sans)' },
}