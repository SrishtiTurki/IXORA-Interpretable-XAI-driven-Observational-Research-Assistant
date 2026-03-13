import { useEffect, useRef } from 'react'

export default function Cursor() {
  const dotRef  = useRef(null)
  const ringRef = useRef(null)

  useEffect(() => {
    const dot  = dotRef.current
    const ring = ringRef.current
    let rx = 0, ry = 0

    const onMove = (e) => {
      dot.style.left  = e.clientX + 'px'
      dot.style.top   = e.clientY + 'px'
      setTimeout(() => {
        ring.style.left = e.clientX + 'px'
        ring.style.top  = e.clientY + 'px'
      }, 70)
    }

    const onEnter = () => { dot.classList.add('hover'); ring.classList.add('hover') }
    const onLeave = () => { dot.classList.remove('hover'); ring.classList.remove('hover') }

    document.addEventListener('mousemove', onMove)

    const addListeners = () => {
      document.querySelectorAll('a,button,[data-hover]').forEach(el => {
        el.addEventListener('mouseenter', onEnter)
        el.addEventListener('mouseleave', onLeave)
      })
    }

    const observer = new MutationObserver(addListeners)
    observer.observe(document.body, { childList: true, subtree: true })
    addListeners()

    return () => {
      document.removeEventListener('mousemove', onMove)
      observer.disconnect()
    }
  }, [])

  return (
    <>
      <div ref={dotRef}  style={styles.dot}  id="cursor" />
      <div ref={ringRef} style={styles.ring} id="cursorRing" />
      <style>{`
        #cursor { width:9px;height:9px;background:var(--bark);border-radius:50%;position:fixed;top:0;left:0;pointer-events:none;z-index:9999;transform:translate(-50%,-50%);transition:width .2s,height .2s,background .2s;mix-blend-mode:multiply; }
        #cursorRing { width:32px;height:32px;border:1.5px solid var(--bark);border-radius:50%;position:fixed;top:0;left:0;pointer-events:none;z-index:9998;transform:translate(-50%,-50%);transition:all .15s ease-out;opacity:.4; }
        #cursor.hover { width:16px;height:16px;background:var(--sage); }
        #cursorRing.hover { width:52px;height:52px;opacity:.2; }
      `}</style>
    </>
  )
}

const styles = { dot: {}, ring: {} }
