'use client'

import { useEffect, useRef, useState } from 'react'
import { Camera, Check, ChevronLeft, Code2, Info, RotateCcw, Sparkles, Trash2, Video, VideoOff, Wifi } from 'lucide-react'

const demoWords = ['سلام', 'ممنون', 'کمک', 'آب', 'دوست', 'خوب']

export default function Page() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [isDetecting, setIsDetecting] = useState(false)
  const [developerMode, setDeveloperMode] = useState(false)
  const [word, setWord] = useState('آماده')
  const [confidence, setConfidence] = useState(0)
  const [sentence, setSentence] = useState<string[]>([])
  const [status, setStatus] = useState('آماده شروع')
  const [cameraError, setCameraError] = useState('')
  const [serverStatus, setServerStatus] = useState<'unknown' | 'online' | 'demo'>('unknown')

  const inferenceApiUrl = process.env.NEXT_PUBLIC_INFERENCE_API_URL

  useEffect(() => () => streamRef.current?.getTracks().forEach((track) => track.stop()), [])

  useEffect(() => {
    if (!isDetecting) return
    const interval = window.setInterval(() => {
      const next = demoWords[Math.floor(Math.random() * demoWords.length)]
      setWord(next)
      setConfidence(Math.floor(86 + Math.random() * 12))
      setStatus(developerMode ? 'حالت توسعه‌دهنده فعال' : 'تشخیص فعال')
    }, 2200)
    return () => window.clearInterval(interval)
  }, [isDetecting, developerMode])

  const toggleDetection = async () => {
    if (isDetecting) {
      streamRef.current?.getTracks().forEach((track) => track.stop())
      streamRef.current = null
      if (videoRef.current) videoRef.current.srcObject = null
      setIsDetecting(false)
      setWord('آماده')
      setConfidence(0)
      setStatus('آماده شروع')
      return
    }
    setCameraError('')
    setStatus('در حال آماده‌سازی دوربین...')
    try {
      const stream = await navigator.mediaDevices?.getUserMedia({ video: { facingMode: 'user' }, audio: false })
      if (videoRef.current && stream) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        streamRef.current = stream
        setIsDetecting(true)
        setStatus('تشخیص فعال')

        if (inferenceApiUrl) {
          try {
            const response = await fetch(`${inferenceApiUrl.replace(/\/$/, '')}/health`)
            const health = await response.json()
            setServerStatus(health.demo_mode ? 'demo' : 'online')
          } catch {
            setServerStatus('unknown')
          }
        }
      } else throw new Error('camera')
    } catch {
      setCameraError('برای شروع، دسترسی دوربین را فعال کنید')
      setStatus('دوربین در دسترس نیست')
    }
  }

  const addWord = () => {
    if (!isDetecting || word === 'آماده') return
    setSentence((current) => current.at(-1) === word ? current : [...current, word].slice(-8))
    setStatus('کلمه به جمله اضافه شد')
  }

  return (
    <main className="app-shell" dir="rtl">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <section className="phone-frame" aria-label="مترجم زبان اشاره فارسی">
        <header className="topbar">
          <button className="icon-button" aria-label="راهنما"><Info size={19} /></button>
          <div className="brand-mark"><Sparkles size={17} /><span>دست‌یار</span></div>
          <button className={`developer-button ${developerMode ? 'active' : ''}`} onClick={() => setDeveloperMode((value) => !value)} aria-pressed={developerMode}>
            <Code2 size={16} /> توسعه‌دهنده
          </button>
        </header>

        <div className="intro">
          <p className="eyebrow"><span className="live-dot" /> مترجم بلادرنگ زبان اشاره</p>
          <h1>با دست‌هایتان <em>صحبت کنید</em></h1>
          <p className="subtitle">دوربین را روشن کنید و زبان اشاره را به متن تبدیل کنید.</p>
        </div>

        <section className={`camera-card ${isDetecting ? 'running' : ''}`} aria-label="نمای دوربین">
          <video ref={videoRef} className="camera-video" muted playsInline aria-label="پیش‌نمایش دوربین جلو" />
          {!isDetecting && <div className="camera-placeholder"><div className="camera-icon"><Camera size={29} /></div><strong>برای شروع آماده‌اید؟</strong><span>دوربین را روشن کنید تا تشخیص آغاز شود</span></div>}
          {isDetecting && <><div className="scan-line" /><div className="camera-badge"><span className="live-dot" /> LIVE</div>{developerMode && <div className="skeleton-overlay" aria-hidden="true"><div className="hand-points points-a" /><div className="hand-points points-b" /><div className="pose-line pose-one" /><div className="pose-line pose-two" /></div>}</>}
          <div className="camera-corners" />
        </section>

        {cameraError && <p className="camera-error" role="alert">{cameraError}</p>}

        <section className="prediction-section" aria-live="polite">
          <div className="prediction-label"><span>کلمه فعلی</span>{developerMode && <span className="confidence"><Check size={13} /> اطمینان {confidence}%</span>}</div>
          <button className="prediction-word" onClick={addWord} aria-label={`افزودن ${word} به جمله`}>{word}</button>
          <div className="progress-track"><span style={{ width: `${isDetecting ? confidence : 0}%` }} /></div>
        </section>

        <section className="sentence-card" aria-label="جمله شما">
          <div className="section-heading"><div><span className="section-kicker">خروجی شما</span><h2>جمله شما</h2></div><button className="clear-button" onClick={() => setSentence([])} disabled={!sentence.length}><Trash2 size={15} /> پاک کردن</button></div>
          <div className={`sentence-content ${sentence.length ? 'has-words' : ''}`}>{sentence.length ? sentence.join('  ') : 'کلمات شناسایی‌شده اینجا نمایش داده می‌شوند'}</div>
          <div className="word-count">{sentence.length} از ۸ کلمه</div>
        </section>

        <div className="status-row"><div className="status-text"><span className={`status-indicator ${isDetecting ? 'active' : ''}`} /> {status}</div><span className="connection"><Wifi size={14} /> {serverStatus === 'online' ? 'سرور متصل' : serverStatus === 'demo' ? 'حالت آزمایشی' : inferenceApiUrl ? 'سرور بررسی نشده' : 'مدل محلی'}</span></div>
        <button className={`start-button ${isDetecting ? 'stop' : ''}`} onClick={toggleDetection}>{isDetecting ? <><VideoOff size={19} /> توقف تشخیص</> : <><Video size={19} /> شروع تشخیص</>}</button>
        <p className="footer-note"><RotateCcw size={12} /> برای بهترین نتیجه، در محیطی با نور مناسب قرار بگیرید</p>
      </section>
    </main>
  )
}
