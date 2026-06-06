import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import ReCAPTCHA from 'react-google-recaptcha'
import { LogIn, KeyRound } from 'lucide-react'
import { apiFetch } from '../api'

const RECAPTCHA_SITE_KEY = import.meta.env.VITE_RECAPTCHA_SITE_KEY || ''
const RECAPTCHA_DISABLED = import.meta.env.VITE_DISABLE_RECAPTCHA === 'true'
const CAPTCHA_REQUIRED = !RECAPTCHA_DISABLED

export default function Login({ onLogin }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [captchaValue, setCaptchaValue] = useState(null)

  const [step, setStep] = useState('login')
  const [verificationCode, setVerificationCode] = useState('')

  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const navigate = useNavigate()

  const handleLogin = async (e) => {
    e.preventDefault()

    if (CAPTCHA_REQUIRED && !captchaValue) {
      setError('Please complete the CAPTCHA to prove you are human.')
      return
    }

    setIsLoading(true)
    setError('')
    setMessage('')

    try {
      const data = await apiFetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, captcha_token: captchaValue || 'local-dev-captcha-disabled' })
      })

      onLogin(data.access_token, data.user)
      navigate('/')

    } catch (err) {
      if (err.message === "Please verify your email before logging in.") {
        setStep('verify')
        return
      }
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleVerify = async (e) => {
    e.preventDefault()
    setIsLoading(true)
    setError('')
    setMessage('')
    try {
      const data = await apiFetch('/api/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, code: verificationCode })
      })

      onLogin(data.access_token, data.user)
      navigate('/')

    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleResendCode = async () => {
    setIsLoading(true)
    setError('')
    setMessage('')
    try {
      await apiFetch('/api/resend_code', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      })

      setMessage('A new verification code has been sent to your email.')

    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  if (step === 'verify') {
    return (
      <div className="animate-fade-in" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '80vh', padding: '2rem 0' }}>
        <div className="glass-panel" style={{ width: '100%', maxWidth: '380px', textAlign: 'center' }}>
          <h2 style={{ marginBottom: '1rem' }}>Account Not Verified</h2>
          <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>
            Please enter the 6-digit verification code sent to <strong>{email}</strong>.
          </p>

          {error && (
            <div style={{ background: 'rgba(248, 113, 113, 0.2)', border: '1px solid #f87171', color: '#f87171', padding: '0.75rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
              {error}
            </div>
          )}

          {message && (
            <div style={{ background: 'rgba(74, 222, 128, 0.15)', border: '1px solid #4ade80', color: '#86efac', padding: '0.75rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
              {message}
            </div>
          )}

          {CAPTCHA_REQUIRED && !RECAPTCHA_SITE_KEY && (
            <div style={{ background: 'rgba(248, 113, 113, 0.2)', border: '1px solid #f87171', color: '#f87171', padding: '0.75rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
              ReCAPTCHA is not configured. Set VITE_RECAPTCHA_SITE_KEY before using login.
            </div>
          )}

          <form onSubmit={handleVerify} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div className="input-group" style={{ marginBottom: 0 }}>
              <input
                type="text"
                className="text-area"
                placeholder="000000"
                style={{ minHeight: '60px', padding: '0.75rem', fontSize: '2rem', textAlign: 'center', letterSpacing: '0.5rem' }}
                value={verificationCode}
                onChange={(e) => setVerificationCode(e.target.value)}
                maxLength={6}
                required
              />
            </div>

            <button type="submit" className="primary-button" style={{ width: '100%', marginTop: '1rem' }} disabled={isLoading}>
              {isLoading ? 'Verifying...' : <><KeyRound size={18} /> Verify Account</>}
            </button>

            <button
              type="button"
              onClick={handleResendCode}
              style={{ background: 'transparent', border: 'none', color: 'var(--sky-light)', cursor: 'pointer', marginTop: '1rem' }}
              disabled={isLoading}
            >
              Didn't receive a code? Resend
            </button>
          </form>
        </div>
      </div>
    )
  }

  return (
    <div className="animate-fade-in" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '80vh' }}>
      <div className="glass-panel" style={{ width: '100%', maxWidth: '340px' }}>
        <h2 style={{ textAlign: 'center', marginBottom: '2rem' }}>Welcome Back</h2>

        {error && (
          <div style={{ background: 'rgba(248, 113, 113, 0.2)', border: '1px solid #f87171', color: '#f87171', padding: '0.75rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
            {error}
          </div>
        )}

        {CAPTCHA_REQUIRED && !RECAPTCHA_SITE_KEY && (
          <div style={{ background: 'rgba(248, 113, 113, 0.2)', border: '1px solid #f87171', color: '#f87171', padding: '0.75rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
            ReCAPTCHA is not configured. Set VITE_RECAPTCHA_SITE_KEY before using login.
          </div>
        )}

        <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="input-group" style={{ marginBottom: 0 }}>
            <label className="input-label">Email</label>
            <input
              type="email"
              className="text-area"
              style={{ minHeight: '45px', padding: '0.75rem' }}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="input-group" style={{ marginBottom: 0 }}>
            <label className="input-label">Password</label>
            <input
              type="password"
              className="text-area"
              style={{ minHeight: '45px', padding: '0.75rem' }}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <div style={{ display: 'flex', justifyContent: 'center', margin: '1rem 0' }}>
            {CAPTCHA_REQUIRED && RECAPTCHA_SITE_KEY && (
              <ReCAPTCHA
                sitekey={RECAPTCHA_SITE_KEY}
                onChange={(val) => { setCaptchaValue(val); setError(''); }}
                theme="dark"
              />
            )}
          </div>

          <button type="submit" className="primary-button" style={{ width: '100%' }} disabled={isLoading || (CAPTCHA_REQUIRED && !RECAPTCHA_SITE_KEY)}>
            {isLoading ? 'Signing In...' : <><LogIn size={18} /> Sign In</>}
          </button>
        </form>

        <p style={{ textAlign: 'center', marginTop: '2rem', color: 'var(--text-muted)' }}>
          Don't have an account? <Link to="/signup" style={{ color: 'var(--sky-light)', textDecoration: 'none' }}>Sign up</Link>
        </p>
      </div>
    </div>
  )
}
