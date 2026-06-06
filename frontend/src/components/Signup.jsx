import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import ReCAPTCHA from 'react-google-recaptcha'
import { UserPlus, KeyRound } from 'lucide-react'
import { apiFetch } from '../api'

const RECAPTCHA_SITE_KEY = import.meta.env.VITE_RECAPTCHA_SITE_KEY || ''
const RECAPTCHA_DISABLED = import.meta.env.VITE_DISABLE_RECAPTCHA === 'true'
const CAPTCHA_REQUIRED = !RECAPTCHA_DISABLED

export default function Signup({ onLogin }) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [county, setCounty] = useState('')
  const [captchaValue, setCaptchaValue] = useState(null)

  // Verification State
  const [step, setStep] = useState('signup')
  const [verificationCode, setVerificationCode] = useState('')

  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [counties, setCounties] = useState([])
  const [isLoadingCounties, setIsLoadingCounties] = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    let isMounted = true

    async function loadCounties() {
      try {
        const data = await apiFetch('/api/counties')
        if (isMounted) {
          setCounties(data.counties || [])
        }
      } catch (err) {
        if (isMounted) {
          setError(`Could not load county list: ${err.message}`)
        }
      } finally {
        if (isMounted) {
          setIsLoadingCounties(false)
        }
      }
    }

    loadCounties()

    return () => {
      isMounted = false
    }
  }, [])

  const handleSignup = async (e) => {
    e.preventDefault()

    // Password strength validation
    const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/
    if (!passwordRegex.test(password)) {
      setError('Password must be at least 8 characters and include uppercase, lowercase, number, and symbol.')
      return
    }

    if (CAPTCHA_REQUIRED && !captchaValue) {
      setError('Please complete the CAPTCHA to prove you are human.')
      return
    }

    setIsLoading(true)
    setError('')
    setMessage('')
    try {
      await apiFetch('/api/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, password, county, captcha_token: captchaValue || 'local-dev-captcha-disabled' })
      })

      setStep('verify')

    } catch (err) {
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
          <h2 style={{ marginBottom: '1rem' }}>Check Your Email</h2>
          <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>
            We sent a 6-digit verification code to <strong>{email}</strong>.
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
              ReCAPTCHA is not configured. Set VITE_RECAPTCHA_SITE_KEY before creating accounts.
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
    <div className="animate-fade-in" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '80vh', padding: '2rem 0' }}>
      <div className="glass-panel" style={{ width: '100%', maxWidth: '380px' }}>
        <h2 style={{ textAlign: 'center', marginBottom: '2rem' }}>Create Account</h2>

        {error && (
          <div style={{ background: 'rgba(248, 113, 113, 0.2)', border: '1px solid #f87171', color: '#f87171', padding: '0.75rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
            {error}
          </div>
        )}

        {CAPTCHA_REQUIRED && !RECAPTCHA_SITE_KEY && (
          <div style={{ background: 'rgba(248, 113, 113, 0.2)', border: '1px solid #f87171', color: '#f87171', padding: '0.75rem', borderRadius: '8px', marginBottom: '1.5rem' }}>
            ReCAPTCHA is not configured. Set VITE_RECAPTCHA_SITE_KEY before creating accounts.
          </div>
        )}

        <form onSubmit={handleSignup} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div className="input-group" style={{ marginBottom: 0 }}>
            <label className="input-label">Full Name</label>
            <input
              type="text"
              className="text-area"
              style={{ minHeight: '45px', padding: '0.75rem' }}
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>

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
            <label className="input-label">Florida County Affiliation (Optional)</label>
            <select
              className="text-area"
              style={{ minHeight: '45px', padding: '0.75rem' }}
              value={county}
              onChange={(e) => setCounty(e.target.value)}
            >
              <option value="">{isLoadingCounties ? 'Loading counties...' : 'No county affiliation'}</option>
              {counties.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
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
            {isLoading ? 'Creating...' : <><UserPlus size={18} /> Sign Up</>}
          </button>
        </form>

        <p style={{ textAlign: 'center', marginTop: '2rem', color: 'var(--text-muted)' }}>
          Already have an account? <Link to="/login" style={{ color: 'var(--sky-light)', textDecoration: 'none' }}>Sign in</Link>
        </p>
      </div>
    </div>
  )
}
