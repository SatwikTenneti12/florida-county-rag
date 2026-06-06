const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

export async function apiFetch(path, options = {}) {
  const { timeoutMs = 60000, ...fetchOptions } = options
  const controller = new AbortController()
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs)

  let response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...fetchOptions,
      signal: fetchOptions.signal || controller.signal,
    })
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error(
        'The request timed out. Check that the API server and embeddings service are responding.',
        { cause: error },
      )
    }
    throw error
  } finally {
    window.clearTimeout(timeoutId)
  }

  const text = await response.text()
  let data = null

  if (text) {
    try {
      data = JSON.parse(text)
    } catch {
      data = { detail: text }
    }
  }

  if (!response.ok) {
    throw new Error(data?.detail || `Request failed with status ${response.status}`)
  }

  return data
}
