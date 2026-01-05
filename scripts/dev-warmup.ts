// Warm up Next.js dev by pre-hitting common routes so they compile once.
const HOST = process.env.DEV_HOST || 'http://localhost:3000'

async function waitForServer(timeoutMs = 30000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(HOST, { cache: 'no-store' })
      if (res.ok || res.status === 404) return true
    } catch {}
    await new Promise(r => setTimeout(r, 1000))
  }
  return false
}

async function warm(url: string, init?: RequestInit) {
  try {
    const res = await fetch(HOST + url, init)
    console.log(`[warm] ${url} -> ${res.status}`)
  } catch (e) {
    console.log(`[warm] ${url} -> error`)
  }
}

async function main() {
  const up = await waitForServer()
  if (!up) {
    console.log('Dev server not detected on', HOST)
    process.exit(1)
  }
  // pages
  await warm('/')
  await warm('/start')
  await warm('/survey/1')
  // APIs
  await warm('/api/questions')
  await warm('/api/register', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) })
  await warm('/api/response', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) })
  console.log('Warmup complete')
}

main()

