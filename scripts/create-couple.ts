import 'dotenv/config'
import { pool } from '../lib/db'
import { randomUUID } from 'crypto'
import readline from 'node:readline/promises'

async function prompt(question: string) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout })
  try { return await rl.question(question) } finally { rl.close() }
}

function normalizeName(first: string, last: string) {
  return [first, last].map(s => s.trim()).filter(Boolean).join(' ')
}

async function main() {
  const code = (await prompt('Couple code: ')).trim()
  const aFirst = (await prompt('Partner A first name: ')).trim()
  const aLast = (await prompt('Partner A last name: ')).trim()
  const bFirst = (await prompt('Partner B first name: ')).trim()
  const bLast = (await prompt('Partner B last name: ')).trim()
  const aNet = (await prompt('Partner A NetID (optional): ')).trim() || undefined
  const bNet = (await prompt('Partner B NetID (optional): ')).trim() || undefined

  if (!code || !aFirst || !aLast || !bFirst || !bLast) {
    console.error('Missing required inputs.')
    process.exit(1)
  }

  const coupleId = randomUUID()
  const aId = randomUUID()
  const bId = randomUUID()

  await pool.query('BEGIN')
  try {
    await pool.query(
      'INSERT INTO couples (id, code, created_at) VALUES ($1,$2,NOW()) ON CONFLICT (code) DO NOTHING',
      [coupleId, code]
    )
    const r = await pool.query('SELECT id FROM couples WHERE code=$1', [code])
    const cid = r.rows[0].id
    await pool.query(
      'INSERT INTO participants (id, couple_id, full_name, net_id, started_at) VALUES ($1,$2,$3,$4,NOW()) ON CONFLICT (couple_id, full_name) DO NOTHING',
      [aId, cid, normalizeName(aFirst, aLast), aNet ?? null]
    )
    await pool.query(
      'INSERT INTO participants (id, couple_id, full_name, net_id, started_at) VALUES ($1,$2,$3,$4,NOW()) ON CONFLICT (couple_id, full_name) DO NOTHING',
      [bId, cid, normalizeName(bFirst, bLast), bNet ?? null]
    )
    await pool.query('COMMIT')
    console.log('Created/ensured couple and participants:')
    console.log({ code, coupleId: cid, partnerAId: aId, partnerBId: bId })
  } catch (e) {
    await pool.query('ROLLBACK')
    throw e
  }
}

main().catch(err => { console.error(err); process.exit(1) })

