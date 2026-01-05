import 'dotenv/config'
import { pool } from '../lib/db'
import fs from 'fs'
import path from 'path'

/**
 * Writes couples.csv with columns: couple_id,partner_a_id,partner_b_id
 * Pairs the first two participants per couple_code by start time.
 */
async function main() {
  const { rows } = await pool.query(
    `SELECT c.code AS couple_code, p.id AS participant_id, p.started_at
       FROM participants p
       LEFT JOIN couples c ON c.id = p.couple_id
       WHERE c.code IS NOT NULL AND p.completed_at IS NOT NULL
       ORDER BY c.code ASC, p.started_at ASC`
  )
  const byCode = new Map<string, string[]>()
  for (const r of rows as any[]) {
    const arr = byCode.get(r.couple_code) || []
    arr.push(r.participant_id)
    byCode.set(r.couple_code, arr)
  }
  const out: string[] = [ 'couple_id,partner_a_id,partner_b_id' ]
  for (const [code, ids] of byCode.entries()) {
    if (ids.length >= 2) {
      out.push(`${code},${ids[0]},${ids[1]}`)
    }
  }
  const outPath = path.join(process.cwd(), 'couples.csv')
  fs.writeFileSync(outPath, out.join('\n'), 'utf8')
  console.log(`Wrote ${outPath}`)
}

main().catch(e => { console.error(e); process.exit(1) })
