
import 'dotenv/config'
import { pool } from '../lib/db'
import fs from 'fs'
import path from 'path'
import questionsRaw from '../data/questions.json'

/**
 * Exports wide CSV with columns:
 * participantId, coupleCode, fullName, netId, SELF_Q1..50, ACC_Q1..50, IMP_Q1..50
 */
async function main() {
  const items = (questionsRaw as any[]).sort((a,b) => (a.order ?? 0) - (b.order ?? 0))
  const qids = items.map(q => q.id)
  // optionally include IMC item in export when enabled to match collection
  if (process.env.ENABLE_IMC === 'true') {
    const mid = Math.floor(qids.length / 2)
    qids.splice(mid, 0, 'IMC1')
  }

  const header = [
    'participantId','coupleCode','fullName','netId',
    ...qids.map(q => `SELF_${q}`),
    ...qids.map(q => `ACC_${q}`),
    ...qids.map(q => `IMP_${q}`),
    ...qids.map(q => `IMP_LABEL_${q}`)
  ]
  const rows: string[][] = []

  const { rows: participants } = await pool.query(`
    SELECT p.id, p.full_name, p.net_id, COALESCE(c.code,'') AS couple_code
    FROM participants p LEFT JOIN couples c ON c.id = p.couple_id
    WHERE p.completed_at IS NOT NULL
    ORDER BY p.started_at ASC
  `)

  const { rows: respRows } = await pool.query(
    'SELECT participant_id, question_id, self_answer, acceptable, importance FROM responses'
  )
  const respByPid = new Map<string, Map<string, {self:string, acc:string, imp:number}>>()
  for (const r of respRows as any[]) {
    const m = respByPid.get(r.participant_id) || new Map()
    m.set(r.question_id, { self: r.self_answer ?? '', acc: r.acceptable ?? '', imp: Number(r.importance ?? '') })
    respByPid.set(r.participant_id, m)
  }

  function labelFromImportance(n: number | undefined): string {
    if (n === undefined || n === null) return ''
    const map: Record<number, string> = { 0: 'Irrelevant', 1: 'A little', 10: 'Somewhat', 50: 'Very', 250: 'Mandatory' }
    return map[Number(n)] ?? ''
  }

  for (const p of participants as any[]) {
    const m = respByPid.get(p.id) || new Map()
    const row: string[] = [p.id, p.couple_code, p.full_name, p.net_id]
    for (const q of qids) row.push(m.get(q)?.self ?? '')
    for (const q of qids) row.push(m.get(q)?.acc ?? '')
    for (const q of qids) row.push(m.get(q) ? String(m.get(q)!.imp) : '')
    for (const q of qids) row.push(m.get(q) ? labelFromImportance(m.get(q)!.imp) : '')
    rows.push(row)
  }

  const outPath = path.join(process.cwd(), 'export.csv')
  const csv = [header.join(','), ...rows.map(r => r.map(v => `"${(v||'').replace(/"/g,'""')}"`).join(','))].join('\n')
  fs.writeFileSync(outPath, csv, 'utf8')
  console.log(`Wrote ${outPath}`)
}

main().catch(e => { console.error(e); process.exit(1) })
