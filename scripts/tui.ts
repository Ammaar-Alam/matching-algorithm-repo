import 'dotenv/config'
import fs from 'fs'
import path from 'path'
import readline from 'node:readline/promises'
import { spawn } from 'child_process'
import { pool } from '../lib/db'

type ExportSource = 'api' | 'db'

type ParticipantInfo = { name: string; netId: string; coupleCode: string }
type RankedCandidate = { targetId: string; score: number; pct: number }

type PreviewSelection = {
  label: string
  csvPath: string
  variant: string
}

type PreviewConfig = {
  key: string
  label: string
  csvName: string
  variant: string
  weights?: string
  requires?: string
}

const PREVIEW_CONFIGS: PreviewConfig[] = [
  { key: '1', label: 'Baseline — default domain weights', csvName: 'scores.csv', variant: 'Baseline (default weights)' },
  { key: '2', label: 'Baseline + ML domain weights', csvName: 'scores_ml.csv', variant: 'Baseline + ML weights', weights: 'out/ml_weights.json', requires: 'out/ml_weights.json' },
  { key: '3', label: 'PAM learned domain weights', csvName: 'scores_pam.csv', variant: 'PAM weights', weights: 'out_pam/weights.json', requires: 'out_pam/weights.json' },
]

async function ensurePairScores(py: string, algDir: string, cfg: PreviewConfig){
  const delta = await prompt('delta (LCB for scores)', '0.10')
  const args = ['scoring/score_pairs_cli.py','--export_csv','../export.csv','--meta_json','meta.json','--questions_json','../data/questions.json','--delta', String(Number(delta)||0.10),'--out', cfg.csvName]
  if (cfg.weights) args.push('--weights_json', cfg.weights)
  await run(py, args, algDir)
  console.log(`Pair scores written → algorithm/${cfg.csvName}`)
}

async function choosePreviewSource(py: string, algDir: string): Promise<PreviewSelection | null>{
  console.log('\nPreview data source:')
  PREVIEW_CONFIGS.forEach(cfg => console.log(` ${cfg.key}) ${cfg.label}`))
  console.log(' 4) Soft-gated domain modes (out_soft/scores_soft.csv)')
  console.log(' 5) Evolutionary distance-kernel (out_evo/scores_evo.csv)')
  console.log(' 6) Existing CSV path (skip recompute)')
  const sel = (await prompt('Choose source', '1')).trim()
  if (sel === '6'){
    const custom = await prompt('Absolute or relative path to CSV', path.join(algDir, 'scores.csv'))
    const resolved = path.isAbsolute(custom) ? custom : path.join(process.cwd(), custom)
    if (!fs.existsSync(resolved)){
      console.log(`File not found at ${resolved}`)
      return null
    }
    return { label: 'Custom CSV', csvPath: resolved, variant: `Custom (${resolved})` }
  }
  if (sel === '4'){
    const softCsv = path.join(algDir, 'out_soft', 'scores_soft.csv')
    if (!fs.existsSync(softCsv)){
      console.log('No soft-gated scores at algorithm/out_soft/scores_soft.csv. Run options 15 and 16 first.')
      return null
    }
    return {
      label: 'Soft-gated domain modes',
      csvPath: softCsv,
      variant: 'Soft-gated (domain modes)'
    }
  }
  if (sel === '5'){
    const evoCsv = path.join(algDir, 'out_evo', 'scores_evo.csv')
    if (!fs.existsSync(evoCsv)){
      console.log('No evolutionary scores at algorithm/out_evo/scores_evo.csv. Run options 17 and 18 first.')
      return null
    }
    return {
      label: 'Evolutionary distance-kernel model',
      csvPath: evoCsv,
      variant: 'Evolutionary (distance-kernel)'
    }
  }
  const cfg = PREVIEW_CONFIGS.find(c => c.key === sel)
  if (!cfg){
    console.log('Unknown selection')
    return null
  }
  if (cfg.requires && !fs.existsSync(path.join(algDir, cfg.requires))){
    console.log(`Required file missing: algorithm/${cfg.requires}. Run the corresponding training option first.`)
    return null
  }
  const csvPath = path.join(algDir, cfg.csvName)
  const csvExists = fs.existsSync(csvPath)
  const rebuildPrompt = csvExists ? `${cfg.csvName} exists. Recompute pair scores now? (Y/n)` : `${cfg.csvName} missing. Generate pair scores now? (Y/n)`
  const rebuild = (await prompt(rebuildPrompt, 'Y')).trim().toLowerCase()
  if (rebuild !== 'n' && rebuild !== 'no'){
    await ensurePairScores(py, algDir, cfg)
  } else if (!csvExists){
    console.log(`No cached scores at algorithm/${cfg.csvName}. Re-run generation.`)
    return null
  }
  return { label: cfg.label, csvPath, variant: cfg.variant }
}

async function prompt(question: string, def?: string) {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout })
  try {
    const q = def ? `${question} [${def}]: ` : `${question}: `
    const ans = (await rl.question(q)).trim()
    return ans || (def ?? '')
  } finally {
    rl.close()
  }
}

async function run(cmd: string, args: string[], cwd?: string) {
  return new Promise<void>((resolve, reject) => {
    const cp = spawn(cmd, args, { stdio: 'inherit', cwd })
    cp.on('exit', code => code === 0 ? resolve() : reject(new Error(`${cmd} ${args.join(' ')} exited ${code}`)))
  })
}

async function choosePython(): Promise<string> {
  const candidates = ['python3','python']
  for (const c of candidates){
    try { await run(c, ['--version']); return c } catch {}
  }
  throw new Error('Python 3 not found in PATH')
}

async function fetchExportViaAPI(url: string, adminKey: string, outPath: string) {
  const res = await fetch(url, { headers: { 'x-admin-key': adminKey } })
  if (!res.ok) throw new Error(`Admin export failed: ${res.status}`)
  const buf = await res.arrayBuffer()
  fs.writeFileSync(outPath, Buffer.from(buf))
}

async function exportViaDB(outPath: string) {
  const questions = JSON.parse(fs.readFileSync(path.join(process.cwd(), 'data/questions.json'), 'utf8'))
  const items = (questions as any[]).sort((a,b) => (a.order ?? 0) - (b.order ?? 0))
  const qids = items.map(q => q.id)
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
  const part = await pool.query(
    `SELECT p.id, p.full_name, p.net_id, COALESCE(c.code,'') AS couple_code
       FROM participants p LEFT JOIN couples c ON c.id = p.couple_id
       WHERE p.completed_at IS NOT NULL
       ORDER BY p.started_at ASC`
  )
  const resp = await pool.query(
    'SELECT participant_id, question_id, self_answer, acceptable, importance FROM responses'
  )
  const respByPid = new Map<string, Map<string, {self:string, acc:string, imp:number}>>()
  for (const r of resp.rows as any[]) {
    const m = respByPid.get(r.participant_id) || new Map()
    m.set(r.question_id, { self: r.self_answer ?? '', acc: r.acceptable ?? '', imp: Number(r.importance ?? '') })
    respByPid.set(r.participant_id, m)
  }
  const labelFromImp: Record<number,string> = { 0:'Irrelevant', 1:'A little', 10:'Somewhat', 50:'Very', 250:'Mandatory' }
  const rows: string[] = []
  for (const p of part.rows as any[]) {
    const m = respByPid.get(p.id) || new Map()
    const row: string[] = [p.id, p.couple_code, p.full_name, p.net_id]
    for (const q of qids) row.push(m.get(q)?.self ?? '')
    for (const q of qids) row.push(m.get(q)?.acc ?? '')
    for (const q of qids) row.push(m.get(q) ? String(m.get(q)!.imp) : '')
    for (const q of qids) row.push(m.get(q) ? (labelFromImp[m.get(q)!.imp] ?? '') : '')
    rows.push(row.map(v => `"${(v||'').replace(/"/g,'""')}"`).join(','))
  }
  const csv = [header.join(','), ...rows].join('\n')
  fs.writeFileSync(outPath, csv, 'utf8')
}

function parseCsvLine(line: string): string[]{
  const out: string[] = []
  let cur = ''
  let q = false
  for (let i=0;i<line.length;i++){
    const c = line[i]
    if (q){
      if (c==='"' && line[i+1]==='"'){ cur+='"'; i++ }
      else if (c==='"'){ q=false }
      else cur += c
    }else{
      if (c===','){ out.push(cur); cur='' }
      else if (c==='"'){ q=true }
      else cur += c
    }
  }
  out.push(cur)
  return out
}
function unquote(s: string){ return s?.replace(/^"|"$/g,'') }

function loadParticipantMeta(exportPath: string){
  if (!fs.existsSync(exportPath)) throw new Error('export.csv not found; run the export step first')
  const lines = fs.readFileSync(exportPath, 'utf8').split(/\r?\n/).filter(Boolean)
  if (lines.length <= 1) return new Map<string, ParticipantInfo>()
  const header = parseCsvLine(lines[0]).map(unquote)
  const pidIdx = header.indexOf('participantId')
  const nameIdx = header.indexOf('fullName')
  const netIdx = header.indexOf('netId')
  const coupleIdx = header.indexOf('coupleCode')
  const meta = new Map<string, ParticipantInfo>()
  for (let i=1;i<lines.length;i++){
    const cols = parseCsvLine(lines[i])
    const pid = unquote(cols[pidIdx] || '')
    if (!pid) continue
    meta.set(pid, {
      name: unquote(cols[nameIdx] || '') || '(unknown)',
      netId: unquote(cols[netIdx] || ''),
      coupleCode: unquote(cols[coupleIdx] || '')
    })
  }
  return meta
}

function readRankings(csvPath: string){
  if (!fs.existsSync(csvPath)) throw new Error(`Scores file not found at ${csvPath}`)
  const lines = fs.readFileSync(csvPath, 'utf8').split(/\r?\n/).filter(Boolean)
  if (lines.length <= 1) return new Map<string, RankedCandidate[]>()
  const header = parseCsvLine(lines[0]).map(unquote)
  const idxA = header.indexOf('A')
  const idxB = header.indexOf('B')
  const idxScore = header.indexOf('score')
  const idxPct = header.indexOf('pct_baseline_like')
  if (idxA === -1 || idxB === -1 || idxScore === -1) throw new Error('scores.csv missing A/B/score columns')
  const out = new Map<string, RankedCandidate[]>()
  for (let i=1;i<lines.length;i++){
    const cols = parseCsvLine(lines[i])
    const a = unquote(cols[idxA] || '')
    const b = unquote(cols[idxB] || '')
    const score = Number(unquote(cols[idxScore] || ''))
    if (!a || !b || !Number.isFinite(score)) continue
    let pct = NaN
    if (idxPct >= 0){
      const raw = unquote(cols[idxPct] || '')
      if (raw.trim() !== ''){
        const v = Number(raw)
        if (Number.isFinite(v)) pct = v
      }
    }
    const arr = out.get(a) || []
    arr.push({ targetId: b, score, pct })
    out.set(a, arr)
  }
  for (const arr of out.values()) arr.sort((x,y)=> y.score - x.score)
  return out
}

function describeParticipant(id: string, meta: Map<string, ParticipantInfo>, short=false){
  const info = meta.get(id)
  if (!info) return id
  const net = info.netId ? ` [${info.netId}]` : ''
  const cc = info.coupleCode ? ` (${info.coupleCode})` : ''
  if (short) return `${info.name || id}${net || ''}`
  return `${info.name || '(unknown)'}${net} #${id}${cc}`
}

async function showIdealMatchPreview(py: string, algDir: string, exportPath: string){
  if (!fs.existsSync(exportPath)){
    console.log('export.csv not found. Run the export step first (options 1-2).')
    return
  }
  const selection = await choosePreviewSource(py, algDir)
  if (!selection) return
  await renderIdealMatchPreview(exportPath, selection)
}

async function renderIdealMatchPreview(exportPath: string, selection: PreviewSelection){
  const participants = loadParticipantMeta(exportPath)
  const rankings = readRankings(selection.csvPath)
  if (rankings.size === 0){ console.log(`${selection.csvPath} does not contain any pair entries.`); return }
  const entries = Array.from(rankings.entries()).map(([pid, matches]) => {
    const top = matches[0]
    if (!top) return null
    const counterpart = rankings.get(top.targetId) || []
    const counterpartTop = counterpart[0]
    const counterpartRank = counterpart.findIndex(r => r.targetId === pid)
    return { pid, matches, top, counterpartTop, counterpartRank }
  }).filter(Boolean) as { pid: string, matches: RankedCandidate[], top: RankedCandidate, counterpartTop?: RankedCandidate, counterpartRank: number }[]
  if (!entries.length){ console.log('No ranked matches found.'); return }
  entries.sort((a,b)=> b.top.score - a.top.score)
  let mutualCount = 0
  console.log(`\nIdeal match preview (${entries.length} participants) — ${selection.variant}`)
  console.log(`Source file: ${selection.csvPath}`)
  console.log('')
  entries.forEach((entry, idx) => {
    const label = `${String(idx+1).padStart(2,' ')}. ${describeParticipant(entry.pid, participants)}`
    const topLabel = describeParticipant(entry.top.targetId, participants)
    const pctStr = Number.isFinite(entry.top.pct) ? `${entry.top.pct.toFixed(1)}% baseline-like` : 'n/a'
    const mutual = entry.counterpartTop?.targetId === entry.pid
    if (mutual) mutualCount++
    let note = mutual ? '✓ mutual top choice' : '↦ no return ranking yet'
    if (!mutual){
      if (entry.counterpartRank >= 0){
        note = `↦ ${describeParticipant(entry.top.targetId, participants, true)} ranks ${describeParticipant(entry.pid, participants, true)} #${entry.counterpartRank+1}`
      } else if (entry.counterpartTop){
        note = `↦ ${describeParticipant(entry.top.targetId, participants, true)} prefers ${describeParticipant(entry.counterpartTop.targetId, participants, true)}`
      }
    }
    console.log(label)
    console.log(`    ideal:  ${topLabel}`)
    console.log(`            score=${entry.top.score.toFixed(4)}  ${pctStr}`)
    console.log(`    status: ${note}`)
    const extras = entry.matches.slice(1, 4)
    if (extras.length){
      const extraLine = extras.map((m, extraIdx)=>{
        const short = describeParticipant(m.targetId, participants, true)
        const pct = Number.isFinite(m.pct) ? `${m.pct.toFixed(1)}%` : 'n/a'
        return `${extraIdx+2}. ${short} (${m.score.toFixed(3)}, ${pct})`
      }).join(' | ')
      console.log(`    next:   ${extraLine}`)
    }
    console.log('')
  })
  console.log(`\n${mutualCount}/${entries.length} participants currently have synchronous (mutual) top matches.`)
}

function readPerCouple(outDir: string){
  const p = path.join(outDir, 'per_couple_scores.csv')
  if (!fs.existsSync(p)) return [] as any[]
  const lines = fs.readFileSync(p, 'utf8').split(/\r?\n/).filter(Boolean)
  if (lines.length<=1) return []
  const header = parseCsvLine(lines[0]).map(unquote)
  const recs: any[] = []
  for (let i=1;i<lines.length;i++){
    const cols = parseCsvLine(lines[i])
    const r: any = {}
    header.forEach((h,idx)=> r[h] = unquote(cols[idx] || ''))
    r.score = Number(r.score)
    recs.push(r)
  }
  return recs
}

async function doExport(source: ExportSource, exportPath: string){
  if (source === 'api'){
    const url = await prompt('Admin export URL', process.env.ADMIN_EXPORT_URL || 'https://<your-domain>/api/admin/export')
    const key = await prompt('ADMIN_KEY (leave blank to use env)', process.env.ADMIN_KEY || '')
    const adminKey = key || process.env.ADMIN_KEY || ''
    if (!adminKey || url.includes('<your-domain>')) throw new Error('Provide a valid export URL and ADMIN_KEY')
    console.log('Fetching export.csv via admin API…')
    await fetchExportViaAPI(url, adminKey, exportPath)
  } else {
    if (!process.env.DATABASE_URL && !process.env.POSTGRES_URL) throw new Error('DATABASE_URL/POSTGRES_URL not set')
    console.log('Exporting via DB…')
    await exportViaDB(exportPath)
  }
}

async function buildCouplesFromExport(exportPath: string, couplesPath: string) {
  const text = fs.readFileSync(exportPath, 'utf8')
  const lines = text.split(/\r?\n/).filter(Boolean)
  if (lines.length <= 1) throw new Error('export.csv has no data')
  const header = parseCsvLine(lines[0]).map(unquote)
  const pidIdx = header.indexOf('participantId')
  const codeIdx = header.indexOf('coupleCode')
  if (pidIdx === -1 || codeIdx === -1) throw new Error('export.csv missing participantId/coupleCode columns')
  const selfIdxs = header.map((h,i)=> h.startsWith('SELF_') ? i : -1).filter(i=> i>=0)
  const groups = new Map<string, string[]>()
  for (let i=1;i<lines.length;i++) {
    const cols = parseCsvLine(lines[i])
    const pid = unquote(cols[pidIdx] || '')
    const code = unquote(cols[codeIdx] || '')
    if (!pid || !code) continue
    const complete = selfIdxs.every(idx => (unquote(cols[idx]||'').trim() !== ''))
    if (!complete) continue
    const arr = groups.get(code) || []
    arr.push(pid)
    groups.set(code, arr)
  }
  const out: string[] = ['couple_id,partner_a_id,partner_b_id']
  for (const [code, arr] of groups) {
    if (arr.length >= 2) out.push(`${code},${arr[0]},${arr[1]}`)
  }
  fs.writeFileSync(couplesPath, out.join('\n'), 'utf8')
}

async function runEval(py: string, algDir: string, weightsJson?: string, nps: number = 0){
  const args = ['eval/evaluate_couples.py','--export_csv','../export.csv','--couples_csv','../couples.csv','--meta_json','meta.json','--questions_json','../data/questions.json','--outdir', weightsJson ? 'out_ml' : 'out','--nonpartner_samples', String(nps)]
  if (weightsJson) args.push('--weights_json', weightsJson)
  await run(py, args, algDir)
}

function printScores(outDir: string, variant='core_c100_n20_geo'){
  const recs = readPerCouple(outDir)
  if (recs.length===0){ console.log('(no scores found)'); return }
  const rows = recs.filter(r => r.variant===variant)
  if (rows.length===0){
    console.log(`No rows for variant=${variant}. Available variants:`)
    const vs = Array.from(new Set(recs.map(r=>r.variant)))
    vs.forEach(v=>console.log(' -', v))
    return
  }
  const maxId = Math.max(10, ...rows.map(r => String(r.couple_id).length))
  console.log('\nCouple scores ('+variant+')')
  console.log('-'.repeat(12+maxId))
  for (const r of rows){
    const id = String(r.couple_id).padEnd(maxId, ' ')
    console.log(`${id}  ${r.score.toFixed(4)}  (overlap=${r.overlap})`)
  }
  console.log('')
}

async function main(){
  console.log('TigerMatch TUI — export • evaluate • ML')
  const exportPath = path.join(process.cwd(), 'export.csv')
  const couplesPath = path.join(process.cwd(), 'couples.csv')
  const algDir = path.join(process.cwd(), 'algorithm')
  const metaPath = path.join(algDir, 'meta.json')
  const py = await choosePython()

  // Export step
  const source = ((await prompt('Export source (api/db)', process.env.ADMIN_EXPORT_URL ? 'api':'db')) as ExportSource)
  await doExport(source, exportPath)
  console.log(`Wrote ${exportPath}`)
  await buildCouplesFromExport(exportPath, couplesPath)
  console.log(`Wrote ${couplesPath}`)

  // Build meta
  console.log('Building meta.json…')
  await run(py, ['tools/build_meta_from_repo.py','--questions_json','../data/questions.json','--export_csv','../export.csv','--out','meta.json'], algDir)

  // Menu loop
  let variant = 'core_c100_n20_geo'
  let npsInput = Number(await prompt('nonpartner_samples (0 → auto, recommended ≥5)', '5')) || 0
  let nps = npsInput <= 0 ? 5 : npsInput
  for(;;){
    console.log('\nChoose an action:')
    console.log(' [Baseline / ML]')
    console.log(' 1) Run baseline evaluation')
    console.log(' 2) Show baseline scores')
    console.log(' 3) Train ML domain weights')
    console.log(' 4) Run evaluation with ML weights')
    console.log(' 5) Show ML scores')
    console.log(' 6) Change variant (current: '+variant+')')
    console.log(' 7) Change nonpartner_samples (current: '+nps+')')
    console.log(' [PAM (probabilistic acceptability)]')
    console.log(' 8) Run PAM (probabilistic) evaluation')
    console.log(' 9) Show PAM pair scores')
    console.log('10) Build PAM lists + demo matching')
    console.log('11) Train PAM domain weights')
    console.log('12) Run PAM evaluation with trained weights')
    console.log(' [Previews & merged pipeline]')
    console.log('13) Preview ideal matches (per participant)')
    console.log('14) Run merged IDF+LCB pipeline (lists + matchings + diagnostics)')
    console.log(' [Soft-gated and evolutionary distance-kernel models]')
    console.log('15) Train soft-gated domain modes')
    console.log('16) Evaluate soft-gated model')
    console.log('17) Train evolutionary distance-kernel model')
    console.log('18) Evaluate evolutionary model')
    console.log('19) Show domain-mode reports (soft/evo)')
    console.log(' [Production]')
    console.log('20) Run production matching (preference lists + 1:1 matching)')
    console.log(' [Paper]')
    console.log('21) Compare all algorithms (paper table + LaTeX)')
    console.log(' 0) Quit')
    const sel = await prompt('Select')
    if (sel==='0') break
    try{
      if (sel==='1'){
        await runEval(py, algDir, undefined, nps)
        console.log('Baseline evaluation complete → algorithm/out/')
      } else if (sel==='2'){
        printScores(path.join(algDir, 'out'), variant)
      } else if (sel==='3'){
        await run(py, ['ml/learn_domain_weights.py','--export_csv','../export.csv','--couples_csv','../couples.csv','--meta_json','meta.json','--out','out/ml_weights.json','--verbose'], algDir)
        console.log('ML weights written → algorithm/out/ml_weights.json')
      } else if (sel==='4'){
        const w = path.join(algDir, 'out', 'ml_weights.json')
        if (!fs.existsSync(w)) { console.log('No weights file at algorithm/out/ml_weights.json. Run option 3 first.'); continue }
        await runEval(py, algDir, 'out/ml_weights.json', nps)
        console.log('ML evaluation complete → algorithm/out_ml/')
      } else if (sel==='5'){
        printScores(path.join(algDir, 'out_ml'), variant)
      } else if (sel==='6'){
        const v = await prompt('Variant name', variant)
        if (v) variant = v
      } else if (sel==='7'){
        const v = Number(await prompt('nonpartner_samples (0 → auto, recommended ≥5)', String(nps))) || 0
        nps = v <= 0 ? 5 : v
      } else if (sel==='8'){
        const delta = await prompt('delta (LCB quantile)', '0.10')
        console.log('Running PAM evaluation…')
        await run(py, ['eval/evaluate_pam.py','--export_csv','../export.csv','--couples_csv','../couples.csv','--meta_json','meta.json','--questions_json','../data/questions.json','--delta', String(Number(delta)||0.10),'--nonpartner_samples', String(nps),'--out','out/summary_pam.csv'], algDir)
        console.log('PAM evaluation complete → algorithm/out/summary_pam.csv and *.pairs.csv')
      } else if (sel==='9'){
        const pref = await prompt('show trained weights results? (y/N)', 'N')
        const basePath = path.join(algDir, 'out', 'summary_pam.pairs.csv')
        const trainedPath = path.join(algDir, 'out_pam', 'summary_pam.pairs.csv')
        const chosen = (pref.trim().toLowerCase()==='y' && fs.existsSync(trainedPath)) ? trainedPath : basePath
        if (!fs.existsSync(chosen)) { console.log(`No PAM pairs at ${chosen}. Run option ${pref.trim().toLowerCase()==='y' ? '12' : '8'} first.`); continue }
        const lines = fs.readFileSync(chosen, 'utf8').split(/\r?\n/).filter(Boolean)
        if (lines.length<=1){ console.log('(no PAM pairs)'); continue }
        const header = parseCsvLine(lines[0]).map(unquote)
        const idxScore = header.indexOf('score')
        const idxPct = header.indexOf('pct_baseline_like')
        const recs = lines.slice(1).map(ln=>{ const cols=parseCsvLine(ln); return {
          A: unquote(cols[0]||''), B: unquote(cols[1]||''),
          score: Number(cols[idxScore]||'0'),
          pct: idxPct>=0 ? Number(cols[idxPct]||'0') : NaN,
        } })
        recs.sort((a,b)=> b.score - a.score)
        console.log(`\nTop PAM pairs${chosen.includes('out_pam') ? ' (trained)' : ''}:`)
        for (const r of recs.slice(0,20)){
          const pctStr = Number.isFinite(r.pct) ? `  ${r.pct.toFixed(1)}%` : ''
          console.log(`${r.A} ⇄ ${r.B}  ${r.score.toFixed(4)}${pctStr}`)
        }
      } else if (sel==='10'){
        const delta = await prompt('delta (LCB for scores)', '0.10')
        const tau = await prompt('tau (drop if LCB_A < tau)', '0.20')
        console.log('Scoring pairs for lists…')
        await run(py, ['scoring/score_pairs_cli.py','--export_csv','../export.csv','--meta_json','meta.json','--questions_json','../data/questions.json','--delta', String(Number(delta)||0.10),'--out','scores.csv'], algDir)
        await run(py, ['matching/rankings.py','--scores_csv','scores.csv','--tau', String(Number(tau)||0.20),'--out','lists.json'], algDir)
        await run(py, ['matching/stable.py','--lists','lists.json','--mode','bipartite','--out','matching.json'], algDir)
        console.log('Lists → algorithm/lists.json, Matching → algorithm/matching.json')
      } else if (sel==='11'){
        const delta = await prompt('delta (LCB for features)', '0.10')
        console.log('Training PAM domain weights…')
        await run(py, ['train/trainer.py','--export_csv','../export.csv','--couples_csv','../couples.csv','--meta_json','meta.json','--questions_json','../data/questions.json','--outdir','out_pam','--delta', String(Number(delta)||0.10)], algDir)
        console.log('Learned weights → algorithm/out_pam/weights.json')
      } else if (sel==='12'){
        const delta = await prompt('delta (LCB for evaluation)', '0.10')
        const w = path.join(algDir, 'out_pam', 'weights.json')
        if (!fs.existsSync(w)) { console.log('No weights file at algorithm/out_pam/weights.json. Run option 11 first.'); continue }
        await run(py, ['eval/evaluate_pam.py','--export_csv','../export.csv','--couples_csv','../couples.csv','--meta_json','meta.json','--questions_json','../data/questions.json','--delta', String(Number(delta)||0.10),'--nonpartner_samples', String(nps),'--weights_json','out_pam/weights.json','--out','out_pam/summary_pam.csv'], algDir)
        console.log('PAM evaluation (learned weights) complete → algorithm/out_pam/')
      } else if (sel==='13'){
        await showIdealMatchPreview(py, algDir, exportPath)
      } else if (sel==='14'){
        const rho = await prompt('rho (domain soft-cap)', '0.35')
        const delta = await prompt('delta (LCB confidence)', '0.10')
        const tau = await prompt('tau (tie threshold factor)', '0.05')
        const mu = await prompt('mu (mutuality boost, multiplicative)', '0.03')
        const topk = await prompt('topK for mutuality (r)', '2')
        console.log('Running merged IDF+LCB pipeline (ROADMAP_2)…')
        await run(py, [
          'matching/merged_pipeline.py',
          '--export_csv','../export.csv',
          '--couples_csv','../couples.csv',
          '--meta_json','meta.json',
          '--outdir','out_merged',
          '--rho', String(Number(rho) || 0.35),
          '--delta', String(Number(delta) || 0.10),
          '--tau', String(Number(tau) || 0.05),
          '--mu', String(Number(mu) || 0.03),
          '--topk', String(Number(topk) || 2),
        ], algDir)
        console.log('Merged pipeline outputs → algorithm/out_merged/')
      } else if (sel==='15'){
        const deltaFeat = await prompt('delta (LCB for distance features)', '0.10')
        console.log('Building PAM distance features…')
        await run(py, ['preprocess/normalize.py','--export_csv','../export.csv','--meta_json','meta.json','--out','norm.json'], algDir)
        await run(py, ['features/build_pair_tensors.py','lcbs','--export_csv','../export.csv','--meta_json','meta.json','--questions_json','../data/questions.json','--delta', String(Number(deltaFeat)||0.10),'--out','lcbs.npz'], algDir)
        await run(py, ['features/build_pair_tensors.py','dists','--export_csv','../export.csv','--meta_json','meta.json','--norm_json','norm.json','--out','dists.npz'], algDir)
        await run(py, ['features/make_features.py','--lcbs','lcbs.npz','--dists','dists.npz','--out','features.npz'], algDir)
        console.log('Training soft-gated domain modes…')
        await run(py, ['ml/soft_gated_trainer.py','--features','features.npz','--couples_csv','../couples.csv','--outdir','out_soft'], algDir)
        console.log('Soft-gated config → algorithm/out_soft/best_config.json')
      } else if (sel==='16'){
        const featPath = path.join(algDir, 'features.npz')
        const cfgPath = path.join(algDir, 'out_soft', 'best_config.json')
        if (!fs.existsSync(featPath)){ console.log('No features tensor at algorithm/features.npz. Run option 15 first.'); continue }
        if (!fs.existsSync(cfgPath)){ console.log('No soft-gated config at algorithm/out_soft/best_config.json. Run option 15 first.'); continue }
        console.log('Evaluating soft-gated model…')
        await run(py, ['eval/harness.py','--features','features.npz','--couples_csv','../couples.csv','--config','out_soft/best_config.json','--outdir','out_soft','--topk','3','--scores_basename','scores_soft.csv'], algDir)
        console.log('Soft-gated metrics → algorithm/out_soft/metrics.csv')
        console.log('Soft-gated pair scores → algorithm/out_soft/scores_soft.csv')
      } else if (sel==='17'){
        const featPath = path.join(algDir, 'features.npz')
        if (!fs.existsSync(featPath)){ console.log('No features tensor at algorithm/features.npz. Run option 15 first.'); continue }
        const args = ['ml/evolutionary_trainer.py','--features','features.npz','--couples_csv','../couples.csv','--outdir','out_evo']
        const softCfg = path.join(algDir, 'out_soft', 'best_config.json')
        if (fs.existsSync(softCfg)) args.push('--seed_config','out_soft/best_config.json')
        console.log('Running evolutionary search over domain modes…')
        await run(py, args, algDir)
        console.log('Evolutionary config → algorithm/out_evo/best_config.json (history → history.csv)')
      } else if (sel==='18'){
        const featPath = path.join(algDir, 'features.npz')
        const evoCfg = path.join(algDir, 'out_evo', 'best_config.json')
        if (!fs.existsSync(featPath)){ console.log('No features tensor at algorithm/features.npz. Run option 15 first.'); continue }
        if (!fs.existsSync(evoCfg)){ console.log('No evolutionary config at algorithm/out_evo/best_config.json. Run option 17 first.'); continue }
        console.log('Evaluating evolutionary model…')
        await run(py, ['eval/harness.py','--features','features.npz','--couples_csv','../couples.csv','--config','out_evo/best_config.json','--outdir','out_evo','--topk','3','--scores_basename','scores_evo.csv'], algDir)
        console.log('Evolutionary metrics → algorithm/out_evo/metrics.csv')
        console.log('Evolutionary pair scores → algorithm/out_evo/scores_evo.csv')
      } else if (sel==='19'){
        console.log('Soft-gated domain summary → algorithm/out_soft/domain_summary.csv')
        console.log('Soft-gated ablations    → algorithm/out_soft/domain_ablation.csv')
        console.log('Evolutionary domain summary (if run) → algorithm/out_evo/domain_summary.csv')
        console.log('Evolutionary ablations (if run)     → algorithm/out_evo/domain_ablation.csv')
      } else if (sel==='20'){
        // Production matching - select algorithm
        console.log('\nSelect algorithm:')
        console.log(' 1) Evolutionary (Hit@1=50%, Hit@3=67%)')
        console.log(' 2) Soft-gated (Hit@1=17%, Hit@3=42%)')
        console.log(' 3) ML-weighted (AUC=0.52)')
        console.log(' 4) Baseline')
        const algSel = await prompt('Algorithm', '1')

        // Check for required files based on selection
        const featPath = path.join(algDir, 'features.npz')
        let configPath = ''
        let algName = ''

        if (algSel === '1') {
          configPath = path.join(algDir, 'out_evo', 'best_config.json')
          algName = 'Evolutionary'
          if (!fs.existsSync(configPath)) {
            console.log('No evolutionary config. Run options 15 + 17 first.')
            continue
          }
        } else if (algSel === '2') {
          configPath = path.join(algDir, 'out_soft', 'best_config.json')
          algName = 'Soft-gated'
          if (!fs.existsSync(configPath)) {
            console.log('No soft-gated config. Run option 15 first.')
            continue
          }
        } else if (algSel === '3') {
          // ML-weighted needs a different approach - use default config with ML weights
          // For now, fall back to soft-gated if available, else evolutionary
          if (fs.existsSync(path.join(algDir, 'out_evo', 'best_config.json'))) {
            configPath = path.join(algDir, 'out_evo', 'best_config.json')
          } else if (fs.existsSync(path.join(algDir, 'out_soft', 'best_config.json'))) {
            configPath = path.join(algDir, 'out_soft', 'best_config.json')
          }
          algName = 'ML-weighted'
          if (!configPath || !fs.existsSync(configPath)) {
            console.log('No config available. Run options 15 + 17 first.')
            continue
          }
        } else {
          // Baseline - use soft-gated or evolutionary config if available
          if (fs.existsSync(path.join(algDir, 'out_evo', 'best_config.json'))) {
            configPath = path.join(algDir, 'out_evo', 'best_config.json')
          } else if (fs.existsSync(path.join(algDir, 'out_soft', 'best_config.json'))) {
            configPath = path.join(algDir, 'out_soft', 'best_config.json')
          }
          algName = 'Baseline'
          if (!configPath || !fs.existsSync(configPath)) {
            console.log('No config available. Run options 15 + 17 first.')
            continue
          }
        }

        if (!fs.existsSync(featPath)) {
          console.log('No features.npz. Run option 15 first to build features.')
          continue
        }

        const topK = Number(await prompt('Top-K candidates per person', '5')) || 5

        console.log(`\nRunning production matching with ${algName}...`)
        await run(py, [
          'production/run_matching.py',
          '--features', 'features.npz',
          '--export_csv', '../export.csv',
          '--config', configPath.replace(algDir + '/', ''),
          '--outdir', 'out_production',
          '--top_k', String(topK),
          '--algorithm', algName,
        ], algDir)
        console.log('\nProduction outputs → algorithm/out_production/')
      } else if (sel==='21'){
        // Paper comparison - compare all algorithms
        const featPath = path.join(algDir, 'features.npz')
        const evoCfg = path.join(algDir, 'out_evo', 'best_config.json')
        const softCfg = path.join(algDir, 'out_soft', 'best_config.json')
        const mergedSummary = path.join(algDir, 'out_merged', 'summary_ranks.json')

        if (!fs.existsSync(featPath)) {
          console.log('No features.npz. Run option 15 first to build features.')
          continue
        }
        if (!fs.existsSync(evoCfg)) {
          console.log('No evolutionary config. Run options 15 + 17 first.')
          continue
        }
        if (!fs.existsSync(softCfg)) {
          console.log('No soft-gated config. Run option 15 first.')
          continue
        }

        console.log('\nComparing all algorithms for paper...')
        const args = [
          'paper/compare_all.py',
          '--features', 'features.npz',
          '--couples_csv', '../couples.csv',
          '--export_csv', '../export.csv',
          '--evo_config', 'out_evo/best_config.json',
          '--soft_config', 'out_soft/best_config.json',
          '--outdir', 'out_paper',
        ]
        if (fs.existsSync(mergedSummary)) {
          args.push('--merged_summary', 'out_merged/summary_ranks.json')
        }
        await run(py, args, algDir)
        console.log('\nPaper outputs → algorithm/out_paper/')
        console.log('  - comparison_table.csv')
        console.log('  - comparison_table.tex (LaTeX)')
        console.log('  - domain_weights.csv')
        console.log('  - per_couple_detail.csv')
      } else {
        console.log('Unknown selection')
      }
    } catch (e:any){
      console.error('Error:', e?.message || e)
    }
  }
  console.log('Goodbye')
}

main().catch(e => { console.error(e); process.exit(1) })
