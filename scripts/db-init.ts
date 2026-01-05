import 'dotenv/config'
import { pool } from '../lib/db'

async function main() {
  // create tables if not exist
  await pool.query(`
  CREATE TABLE IF NOT EXISTS couples (
    id UUID PRIMARY KEY,
    code TEXT UNIQUE NOT NULL,
    months_together INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
  );

  CREATE TABLE IF NOT EXISTS participants (
    id UUID PRIMARY KEY,
    couple_id UUID REFERENCES couples(id) ON DELETE SET NULL,
    full_name TEXT NOT NULL,
    net_id TEXT,
    email TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
  );

  CREATE TABLE IF NOT EXISTS responses (
    id UUID PRIMARY KEY,
    participant_id UUID REFERENCES participants(id) ON DELETE CASCADE,
    question_id TEXT NOT NULL,
    self_answer TEXT NOT NULL,
    acceptable TEXT NOT NULL,
    importance INT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (participant_id, question_id)
  );

  CREATE INDEX IF NOT EXISTS idx_responses_participant ON responses(participant_id);
  CREATE UNIQUE INDEX IF NOT EXISTS uniq_participant_couple_fullname ON participants(couple_id, full_name);
  CREATE UNIQUE INDEX IF NOT EXISTS uniq_participant_couple_fullname_ci ON participants(couple_id, lower(trim(full_name)));
  `)
  console.log('DB schema ensured')
}

main().then(() => process.exit(0)).catch(err => { console.error(err); process.exit(1) })
