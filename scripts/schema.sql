-- Core schema for TigerMatch survey (couples, participants, responses)
CREATE TABLE IF NOT EXISTS couples (
  id UUID PRIMARY KEY,
  code TEXT UNIQUE NOT NULL,
  months_together INT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS participants (
  id UUID PRIMARY KEY,
  couple_id UUID REFERENCES couples(id) ON DELETE SET NULL,
  full_name TEXT NOT NULL,
  net_id TEXT,
  email TEXT,
  started_at TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS responses (
  id UUID PRIMARY KEY,
  participant_id UUID REFERENCES participants(id) ON DELETE CASCADE,
  question_id TEXT NOT NULL,
  self_answer TEXT NOT NULL,
  acceptable TEXT NOT NULL,
  importance INT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (participant_id, question_id)
);

CREATE INDEX IF NOT EXISTS idx_responses_participant ON responses(participant_id);
CREATE UNIQUE INDEX IF NOT EXISTS uniq_participant_couple_fullname ON participants(couple_id, full_name);
-- Case-insensitive uniqueness to prevent duplicates differing only by case/whitespace
CREATE UNIQUE INDEX IF NOT EXISTS uniq_participant_couple_fullname_ci
  ON participants (couple_id, lower(trim(full_name)));
