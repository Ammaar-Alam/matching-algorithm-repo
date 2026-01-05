# IW Results — Current Six‑Couple Dataset (All Algorithm Variants)

## 1) What dataset we are evaluating

### 1.1 Counts

- Participants: **12** (`export.csv` rows)
- Couples: **6** (`couples.csv` rows)
- Items: **50** (`data/questions.json`)

### 1.2 Ground-truth couples (anonymized)

Participant IDs are truncated to the first 8 characters.

| couple_id | partner_a_id | partner_b_id |
|---|---|---|
| PROTO | c2b8317c | 7de011b7 |
| 0260 | 28e08bc8 | 562431c5 |
| E047 | 52e99f9a | 1f7ba3a6 |
| TN10 | d795f6be | 085e8968 |
| MY85 | fe5827c5 | f82ca611 |
| TA170 | 36bfb1da | 11a90732 |

---

## 2) Quick descriptive stats (what the dataset “looks like”)

### 2.1 Domain / response-type composition (50 items)

From `data/questions.json` / `algorithm/meta.json`:
- Domains (counts): Values (21), Personality (10), Lifestyle (6), Social (4), Communication (3), Deal‑breakers (3), Friendship (2), Importance Weighting (1).
- Response types (counts): Likert6 (21), Likert7 (10), Likert5 (6), MC_SINGLE (11), MC_MULTI (2).

### 2.2 Q47 top‑priority selections (12 participants total)

Q47 is “select up to 3 priorities”. Observed option counts:
- Shared values: 9
- Communication style: 8
- Intellectual connection: 6
- Daily routine compatibility: 5
- Activity interests: 3
- Similar background: 3 (ignored by domain mapping in current baseline)

### 2.3 Importance distribution (600 item-responses total = 12 × 50)

Observed `IMP_LABEL_*` distribution:
- A little: 189 (31.5%)
- Somewhat: 181 (30.2%)
- Very: 147 (24.5%)
- Mandatory: 42 (7.0%)
- Irrelevant: 41 (6.8%)

### 2.4 Likert tolerance distribution (444 Likert acceptability entries)

Observed `ACC_*` tolerance (`tol`) distribution across Likert items:
- ±2 steps: 165 (37.2%)
- ±1 step: 157 (35.4%)
- Any answer: 59 (13.3%)
- MISSING: 63 (14.2%)

### 2.5 “Overlap” in the baseline family

For `algorithm/eval/evaluate_couples.py` baseline variants, overlap counts for the main variant (`core_c100_n20_geo`) are:
- overlaps per couple: `[46, 46, 46, 46, 46, 42]` (min 42, median 46, max 46)

---

## 3) Important metric convention note (AUC)

Several repo scripts compute “AUC via Mann–Whitney U” with ranks assigned in **descending** score order, which yields:
- `auc_repo` = **P(true score < random score)** (plus ties at 0.5)
- `auc_std` = **P(true score > random score)** (plus ties at 0.5)

These relate by `auc_std = 1 - auc_repo`.

For consistency with existing CSV outputs, this document reports both where applicable.

---

## 4) Baseline (hard acceptability) results — all preregistered variants

**Source:** `algorithm/eval/evaluate_couples.py`  
**Outputs:** `algorithm/out/summary_by_variant.csv`, `algorithm/out/per_couple_scores.csv`

### 4.1 Variant summary table

From `algorithm/out/summary_by_variant.csv` (AUC shown as both repo and standard):

```
variant	n_couples	auc_repo	auc_std	lift	median_rank	true_mean	rnd_mean
symm_c100_n20_mean	6	0.500000	0.500000	-2.509280047139093	12	59.048864388240666	61.55814443537976
core_c140_n20_geo	6	0.488889	0.511111	-2.1717896003023114	12	52.77629081576313	54.948080416065444
core_c100_n20_geo	6	0.488889	0.511111	-2.5821323119479445	12	58.30107133694427	60.88320364889221
core_c100_n15_geo	6	0.488889	0.511111	-2.5821323119479445	12	58.30107133694427	60.88320364889221
core_c100_n25_geo	6	0.488889	0.511111	-2.5821323119479445	12	58.30107133694427	60.88320364889221
core_c60_n20_geo	6	0.483333	0.516667	-2.5738346731975383	11	64.24449220852145	66.81832688171899
```

### 4.2 Per-couple scores (all variants)

Raw (exact) file content: `algorithm/out/per_couple_scores.csv`

```
variant,couple_id,score,overlap,sA,sB
core_c100_n20_geo,PROTO,75.9446892660642,46,0.8484664807855498,0.9693339722089123
core_c100_n20_geo,0260,54.810639560156304,46,0.572558478712066,0.8449573757415624
core_c100_n20_geo,E047,64.20488972671252,46,0.8427886536156612,0.7395635963571909
core_c100_n20_geo,TN10,79.42196690875878,46,0.8909730363423212,0.9952339524151756
core_c100_n20_geo,MY85,3.385836143819594,46,0.10634348728027077,0.3090909090909091
core_c100_n20_geo,TA170,72.03840641615417,42,0.8881882124666505,0.8613918330457645
core_c60_n20_geo,PROTO,81.8423675122601,46,0.8484664807855498,0.9693339722089123
core_c60_n20_geo,0260,60.70831780635219,46,0.572558478712066,0.8449573757415624
core_c60_n20_geo,E047,70.10256797290842,46,0.8427886536156612,0.7395635963571909
core_c60_n20_geo,TN10,85.31964515495466,46,0.8909730363423212,0.9952339524151756
core_c60_n20_geo,MY85,9.28351439001548,46,0.10634348728027077,0.3090909090909091
core_c60_n20_geo,TA170,78.21054041463786,42,0.8881882124666505,0.8613918330457645
core_c140_n20_geo,PROTO,70.04701101986832,46,0.8484664807855498,0.9693339722089123
core_c140_n20_geo,0260,48.91296131396042,46,0.572558478712066,0.8449573757415624
core_c140_n20_geo,E047,58.30721148051664,46,0.8427886536156612,0.7395635963571909
core_c140_n20_geo,TN10,73.5242886625629,46,0.8909730363423212,0.9952339524151756
core_c140_n20_geo,MY85,0.0,46,0.10634348728027077,0.3090909090909091
core_c140_n20_geo,TA170,65.8662724176705,42,0.8881882124666505,0.8613918330457645
core_c100_n15_geo,PROTO,75.9446892660642,46,0.8484664807855498,0.9693339722089123
core_c100_n15_geo,0260,54.810639560156304,46,0.572558478712066,0.8449573757415624
core_c100_n15_geo,E047,64.20488972671252,46,0.8427886536156612,0.7395635963571909
core_c100_n15_geo,TN10,79.42196690875878,46,0.8909730363423212,0.9952339524151756
core_c100_n15_geo,MY85,3.385836143819594,46,0.10634348728027077,0.3090909090909091
core_c100_n15_geo,TA170,72.03840641615417,42,0.8881882124666505,0.8613918330457645
core_c100_n25_geo,PROTO,75.9446892660642,46,0.8484664807855498,0.9693339722089123
core_c100_n25_geo,0260,54.810639560156304,46,0.572558478712066,0.8449573757415624
core_c100_n25_geo,E047,64.20488972671252,46,0.8427886536156612,0.7395635963571909
core_c100_n25_geo,TN10,79.42196690875878,46,0.8909730363423212,0.9952339524151756
core_c100_n25_geo,MY85,3.385836143819594,46,0.10634348728027077,0.3090909090909091
core_c100_n25_geo,TA170,72.03840641615417,42,0.8881882124666505,0.8613918330457645
symm_c100_n20_mean,PROTO,76.14582703423339,46,0.8484664807855498,0.9693339722089123
symm_c100_n20_mean,0260,56.13159710719171,46,0.572558478712066,0.8449573757415624
symm_c100_n20_mean,E047,64.3734168831529,46,0.8427886536156612,0.7395635963571909
symm_c100_n20_mean,TN10,79.56615382238512,46,0.8909730363423212,0.9952339524151756
symm_c100_n20_mean,MY85,6.027524203069278,46,0.10634348728027077,0.3090909090909091
symm_c100_n20_mean,TA170,72.04866727941157,42,0.8881882124666505,0.8613918330457645
```

---

## 5) Baseline + ML domain weights (hard acceptability) — all variants

**Sources:** `algorithm/ml/learn_domain_weights.py` + `algorithm/eval/evaluate_couples.py`  
**Outputs:** `algorithm/out/ml_weights.json`, `algorithm/out_ml/summary_by_variant.csv`, `algorithm/out_ml/per_couple_scores.csv`

### 5.1 Learned weights

Raw file: `algorithm/out/ml_weights.json`

```
{
  "Values": 0.16666666666666666,
  "Communication": 0.16666666666666666,
  "Lifestyle": 0.16666666666666666,
  "Social": 0.16666666666666666,
  "Personality": 0.16666666666666666,
  "Friendship": 0.16666666666666666,
  "cv_auc": 0.5583333333333333
}
```

### 5.2 Variant summary table

From `algorithm/out_ml/summary_by_variant.csv` (AUC shown as both repo and standard):

```
variant	n_couples	auc_repo	auc_std	lift	median_rank	true_mean	rnd_mean
symm_c100_n20_mean	6	0.522222	0.477778	-1.0517402220057406	14	60.10857603990983	61.16031626191557
core_c100_n20_geo	6	0.516667	0.483333	-0.9352984070716772	14	59.726369173960194	60.66166758103187
core_c100_n15_geo	6	0.516667	0.483333	-0.9352984070716772	14	59.726369173960194	60.66166758103187
core_c100_n25_geo	6	0.516667	0.483333	-0.9352984070716772	14	59.726369173960194	60.66166758103187
core_c140_n20_geo	6	0.516667	0.483333	-0.9435960458221118	14	53.782948302383005	54.72654434820512
core_c60_n20_geo	6	0.511111	0.488889	-0.9270007683212498	14	65.66979004553737	66.59679081385862
```

### 5.3 Per-couple scores (all variants)

Raw (exact) file content: `algorithm/out_ml/per_couple_scores.csv`

```
variant,couple_id,score,overlap,sA,sB
core_c100_n20_geo,PROTO,76.69339040761754,46,0.8671874999999999,0.9641319942611192
core_c100_n20_geo,0260,56.322913446455956,46,0.6160826032540676,0.8197819519243396
core_c100_n20_geo,E047,65.64910385911938,46,0.8425259792166268,0.7671078114912847
core_c100_n20_geo,TN10,79.14726373021466,46,0.9089874857792947,0.9698270081802334
core_c100_n20_geo,MY85,15.964200663272193,46,0.231651376146789,0.4070796460176995
core_c100_n20_geo,TA170,64.58134293708139,42,0.766677620768978,0.8350144092219021
core_c60_n20_geo,PROTO,82.59106865381344,46,0.8671874999999999,0.9641319942611192
core_c60_n20_geo,0260,62.220591692651844,46,0.6160826032540676,0.8197819519243396
core_c60_n20_geo,E047,71.54678210531526,46,0.8425259792166268,0.7671078114912847
core_c60_n20_geo,TN10,85.04494197641054,46,0.9089874857792947,0.9698270081802334
core_c60_n20_geo,MY85,21.86187890946808,46,0.231651376146789,0.4070796460176995
core_c60_n20_geo,TA170,70.75347693556506,42,0.766677620768978,0.8350144092219021
core_c140_n20_geo,PROTO,70.79571216142166,46,0.8671874999999999,0.9641319942611192
core_c140_n20_geo,0260,50.425235200260076,46,0.6160826032540676,0.8197819519243396
core_c140_n20_geo,E047,59.7514256129235,46,0.8425259792166268,0.7671078114912847
core_c140_n20_geo,TN10,73.24958548401878,46,0.9089874857792947,0.9698270081802334
core_c140_n20_geo,MY85,10.066522417076307,46,0.231651376146789,0.4070796460176995
core_c140_n20_geo,TA170,58.4092089385977,42,0.766677620768978,0.8350144092219021
core_c100_n15_geo,PROTO,76.69339040761754,46,0.8671874999999999,0.9641319942611192
core_c100_n15_geo,0260,56.322913446455956,46,0.6160826032540676,0.8197819519243396
core_c100_n15_geo,E047,65.64910385911938,46,0.8425259792166268,0.7671078114912847
core_c100_n15_geo,TN10,79.14726373021466,46,0.9089874857792947,0.9698270081802334
core_c100_n15_geo,MY85,15.964200663272193,46,0.231651376146789,0.4070796460176995
core_c100_n15_geo,TA170,64.58134293708139,42,0.766677620768978,0.8350144092219021
core_c100_n25_geo,PROTO,76.69339040761754,46,0.8671874999999999,0.9641319942611192
core_c100_n25_geo,0260,56.322913446455956,46,0.6160826032540676,0.8197819519243396
core_c100_n25_geo,E047,65.64910385911938,46,0.8425259792166268,0.7671078114912847
core_c100_n25_geo,TN10,79.14726373021466,46,0.9089874857792947,0.9698270081802334
core_c100_n25_geo,MY85,15.964200663272193,46,0.231651376146789,0.4070796460176995
core_c100_n25_geo,TA170,64.58134293708139,42,0.766677620768978,0.8350144092219021
symm_c100_n20_mean,PROTO,76.82177909756624,46,0.8671874999999999,0.9641319942611192
symm_c100_n20_mean,0260,57.04903214343065,46,0.6160826032540676,0.8197819519243396
symm_c100_n20_mean,E047,65.73749391990586,46,0.8425259792166268,0.7671078114912847
symm_c100_n20_mean,TN10,79.19652908248669,46,0.9089874857792947,0.9698270081802334
symm_c100_n20_mean,MY85,17.192355492734713,46,0.231651376146789,0.4070796460176995
symm_c100_n20_mean,TA170,64.65426650333481,42,0.766677620768978,0.8350144092219021
```

---

## 6) Rank-based evaluation across all candidates (12 anchors)

This section treats each participant as an **anchor** and ranks all other participants as candidates using each algorithm’s score output, then reports:
- Hit@K, Mutual@3, MRR, median/mean rank
- AUC in both conventions (`auc_repo` and `auc_std`)

### 6.1 Canonical “paper bundle” comparison table

**Source:** `algorithm/out_paper/comparison_table.csv` and `algorithm/out_paper/per_couple_detail.csv`  
Algorithms included there: Evolutionary, Soft-Gated, Baseline (distance-kernel baseline), Merged IDF+LCB.

Also note: “Merged IDF+LCB” AUC is not computed in that pipeline; see §8 for details.

### 6.2 Consolidated ranking metrics (includes PAM score variants)

These were computed directly from the locally present score matrices:
- `algorithm/scores.csv` (PAM-like score matrix; default domain weights)
- `algorithm/scores_pam.csv` (PAM-like score matrix; trained weights — effectively uniform here)
- `algorithm/out_soft/scores_soft.csv`
- `algorithm/out_evo/scores_evo.csv`
- `algorithm/out_merged/summary_ranks.json` (for merged pipeline ranks)
- `algorithm/out_paper/comparison_table.csv` (for distance-kernel baseline row)

| Algorithm (source) | Hit@1 | Hit@3 | Hit@5 | MRR | AUC (repo) | AUC (std) | Mutual@3 | Median rank | Mean rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PAM_default (`algorithm/scores.csv`) | 25.0% | 25.0% | 33.3% | 0.357 | 0.453 | 0.547 | 16.7% | 6.5 | 6.0 |
| PAM_trained (`algorithm/scores_pam.csv`) | 25.0% | 25.0% | 50.0% | 0.360 | 0.483 | 0.517 | 16.7% | 5.5 | 6.1 |
| Soft-Gated (`algorithm/out_soft/metrics.csv`) | 16.7% | 41.7% | 50.0% | 0.353 | 0.497 | 0.503 | 16.7% | 6.0 | 5.8 |
| Evolutionary (`algorithm/out_evo/metrics.csv`) | 50.0% | 66.7% | 66.7% | 0.593 | 0.356 | 0.644 | 66.7% | 2.0 | 4.1 |
| Distance‑kernel Baseline (`algorithm/out_paper/comparison_table.csv`) | 8.3% | 33.3% | 33.3% | 0.273 | 0.544 | 0.456 | 16.7% | 7.0 | 6.6 |
| Merged IDF+LCB (`algorithm/out_merged/summary_ranks.json`) | 0.0% | 25.0% | 33.3% | 0.158 | — | — | 0.0% | 7.0 | 6.8 |

### 6.3 Per-couple partner ranks (A→B rank, B→A rank; 1-indexed)

This table is oriented by `couples.csv` (A = `partner_a_id`, B = `partner_b_id`):

```
couple_id	A	B	PAM_default (scores.csv)	PAM_trained (scores_pam.csv)	DistanceKernel_Baseline	SoftGated	Evolutionary	Merged_IDF+LCB
PROTO	c2b8317c	7de011b7	(7,1)	(4,1)	(2,7)	(4,1)	(1,1)	(9,5)
0260	28e08bc8	562431c5	(9,7)	(8,10)	(11,10)	(11,10)	(10,7)	(11,8)
E047	52e99f9a	1f7ba3a6	(6,6)	(5,6)	(6,7)	(8,8)	(3,1)	(4,3)
TN10	d795f6be	085e8968	(1,1)	(1,1)	(1,2)	(2,1)	(1,1)	(7,4)
MY85	fe5827c5	f82ca611	(11,11)	(11,11)	(3,11)	(9,2)	(11,9)	(12,12)
TA170	36bfb1da	11a90732	(8,4)	(10,5)	(10,9)	(10,3)	(3,1)	(8,10)
```

---

## 7) Soft-gated and evolutionary model configurations (results artifacts)

### 7.1 Soft-gated (learned domain modes)

Files:
- `algorithm/out_soft/best_config.json`
- `algorithm/out_soft/domain_summary.csv`
- `algorithm/out_soft/domain_ablation.csv`

Domain summary (exact):

```
domain,mode,weight,mu,sigma,alpha,beta
Values,similarity,0.30676775379433896,0.5,0.25,1.0,1.0
Communication,complementarity,0.13743895201784334,0.5,0.25,1.0,1.0
Lifestyle,mixed,0.3051975247240818,0.5,0.25,1.0,1.0
Social,irrelevant,4.642180860336173e-05,0.5,0.25,1.0,1.0
Personality,similarity,0.015554417849437919,0.5,0.25,1.0,1.0
Friendship,similarity,0.23499492980569445,0.5,0.25,1.0,1.0
```

Ablations (exact):

```
domain,mode,weight,hit@1_drop,d_hit@1_drop,hit@1_flip,d_hit@1_flip
Values,similarity,0.30676775379433896,0.16666666666666666,0.0,0.08333333333333333,-0.08333333333333333
Communication,complementarity,0.13743895201784334,0.08333333333333333,-0.08333333333333331,0.16666666666666666,0.0
Lifestyle,mixed,0.3051975247240818,0.16666666666666666,0.0,0.16666666666666666,0.0
Social,irrelevant,4.642180860336173e-05,0.16666666666666666,0.0,0.16666666666666666,0.0
Personality,similarity,0.015554417849437919,0.16666666666666666,0.0,0.16666666666666666,0.0
Friendship,similarity,0.23499492980569445,0.16666666666666666,0.0,0.16666666666666666,0.0
```

### 7.2 Evolutionary (discrete domain mode search)

Files:
- `algorithm/out_evo/best_config.json`
- `algorithm/out_evo/domain_summary.csv`
- `algorithm/out_evo/domain_ablation.csv`
- `algorithm/out_evo/history.csv` (large; search log)

Domain summary (exact):

```
domain,mode,weight,mu,sigma,alpha,beta
Values,complementarity,0.15104188960808168,0.25,0.25,1.0,1.0
Communication,complementarity,0.2570650894724816,0.25,0.25,1.0,1.0
Lifestyle,complementarity,0.21713736800820874,0.25,0.3499999940395355,1.0,1.0
Social,irrelevant,0.21351891662186018,0.25,0.3499999940395355,1.0,1.0
Personality,complementarity,0.11283152140927352,0.5,0.15000000596046448,1.0,1.0
Friendship,complementarity,0.04840521488009435,0.25,0.15000000596046448,1.0,1.0
```

Ablations (exact):

```
domain,mode,weight,hit@1_drop,d_hit@1_drop,hit@1_flip,d_hit@1_flip
Values,complementarity,0.15104188960808168,0.25,-0.25,0.3333333333333333,-0.16666666666666669
Communication,complementarity,0.2570650894724816,0.4166666666666667,-0.08333333333333331,0.4166666666666667,-0.08333333333333331
Lifestyle,complementarity,0.21713736800820874,0.4166666666666667,-0.08333333333333331,0.25,-0.25
Social,irrelevant,0.21351891662186018,0.5,0.0,0.5,0.0
Personality,complementarity,0.11283152140927352,0.4166666666666667,-0.08333333333333331,0.3333333333333333,-0.16666666666666669
Friendship,complementarity,0.04840521488009435,0.25,-0.25,0.25,-0.25
```

---

## 8) Merged IDF+LCB pipeline (tie-aware ranking + matching diagnostics)

**Output directory:** `algorithm/out_merged/`

### 8.1 Summary metrics (from `algorithm/out_merged/summary_ranks.json`)

- n_people: 12
- hit@1: 0.0
- hit@2: 0.0833
- hit@3: 0.25
- median_rank: 7.0
- mean_rank: 6.75
- mutual@3 (computed from ranks + `couples.csv`): 0.0 (0 / 6 couples)
- robustness labels (per-person): fragile = 9, moderate = 3
- top_gap distribution (12 people): min 0.00029, median 0.03468, mean 0.05519, max 0.15909

### 8.2 Per-person partner rank diagnostics (1-indexed)

From `algorithm/out_merged/summary_ranks.json` (IDs truncated):

```
anchor	partner	rank	top_gap	robustness	true_couple_id
c2b8317c	7de011b7	9	0.0040501277100635225	fragile	PROTO
7de011b7	c2b8317c	5	0.00028785860044605593	fragile	PROTO
28e08bc8	562431c5	11	0.029761440004134943	fragile	0260
562431c5	28e08bc8	8	0.1358466558462636	moderate	0260
52e99f9a	1f7ba3a6	4	0.15908976263093177	moderate	E047
1f7ba3a6	52e99f9a	3	0.039588827634416124	fragile	E047
d795f6be	085e8968	7	0.1273574212368362	moderate	TN10
085e8968	d795f6be	4	0.02938512759608819	fragile	TN10
fe5827c5	f82ca611	12	0.04503528774617521	fragile	MY85
f82ca611	fe5827c5	12	0.0007243232076221129	fragile	MY85
36bfb1da	11a90732	8	0.08747294114099241	fragile	TA170
11a90732	36bfb1da	10	0.0037046369913549815	fragile	TA170
```

---

## 9) “Production” demo matching output (global 1:1 matching)

**Output file:** `algorithm/out_production/optimal_matching.csv`

Anonymized view (IDs truncated; `true_couple_id` blank means not a ground-truth couple):

```
A_id	B_id	score	pct	mutual_ranks	true_couple_id
d795f6be	085e8968	0.4574064865620324	100.0	(1, 1)	TN10
52e99f9a	fe5827c5	0.4550310376009533	99.30422195037414	(1, 1)	
7de011b7	c2b8317c	0.44933116002511	97.6347060955587	(1, 1)	PROTO
36bfb1da	562431c5	0.41637544468277493	87.98185153180567	(2, 1)	
28e08bc8	1f7ba3a6	0.3576835627973142	70.79077574940452	(6, 6)	
11a90732	f82ca611	0.21959126072317223	30.343014429437527	(11, 2)	
```

On the six-couple ground truth, this particular matching recovers **2 / 6** true couples (TN10 and PROTO).
