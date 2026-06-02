# Pre-Registration: "Political Attention Is All You Need"

**Timestamp:** This document was committed before any data collection or analysis.
The initial commit hash of this repository serves as the pre-registration timestamp.

---

## Study Title

Political Attention Is All You Need: Does Media Mention Share Predict US Presidential Election Outcomes?

## Motivation

A 2026 Nature paper (Brady et al., "Redesigning algorithms to intervene on social norm
misperceptions during a national election") used Named Entity Recognition on Bluesky posts
to show that Trump received dramatically more mentions than any other political entity across
all three feed algorithm conditions during the 2024 election cycle. This raises a broader
historical question: does the candidate who receives the most media attention consistently
win elections? If so, does this hold across different media eras (print, broadcast, digital,
social)?

## Hypotheses

### H1 (Primary — Confirmatory)
In US presidential elections from 1960–2024, the major-party candidate with the higher
**mention share** in the 12 months preceding Election Day will win the **popular vote**
more often than chance (i.e., more than 50% of elections).

- **Test:** One-sample binomial test against p = 0.50
- **Significance threshold:** α = 0.05 (two-tailed)

### H2 (Secondary — Confirmatory)
Candidate **mention share** (proportion of total major-candidate mentions) will positively
correlate with candidate **popular vote share** across all elections in the dataset.

- **Test:** Pearson r, with 34 data points (17 elections × 2 major candidates)
- **Significance threshold:** α = 0.05

### H3 (Exploratory — Not confirmatory)
The magnitude of the mention gap (winner_share / loser_share) will positively correlate
with the margin of popular vote victory. Reported descriptively; not used to confirm or
disconfirm the main hypothesis.

---

## Scope

- **Elections:** US Presidential, 1960–2024 (17 elections)
- **Candidates:** Major-party nominees only (Democrat + Republican)
  - Third-party candidates (e.g., Perot 1992) included in mention share denominator
    but excluded from H1 win/loss analysis
- **Outcome variable:** Popular vote (not Electoral College)
- **Measurement window:** 12 calendar months ending on Election Day

---

## Data Sources by Era

| Era | Primary Source | Notes |
|---|---|---|
| 1960–1984 | GDELT + ProQuest Historical Newspapers (NYT, WaPo) | Print/wire era |
| 1984–2000 | GDELT Global Knowledge Graph | Broadcast + early digital |
| 2000–2012 | GDELT GKG v2 | Digital news era |
| 2012–2020 | GDELT GKG v2 + Google Trends (supplementary) | Social media era begins |
| 2020–2024 | GDELT GKG v2 + Google Trends | Replication of Brady et al. era |

**Search unit:** Candidate last name (e.g., "Nixon", "Kennedy"). For ambiguous surnames
(Bush 1988/1992/2000), full name or "George H.W. Bush" / "George W. Bush" will be used
and documented. Edge cases logged in `data/edge_cases.md`.

---

## Primary Metric

```
mention_share(candidate, year) =
    mentions(candidate, window) / sum(mentions(all_major_candidates, window))
```

Where `window` = 365 days ending on Election Day.

---

## Statistical Analysis Plan

### Step 1 — Data collection
Collect raw mention counts per candidate per election via GDELT API and supplementary
sources. Log all queries and raw counts in `data/raw/`.

### Step 2 — Compute mention share
For each election year, compute mention_share for each major candidate.
Identify the "attention winner" (candidate with higher mention share).

### Step 3 — H1: Binomial test
Count elections where attention winner = popular vote winner.
Run `scipy.stats.binomtest(k, n=17, p=0.5, alternative='two-sided')`.

### Step 4 — H2: Correlation
Build a 34-row dataframe (one row per candidate-election).
Run `scipy.stats.pearsonr(mention_share, vote_share)`.
Plot scatter with labeled data points.

### Step 5 — H3: Exploratory
Compute `mention_gap = winner_mention_share / loser_mention_share`.
Compute `vote_margin = winner_vote_share - loser_vote_share`.
Plot and report Pearson r descriptively.

### Step 6 — Sensitivity checks (pre-specified)
- Repeat H1/H2 excluding incumbent elections (years where sitting president is running)
- Repeat with 6-month and 3-month windows (reported as sensitivity, not as primary test)
- Repeat excluding 1992 (Perot effect on denominator)

---

## Pre-committed Exclusion Rules

- We will **not** drop any election after seeing results
- We will **not** change the primary metric after seeing which performs better
- We will **not** switch the primary outcome from popular vote after seeing results
- Elections with ambiguous search terms will be documented before data collection begins
- All raw query results will be committed alongside analysis

---

## What We Are NOT Testing

- Whether attention *causes* electoral success (causal inference not claimed)
- Whether the effect holds for non-presidential elections
- Whether social media mentions vs. print mentions differ in predictive power
  (this is H3-level exploratory territory for a follow-up)

---

## Relationship to Brady et al. (2026)

Brady et al. measured within-platform algorithmic amplification of political entities
during the 2024 election on Bluesky. Their Figure 2 (NER entity frequency by feed type)
shows Trump receiving 255k–321k mentions vs. Harris at 85k–101k depending on algorithm
condition — roughly a 3:1 attention ratio. Our study extends this observation backward
through media history to test whether attention asymmetry is a reliable predictor of
electoral outcomes across eras.

---

*Pre-registration timestamp = initial git commit date/hash of this repository.*
