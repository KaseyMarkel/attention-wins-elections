# Is Attention All You Need (to Win Elections)?

> A pre-registered study testing whether the candidate who receives the most media coverage also wins — across US presidential, Senate, and House elections from 1960 to 2024.

---

## Background

A [2026 Nature paper](https://www.nature.com/articles/s41586-026-10536-1) (Brady et al.) ran a field experiment on Bluesky during the 2024 election, randomly assigning 2,000 users to different feed algorithms. One of their figures shows something striking: Trump received roughly **3× more mentions** than Harris across every feed condition tested.

That raises a broader question: is lopsided attention distribution a consistent feature of elections that the winner dominates? Does it hold not just on social media in 2024, but across all of modern electoral history — and at every level of government?

---

## Pre-Registration

This study was **pre-registered before any data was collected.** The initial commit of this repository timestamps our hypotheses, data sources, and statistical analysis plan. We committed to our methods before seeing results — the same approach Brady et al. used.

See [`PREREGISTRATION.md`](PREREGISTRATION.md) for the full design.

| | |
|---|---|
| **Presidential elections** | 1960–2024 (17 elections) |
| **Senate races** | ~33 races/cycle, every 2 years 1960–2024 (~500 races) |
| **House** | Party-aggregate attention share vs. seat share, 33 cycles |
| **Attention metric** | Mention share = candidate mentions ÷ total candidate mentions, 12 months pre-election |
| **Primary outcome** | Popular vote winner (Presidential); race winner (Senate); chamber majority (House) |
| **Data sources** | GDELT (1979–2024), Google Ngrams (1960–1979) |

---

## Results

> ⚠️ **The figures below use synthetic data** generated to validate the analysis pipeline and figure layouts. All candidate names are fictional placeholders. Real data collection is in progress.

---

### Part 1 — Presidential Elections

#### Figure 1. Does the attention leader win the popular vote?

The candidate with higher mention share in the 12 months before election day wins the popular vote in **14 out of 17 elections (82%)** — significantly above the 50% you'd expect by chance.

![Figure 1](figures/fig1_h1_win_rate.png)

---

#### Figure 2. Does mention share track vote share continuously?

Beyond predicting winners, mention share correlates with the actual vote percentage each candidate receives. A candidate who dominates coverage doesn't just tend to win — they tend to win by more.

![Figure 2](figures/fig2_h2_scatter.png)

*Each dot is a candidate-election. Blue = Democrat, Red = Republican. Labels shown for notable outliers; hover for all.*

---

#### Figure 3. Does a bigger attention gap mean a bigger win?

When one candidate dominates coverage by a larger margin, do they win by a larger vote margin? Red dots are **attention upsets** — elections where the less-covered candidate won.

![Figure 3](figures/fig3_h3_gap_margin.png)

---

#### Figure 4. The attention gap across 64 years of presidential elections

![Figure 4](figures/fig4_timeline.png)

*Dark = attention predicted the winner. Red = the less-covered candidate won (attention upset).*

---

### Part 2 — Scaling Up: Senate and House

The presidential finding is suggestive but rests on only 17 elections. Senate races provide ~500 independent tests of the same hypothesis. The House adds a different question: does the party that dominates national political coverage win more seats that cycle?

#### Figure 5. The effect holds at every level of government

![Figure 5](figures/fig5_win_rate_comparison.png)

The effect replicates in Senate races (1,089 contests). The House signal is weaker — most individual House races fly below the national media radar, diluting the party-aggregate signal.

---

#### Figure 6. Senate: mention share vs. vote share (~1,100 races)

With hundreds of races, the scatter fills in. The correlation is consistent with the presidential finding, and the volume of points makes the linear trend hard to dismiss.

![Figure 6](figures/fig6_senate_scatter.png)

---

#### Figure 7. House: does the party dominating coverage win the chamber?

Each point is one election cycle. Color indicates which party held the majority. The positive slope suggests that the party with more national coverage tends to win more seats — but the signal is noisier than at the individual-race level.

![Figure 7](figures/fig7_house_seat_share.png)

---

#### Figure 8. How lopsided is attention, and does it vary by race type?

Presidential races (tick marks at bottom, KDE curve) show a wider spread than Senate races — some elections see dramatic attention asymmetries while others are near 1×. Senate races cluster more tightly around modest coverage advantages.

![Figure 8](figures/fig8_gap_distribution.png)

---

## Supplemental Figures

#### Figure S1. Does the effect vary by media era?

The attention–outcome link appears across all four media eras, though with wide confidence intervals given the small number of elections per era. The social media era shows the strongest signal in this synthetic data — an empirical question for the real analysis.

![Figure S1](figures/figS1_by_era.png)

---

#### Figure S2. Sensitivity: does the measurement window matter?

The 12-month window (primary pre-registered measure) performs best. Shorter windows are noisier proxies. This is consistent with the hypothesis that sustained, cumulative attention — not just late-campaign coverage — drives the effect.

![Figure S2](figures/figS2_window_sensitivity.png)

---

#### Table S1. Presidential elections: complete data

[`data/processed/table_s1_presidential_FAKE.csv`](data/processed/table_s1_presidential_FAKE.csv) — all 17 elections with candidate names, mention shares, vote shares, and whether the attention leader won. *(Synthetic data.)*

---

## Repo Structure

```
attention-wins-elections/
├── PREREGISTRATION.md                      # Full pre-registered design (read first)
├── data/
│   ├── raw/                                # Raw query results (post-collection)
│   ├── processed/
│   │   ├── election_results.csv            # Official vote shares, 1960–2024
│   │   ├── mention_share.csv               # Computed mention shares (post-collection)
│   │   └── table_s1_presidential_FAKE.csv  # Supplemental Table S1 (synthetic)
│   └── edge_cases.md                       # Disambiguation decisions (Bush, Clinton, etc.)
├── figures/                                # All exported figures (fig1–fig8, figS1–figS2)
├── notebooks/
│   ├── 01_data_collection.ipynb            # GDELT + Ngrams queries
│   ├── 02_data_cleaning.ipynb              # Standardize and compute mention share
│   └── 03_analysis.ipynb                   # Hypothesis tests + figures
├── scripts/
│   └── generate_fake_data_and_figures.py   # Synthetic data for layout testing
└── requirements.txt
```

## Running

```bash
git clone https://github.com/KaseyMarkel/attention-wins-elections
cd attention-wins-elections
pip install -r requirements.txt

# Reproduce synthetic figures
python scripts/generate_fake_data_and_figures.py

# Real data pipeline
jupyter lab  # → notebooks/01_data_collection.ipynb
```

---

## Status

- [x] Pre-registration committed
- [x] Figure layout validated with synthetic data (presidential + senate + house)
- [x] Supplemental figures (by era, window sensitivity) + Table S1
- [ ] Data collection — GDELT 1979–2024
- [ ] Data collection — Google Ngrams 1960–1978
- [ ] Data cleaning and mention share computation
- [ ] Analysis with real data
- [ ] Blog post

---

*Inspired by Brady et al. (2026), "Redesigning algorithms to intervene on social norm misperceptions during a national election," Nature.*
