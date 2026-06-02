# Political Attention Is All You Need

> **Can we predict who wins a US presidential election just by counting who gets mentioned more in the news?**

This is a pre-registered study testing whether **mention share** — the fraction of media coverage a candidate receives in the year before an election — predicts popular vote outcomes across all US presidential elections from 1960 to 2024.

---

## The Idea

A [2026 Nature paper](https://www.nature.com/articles/s41586-026-10536-1) (Brady et al.) ran a field experiment on Bluesky during the 2024 election, randomly assigning 2,000 users to different feed algorithms. One of their figures — reproduced below from their paper — shows something striking: Trump received roughly **3× more mentions** than Harris across every feed condition tested.

That got us wondering: is this kind of lopsided attention distribution normal? Does it always characterize the eventual winner? And does it hold not just on social media in 2024, but across all of modern electoral history?

---

## Pre-Registration

This study was **pre-registered before any data was collected.** The initial commit of this repository timestamps our hypotheses, data sources, and statistical analysis plan. We committed to our methods before seeing results — the same approach Brady et al. used, and the same principle that distinguishes confirmatory from exploratory science.

See [`PREREGISTRATION.md`](PREREGISTRATION.md) for the full design. In short:

| | |
|---|---|
| **Elections** | US Presidential, 1960–2024 (17 elections) |
| **Candidates** | Major-party nominees only |
| **Metric** | Mention share = candidate mentions ÷ total candidate mentions, 12 months pre-election |
| **Primary outcome** | Popular vote winner |
| **Sources** | GDELT (1979–2024), Google Ngrams (1960–1979) |

---

## Results

> ⚠️ **Note: the figures below use synthetic data** generated to validate the analysis pipeline and figure layouts. Real data collection is in progress. This section will be updated when real data is in.

### H1 — Does the attention leader win?

The candidate with higher mention share in the 12 months before election day wins the popular vote in **14 out of 17 elections (82%)** in our synthetic dataset — significantly more often than the 50% you'd expect by chance.

![H1: Win rate](figures/fig1_h1_win_rate.png)

---

### H2 — Does mention share track vote share continuously?

Beyond just predicting winners, mention share correlates with the actual vote share each candidate receives. A candidate who dominates coverage doesn't just tend to win — they tend to win by more.

![H2: Scatter plot](figures/fig2_h2_scatter.png)

*Each dot is a candidate-election. Blue = Democrat, Red = Republican. Labels shown for notable outliers.*

---

### H3 (Exploratory) — Does a bigger attention gap mean a bigger win?

When one candidate dominates coverage by a larger margin, do they win by a larger margin in votes? This is exploratory — not part of the pre-registered confirmatory analysis — but it's the most intuitive version of the hypothesis.

![H3: Gap vs. margin](figures/fig3_h3_gap_margin.png)

*Red dots are "attention upsets" — elections where the candidate with less coverage won anyway.*

---

### Timeline — The attention gap across 64 years

How lopsided was media attention in each election, and did the more-covered candidate always win?

![Timeline](figures/fig4_timeline.png)

*Bars above 1× = winner got more coverage than loser. Red bars = the loser actually got more coverage (attention upset).*

---

## Sensitivity Analyses

Per pre-registration, we also run:
- **Excluding incumbent elections** — incumbents have name recognition advantages unrelated to campaigns
- **GDELT-only (1979+)** — removes the source heterogeneity of the pre-GDELT era
- **Excluding 1992** — Perot inflates the third-party denominator

---

## Repo Structure

```
attention-wins-elections/
├── PREREGISTRATION.md              # Full pre-registered design (read first)
├── data/
│   ├── raw/                        # Raw query results
│   ├── processed/
│   │   ├── election_results.csv    # Official vote shares, 1960–2024
│   │   └── mention_share.csv       # Computed mention shares (post-collection)
│   └── edge_cases.md               # Disambiguation decisions (Bush, Clinton, etc.)
├── figures/                        # All exported figures
├── notebooks/
│   ├── 01_data_collection.ipynb    # GDELT + Ngrams queries
│   ├── 02_data_cleaning.ipynb      # Standardize and compute mention share
│   └── 03_analysis.ipynb           # Hypothesis tests + figures
├── scripts/
│   └── generate_fake_data_and_figures.py   # Synthetic data for layout testing
└── requirements.txt
```

## Running the Analysis

```bash
git clone https://github.com/KaseyMarkel/attention-wins-elections
cd attention-wins-elections
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/01_data_collection.ipynb` to start data collection, or run
`scripts/generate_fake_data_and_figures.py` to reproduce the synthetic figures.

---

## Status

- [x] Pre-registration committed
- [x] Figure layout validated with synthetic data
- [ ] Data collection (GDELT 1979–2024)
- [ ] Data collection (Google Ngrams 1960–1978)
- [ ] Data cleaning and mention share computation
- [ ] Analysis with real data
- [ ] Blog post

---

*Inspired by Brady et al. (2026), "Redesigning algorithms to intervene on social norm misperceptions during a national election," Nature.*
