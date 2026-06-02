# Political Attention Is All You Need

> A pre-registered study testing whether the US presidential candidate with the most media
> mentions wins the popular vote — across every election from 1960 to 2024.

**Inspired by:** Brady et al. (2026), "Redesigning algorithms to intervene on social norm
misperceptions during a national election," *Nature*. Their Figure 2 shows Trump receiving
3× more Bluesky mentions than Harris. We ask: is this pattern consistent across all of
modern electoral history?

## Hypothesis

The candidate with the higher **mention share** in the 12 months before Election Day wins
the popular vote more often than chance.

See [`PREREGISTRATION.md`](PREREGISTRATION.md) for the full pre-registered study design,
including all hypotheses, data sources, and statistical analysis plan. The initial commit
timestamp of this repository serves as the pre-registration date.

## Repository Structure

```
attention-wins-elections/
├── PREREGISTRATION.md          # Full pre-registered design (read before touching data)
├── data/
│   ├── raw/                    # Raw query results from GDELT and other sources
│   ├── processed/              # Cleaned mention counts and vote share data
│   └── edge_cases.md           # Documentation of ambiguous search terms
├── notebooks/
│   ├── 01_data_collection.ipynb    # GDELT API queries + data pulls
│   ├── 02_data_cleaning.ipynb      # Standardize, validate, compute mention share
│   └── 03_analysis.ipynb           # Hypothesis tests + figures
└── blog/
    └── draft.md                # Blog post draft
```

## Quick Start

```bash
pip install -r requirements.txt
jupyter lab
# Open notebooks/01_data_collection.ipynb
```

## Status

- [x] Pre-registration committed
- [ ] Data collection (GDELT 1960–2024)
- [ ] Data cleaning and mention share computation
- [ ] Analysis and figures
- [ ] Blog post
