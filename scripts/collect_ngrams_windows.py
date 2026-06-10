"""
Window-sensitivity collection for the presidential Ngrams measure (Figure S2).

Pre-registration Step 6 asks whether the result is robust to the length of the
pre-election measurement window. Google Ngrams is annual book-corpus data, so
the natural analogue of a "window" is the number of years averaged ending on
the election year:

  1-year : election year only
  2-year : election year + prior year   (the primary measure used everywhere)
  3-year : election year + two prior years

For each election we issue ONE query per candidate spanning [year-2, year]
(a 3-value series — never a single year, which would trigger the Ngrams
"return the whole corpus" quirk) and derive all three windows from it.

Output: data/raw/ngrams/presidential_windows.csv
        one row per (year, window_years) with mention shares + attention winner.
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
import sys

import pandas as pd
from scipy.stats import binomtest, pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.candidates import PRESIDENTIAL

OUT_DIR = Path('data/raw/ngrams')
OUT_DIR.mkdir(parents=True, exist_ok=True)
HEADERS = {'User-Agent': 'attention-wins-elections/1.0 (kaseymarkel@semillanueva.org)'}
WINDOWS = [1, 2, 3]


def fetch_span(name: str, year_start: int, year_end: int) -> dict:
    """Return {year: freq} for the inclusive span. Raises on length mismatch."""
    params = {'content': name, 'year_start': year_start, 'year_end': year_end,
              'corpus': 'en-2019', 'smoothing': 0}
    url = 'https://books.google.com/ngrams/json?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    time.sleep(1.0)
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read())
    expected = year_end - year_start + 1
    years = list(range(year_start, year_end + 1))
    for series in data:
        if series['ngram'] == name:
            ts = series['timeseries']
            if len(ts) != expected:
                raise ValueError(f'{name!r}: got {len(ts)} pts, expected {expected}')
            return dict(zip(years, ts))
    return {y: 0.0 for y in years}      # name absent from corpus


def window_freq(series: dict, year: int, n_years: int) -> float:
    """Sum freq over the n_years ending on `year`."""
    return sum(series.get(y, 0.0) for y in range(year - n_years + 1, year + 1))


def main():
    rows = []
    for yr, wn, ln, wp, lp, wv, lv, _ in PRESIDENTIAL:
        if yr > 2019:
            continue
        print(f'=== {yr}: {wn} vs {ln} ===')
        w_series = fetch_span(wn, yr - 2, yr)
        l_series = fetch_span(ln, yr - 2, yr)
        for n_years in WINDOWS:
            wf = window_freq(w_series, yr, n_years)
            lf = window_freq(l_series, yr, n_years)
            total = wf + lf
            w_share = wf / total if total else 0.5
            rows.append({
                'year': yr, 'window_years': n_years,
                'winner_name': wn, 'loser_name': ln, 'winner_party': wp,
                'winner_mention_share': round(w_share, 6),
                'loser_mention_share': round(1 - w_share, 6),
                'winner_vote_pct': wv, 'loser_vote_pct': lv,
                'attention_leader_won': int(w_share > 0.5),
                'winner_raw_freq': wf, 'loser_raw_freq': lf,
            })
            print(f'  {n_years}yr: winner share {w_share:.1%} '
                  f'(leader won: {bool(w_share > 0.5)})')

    df = pd.DataFrame(rows)
    out = OUT_DIR / 'presidential_windows.csv'
    df.to_csv(out, index=False)
    print(f'\nSaved {len(df)} rows to {out}\n')

    for n_years in WINDOWS:
        sub = df[df['window_years'] == n_years]
        k, n = int(sub['attention_leader_won'].sum()), len(sub)
        p = binomtest(k, n, 0.5, alternative='two-sided').pvalue
        r, _ = pearsonr(sub['winner_mention_share'], sub['winner_vote_pct'])
        print(f'{n_years}-year window: H1 {k}/{n} ({k/n:.0%}), p={p:.4f}  H2 r={r:.3f}')


if __name__ == '__main__':
    main()
