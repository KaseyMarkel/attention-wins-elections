"""
GDELT TV (broadcast) attention — presidential candidates, 2012–2024.

Source: GDELT Television Explorer 2.0 (Internet Archive TV News Archive, 2009–present).
Metric: each candidate's share of national cable-news airtime (CNN + Fox News +
        MSNBC) over the 12 months ending on Election Day, via the TV API's
        `timelinevol` mode (percent of 15-second clips mentioning the query).
        Mention share = winner_vol / (winner_vol + loser_vol).

This is a genuinely different medium from the book/search/news sources: spoken
broadcast coverage. It covers the modern elections the Ngrams corpus misses
(2020, 2024). Only 4 elections fall in the TV archive window, so n is small —
shown faded in Figure 7.

Output: data/raw/gdelt_tv/presidential_mention_shares.csv
"""

import time
from pathlib import Path
import sys

import requests
import pandas as pd
from scipy.stats import binomtest, pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.candidates import PRESIDENTIAL

OUT_DIR = Path('data/raw/gdelt_tv')
OUT_DIR.mkdir(parents=True, exist_ok=True)
API = 'https://api.gdeltproject.org/api/v2/tv/tv'
STATIONS = '(station:CNN OR station:FOXNEWS OR station:MSNBC)'
HEADERS = {'User-Agent': 'attention-wins-elections/1.0 (kaseymarkel@semillanueva.org)'}

# GDELT TV archive begins mid-2009 → only these election windows are covered.
TV_YEARS = [2012, 2016, 2020, 2024]


def airtime(full_name: str, year: int) -> float:
    """Mean % of national cable airtime mentioning `full_name` over the 12
    months ending on ~Election Day."""
    start = f'{year-1}1105000000'
    end   = f'{year}1105000000'
    params = {'query': f'"{full_name}" {STATIONS}', 'mode': 'timelinevol',
              'format': 'json', 'startdatetime': start, 'enddatetime': end}
    for attempt in range(3):
        try:
            r = requests.get(API, params=params, timeout=40, headers=HEADERS)
            j = r.json()
            tl = j.get('timeline', [])
            if not tl:
                return 0.0
            vals = [d['value'] for d in tl[0]['data']]
            return sum(vals) / len(vals) if vals else 0.0
        except Exception:
            time.sleep(4 * (attempt + 1))
    return 0.0


def main():
    by_year = {p[0]: p for p in PRESIDENTIAL}
    rows = []
    for yr in TV_YEARS:
        _, wn, ln, wp, lp, wv, lv, _ = by_year[yr]
        print(f'=== {yr}: {wn} vs {ln} ===')
        w_vol = airtime(wn, yr); time.sleep(1.0)
        l_vol = airtime(ln, yr); time.sleep(1.0)
        total = w_vol + l_vol
        w_share = w_vol / total if total else 0.5
        rows.append({
            'year': yr, 'source': 'GDELT TV',
            'winner_name': wn, 'loser_name': ln, 'winner_party': wp,
            'winner_mention_share': round(w_share, 6),
            'loser_mention_share': round(1 - w_share, 6),
            'winner_vote_pct': wv, 'loser_vote_pct': lv,
            'attention_leader_won': int(w_share > 0.5),
            'winner_airtime_pct': round(w_vol, 4), 'loser_airtime_pct': round(l_vol, 4),
            'stations': 'CNN+FoxNews+MSNBC',
        })
        print(f'  {wn}: {w_vol:.2f}%  vs  {ln}: {l_vol:.2f}%  '
              f'-> winner share {w_share:.1%} (leader won: {bool(w_share > 0.5)})')

    df = pd.DataFrame(rows)
    out = OUT_DIR / 'presidential_mention_shares.csv'
    df.to_csv(out, index=False)
    print(f'\nSaved {len(df)} elections to {out}')
    k, n = int(df['attention_leader_won'].sum()), len(df)
    print(f'H1 (GDELT TV): {k}/{n} ({k/n:.0%}), p={binomtest(k, n, 0.5).pvalue:.3f}')
    if n >= 3:
        r, p = pearsonr(df['winner_mention_share'], df['winner_vote_pct'])
        print(f'H2 (GDELT TV): r={r:.3f}, p={p:.4f}')


if __name__ == '__main__':
    main()
