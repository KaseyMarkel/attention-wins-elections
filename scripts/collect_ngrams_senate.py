"""
Google Ngrams — Senate race mention shares.

For each D-vs-R Senate race (1976–2018), queries both candidate names together
in a single Ngrams request for the 2 years ending on election day.

Key limitation: most Senate candidates have near-zero Ngrams frequency.
Races with non-zero data tend to be high-profile (senior senators, leadership).
We report the fraction with usable data and analyze only non-zero pairs.

Output:
  data/raw/ngrams/senate_mention_shares.csv    — one row per race with data
  data/raw/ngrams/senate_all_races.csv         — all races including zero-data
  data/raw/ngrams/senate_raw_YEAR_STATE.json   — raw API responses
"""

import json, time, urllib.request, urllib.parse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT_DIR = Path('data/raw/ngrams')
OUT_DIR.mkdir(parents=True, exist_ok=True)

NGRAMS_URL = 'https://books.google.com/ngrams/json'
HEADERS = {'User-Agent': 'attention-wins-elections/1.0 (kaseymarkel@semillanueva.org)'}


def ngrams_pair(term1: str, term2: str, year: int) -> tuple[float, float]:
    """
    Fetch Ngrams frequency for two terms in year-1 and year.
    Returns (freq1, freq2) summed over window.
    Saves raw JSON response.
    """
    year_start = max(1960, year - 1)
    year_end   = min(2019, year)
    if year > 2019:
        return None, None

    params = {
        'content': f'{term1},{term2}',
        'year_start': year_start,
        'year_end': year_end,
        'corpus': 'en-2019',
        'smoothing': 0,
    }
    url = NGRAMS_URL + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read()
            break
        except Exception as e:
            time.sleep(5 * (attempt + 1))
    else:
        return None, None

    data = json.loads(raw)
    freqs = {}
    for series in data:
        name = series['ngram']
        years_range = list(range(year_start, year_end + 1))
        freqs[name] = sum(series['timeseries'])

    f1 = freqs.get(term1, 0.0)
    f2 = freqs.get(term2, 0.0)
    return f1, f2


def main():
    senate = pd.read_csv('data/raw/senate_cleaned.csv')
    # Limit to Ngrams coverage (needs election year ≤ 2019, and we need year-1 data)
    senate = senate[senate['year'] <= 2018].copy()
    print(f'{len(senate)} Senate races in Ngrams coverage window (1976–2018)')

    all_rows = []
    n_total = len(senate)

    for i, row in senate.iterrows():
        yr = int(row['year'])
        winner = row['winner']
        loser  = row['loser']
        state  = row['state_po']

        if i % 50 == 0:
            print(f'  [{i}/{n_total}] {yr} {state}: {winner} vs {loser}')

        f_w, f_l = ngrams_pair(winner, loser, yr)
        time.sleep(1.2)  # polite rate limit

        if f_w is None:
            status = 'skip_year'
            winner_share = None
        elif f_w == 0 and f_l == 0:
            status = 'both_zero'
            winner_share = None
        elif f_w == 0:
            status = 'winner_zero'
            winner_share = 0.0
        elif f_l == 0:
            status = 'loser_zero'
            winner_share = 1.0
        else:
            status = 'ok'
            winner_share = f_w / (f_w + f_l)

        attn_won = int(winner_share > 0.5) if winner_share is not None else None

        all_rows.append({
            'year': yr, 'state': state,
            'winner': winner, 'loser': loser,
            'winner_party': row['winner_party'],
            'winner_vote_pct': row['winner_pct'],
            'loser_vote_pct': row['loser_pct'],
            'winner_ngrams_freq': f_w,
            'loser_ngrams_freq': f_l,
            'winner_mention_share': winner_share,
            'attention_leader_won': attn_won,
            'status': status,
        })

        # Save checkpoint every 50 races
        if (i + 1) % 50 == 0:
            pd.DataFrame(all_rows).to_csv(OUT_DIR / 'senate_all_races_checkpoint.csv', index=False)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(OUT_DIR / 'senate_all_races.csv', index=False)

    df_ok = df_all[df_all['status'] == 'ok'].copy()
    df_ok.to_csv(OUT_DIR / 'senate_mention_shares.csv', index=False)

    print(f'\n=== SENATE NGRAMS RESULTS ===')
    print(f'Total races: {len(df_all)}')
    status_counts = df_all['status'].value_counts()
    for s, c in status_counts.items():
        print(f'  {s}: {c}')

    if len(df_ok) >= 5:
        k = df_ok['attention_leader_won'].sum()
        n = len(df_ok)
        result = stats.binomtest(int(k), n, p=0.5, alternative='two-sided')
        print(f'\nH1 (Senate Ngrams, ok-only n={n}): {k}/{n} ({k/n:.1%}), p={result.pvalue:.4f}')

        r, p = pearsonr(df_ok['winner_mention_share'], df_ok['winner_vote_pct'])
        print(f'H2 (Senate Ngrams): r={r:.3f}, p={p:.4f}')

    print('\nSample results (status=ok):')
    print(df_ok[['year','state','winner','winner_mention_share','winner_vote_pct','attention_leader_won']].head(20).to_string(index=False))


if __name__ == '__main__':
    main()
