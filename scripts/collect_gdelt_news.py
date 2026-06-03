"""
GDELT News data collection — presidential candidate mention shares.

Source: GDELT Project v2 Full-Text Search API (https://api.gdeltproject.org/api/v2/doc/doc)
Coverage: 2016, 2020, 2024 (GDELT full-text API coverage begins ~Feb 18, 2015)
Methodology: Query article volume mentioning each candidate's name in the
             12 months before election day. Use quoted exact-phrase search.
             Mention share = winner_vol / (winner_vol + loser_vol)

Rate limits: GDELT enforces strict rate limits (~1 query / 5s).
             Script sleeps generously between queries.
All raw JSON responses saved for reproducibility.

Output: data/raw/gdelt_news/presidential_mention_shares.csv
        data/raw/gdelt_news/raw_{year}_{candidate}.json
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.candidates import PRESIDENTIAL

OUT_DIR = Path('data/raw/gdelt_news')
OUT_DIR.mkdir(parents=True, exist_ok=True)

GDELT_API = 'https://api.gdeltproject.org/api/v2/doc/doc'
HEADERS = {'User-Agent': 'attention-wins-elections/1.0 (kaseymarkel@semillanueva.org)'}

# GDELT only covers from ~2015-02-18; only 2016, 2020, 2024 have full 12-month windows
GDELT_ELECTIONS = {
    # year: (winner_query, loser_query, start_yyyymmdd, end_yyyymmdd)
    2016: ('"Hillary Clinton"', '"Donald Trump"',  '20151108', '20161108'),
    2020: ('"Joe Biden"',       '"Donald Trump"',  '20191103', '20201103'),
    2024: ('"Donald Trump"',    '"Kamala Harris"', '20231105', '20241105'),
}


def gdelt_timeline_volume(query: str, start_date: str, end_date: str,
                           label: str, year: int) -> float | None:
    """
    Query GDELT timelinevol for a search term.
    Returns total article volume sum. Saves raw JSON.
    """
    params = {
        'query': query,
        'mode': 'timelinevol',
        'format': 'json',
        'startdatetime': start_date + '000000',
        'enddatetime':   end_date   + '000000',
        'smoothing': 0,
    }
    url = GDELT_API + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    print(f'  GET {query} | {start_date}–{end_date}')

    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
            break
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f'  Attempt {attempt+1} failed ({e}), waiting {wait}s...')
            time.sleep(wait)
    else:
        print(f'  FAILED after 4 attempts')
        return None

    # Save raw response
    safe_label = urllib.parse.quote(label, safe='')[:40]
    raw_path = OUT_DIR / f'raw_{year}_{safe_label}.json'
    raw_path.write_bytes(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f'  ERROR: non-JSON response: {raw[:200]}')
        return None

    timeline = data.get('timeline', [{}])[0].get('data', [])
    total = sum(item.get('value', 0) for item in timeline)
    print(f'  → {total:,} article-mentions across {len(timeline)} time points')
    return total


def main():
    rows = []
    vote_lookup = {yr: (wn, ln, wp, lp, wv, lv) for yr, wn, ln, wp, lp, wv, lv, _ in PRESIDENTIAL}

    for year, (wq, lq, start_d, end_d) in sorted(GDELT_ELECTIONS.items()):
        wn, ln, wp, lp, wv, lv = vote_lookup[year]
        print(f'\n=== {year}: {wn} vs {ln} ===')

        winner_vol = gdelt_timeline_volume(wq, start_d, end_d, wn, year)
        time.sleep(8)  # Respect rate limits strictly
        loser_vol  = gdelt_timeline_volume(lq, start_d, end_d, ln, year)
        time.sleep(8)

        if winner_vol is None or loser_vol is None:
            print(f'  Skipping {year} — data fetch failed')
            continue

        total = winner_vol + loser_vol
        if total == 0:
            print(f'  WARNING: zero total for {year}')
            continue

        winner_share = winner_vol / total
        loser_share  = loser_vol  / total
        attn_leader_won = int(winner_share > 0.5)

        print(f'  {wn}: {winner_vol:,} ({winner_share:.1%})')
        print(f'  {ln}: {loser_vol:,} ({loser_share:.1%})')
        print(f'  Attention leader won: {bool(attn_leader_won)} (vote: {wv:.1f}% vs {lv:.1f}%)')

        rows.append({
            'year': year,
            'source': 'GDELT News',
            'winner_name': wn,
            'loser_name': ln,
            'winner_party': wp,
            'winner_mention_share': round(winner_share, 6),
            'loser_mention_share': round(loser_share, 6),
            'winner_vote_pct': wv,
            'loser_vote_pct': lv,
            'attention_leader_won': attn_leader_won,
            'winner_article_volume': winner_vol,
            'loser_article_volume': loser_vol,
            'window': f'{start_d}–{end_d}',
        })

    if not rows:
        print('No data collected!')
        return

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / 'presidential_mention_shares.csv'
    df.to_csv(out_path, index=False)
    print(f'\nSaved {len(df)} rows to {out_path}')

    k = df['attention_leader_won'].sum()
    n = len(df)
    print(f'\nH1 (GDELT News): Attention leader won {k}/{n} elections ({k/n:.1%})')
    print('NOTE: n=3 elections — very low statistical power; treat as descriptive')

    print('\nFull results:')
    print(df[['year', 'winner_name', 'loser_name', 'winner_mention_share',
              'winner_vote_pct', 'attention_leader_won']].to_string(index=False))


if __name__ == '__main__':
    main()
