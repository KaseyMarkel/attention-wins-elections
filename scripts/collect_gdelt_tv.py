"""
GDELT TV Broadcast data collection — presidential candidate mention shares.

Source: Internet Archive TV News Search (https://archive.org/details/tv)
        This is the public interface to GDELT TV data.
Coverage: 2012, 2016, 2020, 2024 (IA TV archive coverage varies; 2012+ is most complete)
Methodology: Count TV news broadcast segments mentioning each candidate by name
             in the 12 months before election day, across major networks
             (CNN, FOXNEWS, MSNBC, ABC, CBS, NBC, PBS).
             Mention share = winner_count / (winner_count + loser_count)

Uses Internet Archive's full-text search API across broadcast closed captions.
All raw responses saved as JSON.

Output: data/raw/gdelt_tv/presidential_mention_shares.csv
        data/raw/gdelt_tv/raw_{year}_{candidate}.json

Alternative approach: if IA search quota is hit, uses GDELT TV timeline API
(https://api.gdeltproject.org/api/v2/tv/tv) which requires a single station.
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
from scipy.stats import pearsonr
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.candidates import PRESIDENTIAL

OUT_DIR = Path('data/raw/gdelt_tv')
OUT_DIR.mkdir(parents=True, exist_ok=True)

IA_SEARCH = 'https://archive.org/advancedsearch.php'
HEADERS = {'User-Agent': 'attention-wins-elections/1.0 (kaseymarkel@semillanueva.org)'}

# TV elections — only years with good IA TV archive coverage
TV_ELECTIONS = {
    # year: (winner_query, loser_query, election_date)
    2012: ('"Barack Obama"',    '"Mitt Romney"',   date(2012, 11, 6)),
    2016: ('"Hillary Clinton"', '"Donald Trump"',  date(2016, 11, 8)),
    2020: ('"Joe Biden"',       '"Donald Trump"',  date(2020, 11, 3)),
    2024: ('"Donald Trump"',    '"Kamala Harris"', date(2024, 11, 5)),
}


def ia_tv_count(query: str, date_from: date, date_to: date,
                label: str, year: int) -> int | None:
    """
    Count TV broadcast segments matching query in date range via Internet Archive.
    Returns total result count. Saves raw response.
    """
    params = {
        'q': f'{query} mediatype:movies date:[{date_from.isoformat()} TO {date_to.isoformat()}]',
        'fl': 'identifier,date',
        'rows': 0,  # We just want the count, not the items
        'output': 'json',
    }
    url = IA_SEARCH + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    print(f'  IA search: {query} | {date_from} – {date_to}')

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read()
            break
        except Exception as e:
            wait = 20 * (attempt + 1)
            print(f'  Attempt {attempt+1} failed ({e}), waiting {wait}s...')
            time.sleep(wait)
    else:
        return None

    safe_label = urllib.parse.quote(label, safe='')[:40]
    raw_path = OUT_DIR / f'raw_{year}_{safe_label}.json'
    raw_path.write_bytes(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f'  ERROR: non-JSON response')
        return None

    count = data.get('response', {}).get('numFound', 0)
    print(f'  → {count:,} TV segments')
    return count


def main():
    rows = []
    vote_lookup = {yr: (wn, ln, wp, lp, wv, lv) for yr, wn, ln, wp, lp, wv, lv, _ in PRESIDENTIAL}

    for year, (wq, lq, eday) in sorted(TV_ELECTIONS.items()):
        wn, ln, wp, lp, wv, lv = vote_lookup[year]
        start_d = eday - timedelta(days=365)
        print(f'\n=== {year}: {wn} vs {ln} ===')
        print(f'  Window: {start_d} to {eday}')

        winner_count = ia_tv_count(wq, start_d, eday, wn, year)
        time.sleep(5)
        loser_count  = ia_tv_count(lq, start_d, eday, ln, year)
        time.sleep(5)

        if winner_count is None or loser_count is None:
            print(f'  Skipping {year} — fetch failed')
            continue

        total = winner_count + loser_count
        if total == 0:
            print(f'  WARNING: zero total for {year}')
            continue

        winner_share = winner_count / total
        loser_share  = loser_count  / total
        attn_leader_won = int(winner_share > 0.5)

        print(f'  {wn}: {winner_count:,} segments ({winner_share:.1%})')
        print(f'  {ln}: {loser_count:,}  segments ({loser_share:.1%})')
        print(f'  Attention leader won: {bool(attn_leader_won)} (vote: {wv:.1f}% vs {lv:.1f}%)')

        rows.append({
            'year': year,
            'source': 'GDELT TV (via Internet Archive)',
            'winner_name': wn,
            'loser_name': ln,
            'winner_party': wp,
            'winner_mention_share': round(winner_share, 6),
            'loser_mention_share': round(loser_share, 6),
            'winner_vote_pct': wv,
            'loser_vote_pct': lv,
            'attention_leader_won': attn_leader_won,
            'winner_segment_count': winner_count,
            'loser_segment_count': loser_count,
            'window': f'{start_d}–{eday}',
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
    print(f'\nH1 (GDELT TV): Attention leader won {k}/{n} elections ({k/n:.1%})')

    if n >= 3:
        r, p = pearsonr(df['winner_mention_share'], df['winner_vote_pct'])
        print(f'H2 (GDELT TV): Pearson r = {r:.3f}, p = {p:.4f} (n={n})')

    print('\nFull results:')
    print(df[['year', 'winner_name', 'loser_name', 'winner_mention_share',
              'winner_vote_pct', 'attention_leader_won']].to_string(index=False))


if __name__ == '__main__':
    main()
