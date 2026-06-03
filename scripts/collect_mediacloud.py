"""
MediaCloud data collection — presidential candidate mention shares.

Source: MediaCloud (https://mediacloud.org/) via their public API
Coverage: 2012–2024 (API coverage varies; requires free API key)
Methodology: Count news stories mentioning each candidate in US national sources
             (MediaCloud "US Mainstream Media" collection) in 12 months before election.
             Mention share = winner_count / (winner_count + loser_count)

API key: Set environment variable MEDIACLOUD_API_KEY or pass via --api-key flag.
         Get a free key at https://search.mediacloud.org/

If no API key is available, this script attempts the MediaCloud Explorer
search API which has limited unauthenticated access.

Output: data/raw/mediacloud/presidential_mention_shares.csv
        data/raw/mediacloud/raw_{year}_{candidate}.json
"""

import json
import os
import time
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import date, timedelta
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.candidates import PRESIDENTIAL

OUT_DIR = Path('data/raw/mediacloud')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MediaCloud API v4
MC_BASE = 'https://api.mediacloud.org/api/v2'
HEADERS = {'User-Agent': 'attention-wins-elections/1.0 (kaseymarkel@semillanueva.org)'}

# "US Mainstream Media" collection ID
# Source: https://sources.mediacloud.org/#/collections/34412234
US_MAINSTREAM_COLLECTION = 34412234

MC_ELECTIONS = {
    2012: ('"Barack Obama"',    '"Mitt Romney"',   date(2012, 11, 6)),
    2016: ('"Hillary Clinton"', '"Donald Trump"',  date(2016, 11, 8)),
    2020: ('"Joe Biden"',       '"Donald Trump"',  date(2020, 11, 3)),
    2024: ('"Donald Trump"',    '"Kamala Harris"', date(2024, 11, 5)),
}


def mediacloud_count(query: str, collection_id: int,
                     date_from: date, date_to: date,
                     label: str, year: int,
                     api_key: str | None = None) -> int | None:
    """
    Count MediaCloud stories matching query in date range.
    Falls back to public search endpoint if no API key.
    """
    params = {
        'q': f'{query} publish_date:[{date_from.isoformat()} TO {date_to.isoformat()}]',
        'fq': f'tags_id_media:{collection_id}',
        'rows': 0,
    }
    if api_key:
        params['key'] = api_key

    url = f'{MC_BASE}/stories_public/count?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    print(f'  MediaCloud: {query} | {date_from} – {date_to}')

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read()
            break
        except Exception as e:
            print(f'  Attempt {attempt+1} failed: {e}')
            time.sleep(15)
    else:
        print(f'  FAILED')
        return None

    safe_label = urllib.parse.quote(label, safe='')[:40]
    raw_path = OUT_DIR / f'raw_{year}_{safe_label}.json'
    raw_path.write_bytes(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f'  ERROR non-JSON: {raw[:200]}')
        return None

    count = data.get('count', 0)
    print(f'  → {count:,} stories')
    return count


def main():
    api_key = os.environ.get('MEDIACLOUD_API_KEY')
    if not api_key:
        print('WARNING: MEDIACLOUD_API_KEY not set. Attempting unauthenticated access.')
        print('         Set this env var to get full access: export MEDIACLOUD_API_KEY=your_key')
        print('         Get a free key at https://search.mediacloud.org/')

    import pandas as pd
    rows = []
    vote_lookup = {yr: (wn, ln, wp, lp, wv, lv) for yr, wn, ln, wp, lp, wv, lv, _ in PRESIDENTIAL}

    for year, (wq, lq, eday) in sorted(MC_ELECTIONS.items()):
        wn, ln, wp, lp, wv, lv = vote_lookup[year]
        start_d = eday - timedelta(days=365)
        print(f'\n=== {year}: {wn} vs {ln} ===')

        winner_count = mediacloud_count(wq, US_MAINSTREAM_COLLECTION, start_d, eday, wn, year, api_key)
        time.sleep(3)
        loser_count  = mediacloud_count(lq, US_MAINSTREAM_COLLECTION, start_d, eday, ln, year, api_key)
        time.sleep(3)

        if winner_count is None or loser_count is None:
            print(f'  Skipping {year}')
            continue

        total = winner_count + loser_count
        if total == 0:
            print(f'  WARNING: zero total — API may require authentication')
            continue

        winner_share = winner_count / total
        loser_share  = loser_count  / total
        attn_leader_won = int(winner_share > 0.5)

        print(f'  {wn}: {winner_count:,} ({winner_share:.1%})')
        print(f'  {ln}: {loser_count:,} ({loser_share:.1%})')
        print(f'  Attention leader won: {bool(attn_leader_won)}')

        rows.append({
            'year': year,
            'source': 'MediaCloud',
            'winner_name': wn,
            'loser_name': ln,
            'winner_party': wp,
            'winner_mention_share': round(winner_share, 6),
            'loser_mention_share': round(loser_share, 6),
            'winner_vote_pct': wv,
            'loser_vote_pct': lv,
            'attention_leader_won': attn_leader_won,
            'winner_story_count': winner_count,
            'loser_story_count': loser_count,
            'authenticated': bool(api_key),
        })

    if not rows:
        print('\nNo MediaCloud data collected — likely requires API key.')
        print('This source will be marked as pending in the compiled output.')
        return

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / 'presidential_mention_shares.csv'
    df.to_csv(out_path, index=False)
    print(f'\nSaved {len(df)} rows to {out_path}')


if __name__ == '__main__':
    main()
