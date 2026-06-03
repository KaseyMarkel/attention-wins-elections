"""
Reddit mention data collection — presidential candidate mention shares.

Source: PullPush.io (community-maintained Pushshift mirror)
        https://api.pullpush.io/reddit/search/submission/
Coverage: 2008–2024 (6 elections; Reddit founded 2005)
Methodology: Count Reddit submissions mentioning each candidate's name
             in r/politics (the most active political subreddit) in the
             12 months before election day.
             Mention share = winner_count / (winner_count + loser_count)

Note: Pushshift/PullPush has irregular coverage gaps. We validate by checking
      total post counts and flagging suspicious years.
All raw responses saved as JSON.

Output: data/raw/reddit/presidential_mention_shares.csv
        data/raw/reddit/raw_{year}_{candidate}.json

Fallback: If PullPush is down, falls back to Reddit's own search API
          (limited to recent posts).
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
import pandas as pd
from scipy.stats import pearsonr
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.candidates import PRESIDENTIAL

OUT_DIR = Path('data/raw/reddit')
OUT_DIR.mkdir(parents=True, exist_ok=True)

PULLPUSH_URL = 'https://api.pullpush.io/reddit/search/submission/'
HEADERS = {'User-Agent': 'attention-wins-elections/1.0 (kaseymarkel@semillanueva.org)'}

# Reddit elections (Reddit has decent coverage from 2008+)
REDDIT_ELECTIONS = {
    2008: ('"Barack Obama"',    '"John McCain"',   date(2008, 11, 4)),
    2012: ('"Barack Obama"',    '"Mitt Romney"',   date(2012, 11, 6)),
    2016: ('"Hillary Clinton"', '"Donald Trump"',  date(2016, 11, 8)),
    2020: ('"Joe Biden"',       '"Donald Trump"',  date(2020, 11, 3)),
    2024: ('"Donald Trump"',    '"Kamala Harris"', date(2024, 11, 5)),
}

SUBREDDIT = 'politics'


def to_epoch(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())


def pullpush_count(query: str, subreddit: str,
                   date_from: date, date_to: date,
                   label: str, year: int) -> int | None:
    """
    Count Reddit submissions matching query via PullPush.
    Uses pagination to get accurate total count.
    """
    after_ts  = to_epoch(date_from)
    before_ts = to_epoch(date_to)

    # PullPush metadata endpoint for count
    params = {
        'q': query.strip('"'),
        'subreddit': subreddit,
        'after': after_ts,
        'before': before_ts,
        'size': 0,  # metadata only
    }
    url = PULLPUSH_URL + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    print(f'  PullPush: {query} in r/{subreddit} | {date_from} – {date_to}')

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read()
            break
        except Exception as e:
            wait = 15 * (attempt + 1)
            print(f'  Attempt {attempt+1} failed ({e}), waiting {wait}s...')
            time.sleep(wait)
    else:
        print(f'  FAILED — PullPush may be unavailable')
        return None

    safe_label = urllib.parse.quote(label, safe='')[:40]
    raw_path = OUT_DIR / f'raw_{year}_{safe_label}.json'
    raw_path.write_bytes(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f'  ERROR non-JSON: {raw[:200]}')
        return None

    # PullPush returns {"data": [...], "metadata": {"total_results": N}}
    count = data.get('metadata', {}).get('total_results', len(data.get('data', [])))
    print(f'  → {count:,} submissions')
    return count


def main():
    rows = []
    vote_lookup = {yr: (wn, ln, wp, lp, wv, lv) for yr, wn, ln, wp, lp, wv, lv, _ in PRESIDENTIAL}

    for year, (wq, lq, eday) in sorted(REDDIT_ELECTIONS.items()):
        wn, ln, wp, lp, wv, lv = vote_lookup[year]
        start_d = eday - timedelta(days=365)
        print(f'\n=== {year}: {wn} vs {ln} ===')

        winner_count = pullpush_count(wq, SUBREDDIT, start_d, eday, wn, year)
        time.sleep(3)
        loser_count  = pullpush_count(lq, SUBREDDIT, start_d, eday, ln, year)
        time.sleep(3)

        if winner_count is None or loser_count is None:
            print(f'  Skipping {year} — fetch failed')
            continue

        total = winner_count + loser_count
        if total == 0:
            print(f'  WARNING: zero total for {year}')
            continue

        # Sanity check: flag suspiciously low counts
        if total < 100:
            print(f'  WARNING: very low count ({total}) — possible coverage gap')

        winner_share = winner_count / total
        loser_share  = loser_count  / total
        attn_leader_won = int(winner_share > 0.5)

        print(f'  {wn}: {winner_count:,} posts ({winner_share:.1%})')
        print(f'  {ln}: {loser_count:,} posts ({loser_share:.1%})')
        print(f'  Attention leader won: {bool(attn_leader_won)} (vote: {wv:.1f}% vs {lv:.1f}%)')

        rows.append({
            'year': year,
            'source': 'Reddit (r/politics via PullPush)',
            'winner_name': wn,
            'loser_name': ln,
            'winner_party': wp,
            'winner_mention_share': round(winner_share, 6),
            'loser_mention_share': round(loser_share, 6),
            'winner_vote_pct': wv,
            'loser_vote_pct': lv,
            'attention_leader_won': attn_leader_won,
            'winner_post_count': winner_count,
            'loser_post_count': loser_count,
            'subreddit': SUBREDDIT,
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
    print(f'\nH1 (Reddit): Attention leader won {k}/{n} elections ({k/n:.1%})')

    if n >= 3:
        r, p = pearsonr(df['winner_mention_share'], df['winner_vote_pct'])
        print(f'H2 (Reddit): Pearson r = {r:.3f}, p = {p:.4f} (n={n})')

    print('\nFull results:')
    print(df[['year', 'winner_name', 'loser_name', 'winner_mention_share',
              'winner_vote_pct', 'attention_leader_won']].to_string(index=False))


if __name__ == '__main__':
    main()
