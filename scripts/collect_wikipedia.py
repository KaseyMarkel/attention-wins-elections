"""
Wikipedia Pageviews data collection — presidential candidate mention shares.

Source: Wikimedia REST API (https://wikimedia.org/api/rest_v1/)
Coverage: 2016, 2020, 2024 (API starts July 2015; only covers last 3 elections)
Methodology: Monthly pageviews for each candidate's Wikipedia article in the
             12 months before election day.
             Mention share = winner_views / (winner_views + loser_views)

Article titles verified against Wikipedia redirects.
All raw API responses saved as JSON.

Output: data/raw/wikipedia/presidential_mention_shares.csv
        data/raw/wikipedia/raw_{year}_{candidate}.json
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

OUT_DIR = Path('data/raw/wikipedia')
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIKI_BASE = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents'
HEADERS = {'User-Agent': 'attention-wins-elections/1.0 (kaseymarkel@semillanueva.org)'}

# Wikipedia article titles for each candidate-year
# Must match exact Wikipedia article title (with underscores)
WIKI_ARTICLES = {
    # year: (winner_article, loser_article)
    2016: ('Hillary_Clinton',   'Donald_Trump'),
    2020: ('Joe_Biden',         'Donald_Trump'),
    2024: ('Donald_Trump',      'Kamala_Harris'),
}

ELECTION_DAYS = {
    2016: ('20151108', '20161108'),  # 12-month window ending on election day
    2020: ('20191103', '20201103'),
    2024: ('20231105', '20241105'),
}


def fetch_monthly_views(article: str, start_yyyymm: str, end_yyyymm: str) -> dict:
    """
    Fetch monthly pageviews for a Wikipedia article.
    start/end format: YYYYMM → API uses YYYYMMDD so we pad with 01/30.
    Returns {yyyymm: views} dict.
    """
    # API format: YYYYMMDD00 (daily) or use monthly endpoint
    url = f'{WIKI_BASE}/{urllib.parse.quote(article)}/monthly/{start_yyyymm}01/{end_yyyymm}30'
    req = urllib.request.Request(url, headers=HEADERS)
    print(f'  GET .../{article}/monthly/{start_yyyymm}01/{end_yyyymm}30')

    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read()

    data = json.loads(raw)
    items = data.get('items', [])

    result = {}
    for item in items:
        ym = item['timestamp'][:6]  # YYYYMM
        result[ym] = item['views']

    return result, raw


def main():
    rows = []
    vote_lookup = {yr: (wn, ln, wp, lp, wv, lv) for yr, wn, ln, wp, lp, wv, lv, _ in PRESIDENTIAL}

    for year, (winner_article, loser_article) in sorted(WIKI_ARTICLES.items()):
        wn, ln, wp, lp, wv, lv = vote_lookup[year]
        start_ym, end_ym = ELECTION_DAYS[year]
        start_api = start_ym[:6]   # YYYYMM
        end_api   = end_ym[:6]

        print(f'\n=== {year}: {wn} ({winner_article}) vs {ln} ({loser_article}) ===')
        print(f'  Window: {start_api} to {end_api}')

        try:
            winner_views, raw_w = fetch_monthly_views(winner_article, start_api, end_api)
            raw_path = OUT_DIR / f'raw_{year}_{winner_article}.json'
            raw_path.write_bytes(raw_w)

            time.sleep(1)

            loser_views, raw_l = fetch_monthly_views(loser_article, start_api, end_api)
            raw_path = OUT_DIR / f'raw_{year}_{loser_article}.json'
            raw_path.write_bytes(raw_l)

        except Exception as e:
            print(f'  ERROR: {e}')
            continue

        w_total = sum(winner_views.values())
        l_total = sum(loser_views.values())
        grand_total = w_total + l_total

        if grand_total == 0:
            print('  WARNING: zero total views')
            continue

        winner_share = w_total / grand_total
        loser_share  = l_total / grand_total
        attn_leader_won = int(winner_share > 0.5)

        print(f'  {wn}: {w_total:,} views ({winner_share:.1%})')
        print(f'  {ln}: {l_total:,} views ({loser_share:.1%})')
        print(f'  Months covered: {len(winner_views)} winner, {len(loser_views)} loser')
        print(f'  Attention leader won: {bool(attn_leader_won)} (vote: {wv:.1f}% vs {lv:.1f}%)')

        rows.append({
            'year': year,
            'source': 'Wikipedia Pageviews',
            'winner_name': wn,
            'winner_article': winner_article,
            'loser_name': ln,
            'loser_article': loser_article,
            'winner_party': wp,
            'winner_mention_share': round(winner_share, 6),
            'loser_mention_share': round(loser_share, 6),
            'winner_vote_pct': wv,
            'loser_vote_pct': lv,
            'attention_leader_won': attn_leader_won,
            'winner_total_views': w_total,
            'loser_total_views': l_total,
            'n_months_winner': len(winner_views),
            'n_months_loser': len(loser_views),
        })

        time.sleep(2)

    if not rows:
        print('No data collected!')
        return

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / 'presidential_mention_shares.csv'
    df.to_csv(out_path, index=False)
    print(f'\nSaved {len(df)} rows to {out_path}')

    k = df['attention_leader_won'].sum()
    n = len(df)
    print(f'\nH1 (Wikipedia): Attention leader won {k}/{n} elections ({k/n:.1%})')

    if n >= 3:
        r, p = pearsonr(df['winner_mention_share'], df['winner_vote_pct'])
        print(f'H2 (Wikipedia): Pearson r = {r:.3f}, p = {p:.4f} (n={n}) — NOTE: n=3, very low power')

    print('\nFull results:')
    print(df[['year', 'winner_name', 'loser_name', 'winner_mention_share',
              'winner_vote_pct', 'attention_leader_won']].to_string(index=False))


if __name__ == '__main__':
    main()
