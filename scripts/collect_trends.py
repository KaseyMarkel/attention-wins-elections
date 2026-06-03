"""
Google Trends data collection — presidential candidate mention shares.

Source: Google Trends (https://trends.google.com)
Coverage: 2004–2024 (6 elections: 2004, 2008, 2012, 2016, 2020, 2024)
Methodology: Compare weekly search interest for both candidates simultaneously
             in the same query (so scores are directly comparable).
             Mention share = sum(candidate_score) / sum(both_scores) over 52-week window.
             Google normalizes to max=100 within each query — querying both candidates
             together gives their relative interest directly.

Rate limit: pytrends has a daily quota; script sleeps between queries.
All raw responses saved as CSV for reproducibility.

Disambiguation:
- 2000: "Al Gore" vs "George W. Bush" (disambiguation via full name)
- 1992/1996: George Bush → "George H.W. Bush" for disambiguation

Output: data/raw/trends/presidential_mention_shares.csv
        data/raw/trends/raw_{year}.csv
"""

import time
import json
from pathlib import Path
import pandas as pd
from pytrends.request import TrendReq
import sys
from scipy import stats
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.candidates import PRESIDENTIAL

OUT_DIR = Path('data/raw/trends')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pytrends search terms — use the most recognizable name form
# Querying both candidates together ensures scores are directly comparable
TRENDS_TERMS = {
    2004: ('George W. Bush', 'John Kerry'),
    2008: ('Barack Obama',   'John McCain'),
    2012: ('Barack Obama',   'Mitt Romney'),
    2016: ('Hillary Clinton','Donald Trump'),
    2020: ('Joe Biden',      'Donald Trump'),
    2024: ('Donald Trump',   'Kamala Harris'),
}

# Election day for each year (first Tuesday after first Monday in November)
ELECTION_DAYS = {
    2004: '2004-11-02',
    2008: '2008-11-04',
    2012: '2012-11-06',
    2016: '2016-11-08',
    2020: '2020-11-03',
    2024: '2024-11-05',
}


def collect_trends_for_election(year: int, cand1: str, cand2: str,
                                 election_day: str) -> dict | None:
    """
    Fetch 52-week Google Trends data for two candidates.
    Returns dict with mention shares and raw data path.
    """
    import datetime
    ed = datetime.date.fromisoformat(election_day)
    start = ed.replace(year=ed.year - 1) + datetime.timedelta(days=1)
    timeframe = f'{start.isoformat()} {election_day}'

    print(f'  {year}: fetching "{cand1}" vs "{cand2}" | {timeframe}')

    pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 30), retries=3, backoff_factor=0.5)

    for attempt in range(3):
        try:
            pytrends.build_payload([cand1, cand2], timeframe=timeframe, geo='US', gprop='')
            df = pytrends.interest_over_time()
            break
        except Exception as e:
            print(f'  Attempt {attempt+1} failed: {e}')
            time.sleep(30 * (attempt + 1))
    else:
        print(f'  FAILED after 3 attempts for {year}')
        return None

    if df.empty:
        print(f'  WARNING: empty response for {year}')
        return None

    # Remove 'isPartial' column if present
    df = df.drop(columns=['isPartial'], errors='ignore')

    # Save raw data
    raw_path = OUT_DIR / f'raw_{year}.csv'
    df.to_csv(raw_path)
    print(f'  Saved raw data: {raw_path} ({len(df)} weeks)')

    c1_sum = df[cand1].sum()
    c2_sum = df[cand2].sum()
    total = c1_sum + c2_sum

    if total == 0:
        print(f'  WARNING: zero total for {year}')
        return None

    c1_share = c1_sum / total
    c2_share = c2_sum / total

    print(f'  {cand1}: sum={c1_sum}, share={c1_share:.1%}')
    print(f'  {cand2}: sum={c2_sum}, share={c2_share:.1%}')

    return {
        'c1_sum': c1_sum, 'c2_sum': c2_sum,
        'c1_share': c1_share, 'c2_share': c2_share,
        'n_weeks': len(df),
        'timeframe': timeframe,
    }


def main():
    rows = []

    # Build lookup for vote shares
    vote_lookup = {yr: (wn, ln, wp, wv, lv) for yr, wn, ln, wp, lp, wv, lv, _ in PRESIDENTIAL}

    for year in sorted(TRENDS_TERMS.keys()):
        cand1, cand2 = TRENDS_TERMS[year]
        election_day = ELECTION_DAYS[year]
        wn, ln, wp, wv, lv = vote_lookup[year]

        print(f'\n=== {year}: {cand1} vs {cand2} ===')

        result = collect_trends_for_election(year, cand1, cand2, election_day)
        if result is None:
            continue

        # Identify which Trends candidate is winner/loser
        # (terms may not be in winner-first order)
        if cand1 == wn or (wn in cand1) or (cand1 in wn):
            winner_share = result['c1_share']
            loser_share  = result['c2_share']
        else:
            winner_share = result['c2_share']
            loser_share  = result['c1_share']

        attn_leader_won = int(winner_share > 0.5)
        print(f'  Winner ({wn}) share: {winner_share:.1%} — attention leader won: {bool(attn_leader_won)}')

        rows.append({
            'year': year,
            'source': 'Google Trends',
            'winner_name': wn,
            'loser_name': ln,
            'winner_party': wp,
            'winner_mention_share': round(winner_share, 6),
            'loser_mention_share': round(loser_share, 6),
            'winner_vote_pct': wv,
            'loser_vote_pct': lv,
            'attention_leader_won': attn_leader_won,
            'winner_raw_sum': result['c1_sum'] if (cand1 == wn or wn in cand1) else result['c2_sum'],
            'loser_raw_sum':  result['c2_sum'] if (cand1 == wn or wn in cand1) else result['c1_sum'],
            'n_weeks': result['n_weeks'],
            'timeframe': result['timeframe'],
        })

        time.sleep(15)  # Respect rate limits between elections

    if not rows:
        print('No data collected!')
        return

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / 'presidential_mention_shares.csv'
    df.to_csv(out_path, index=False)
    print(f'\nSaved {len(df)} rows to {out_path}')

    k = df['attention_leader_won'].sum()
    n = len(df)
    print(f'\nH1 (Google Trends): Attention leader won {k}/{n} elections ({k/n:.1%})')
    result = stats.binomtest(k, n, p=0.5, alternative='two-sided')
    print(f'Binomial test p = {result.pvalue:.4f}')

    r, p = pearsonr(df['winner_mention_share'], df['winner_vote_pct'])
    print(f'H2 (Google Trends): Pearson r = {r:.3f}, p = {p:.4f} (n={n})')

    print('\nFull results:')
    print(df[['year', 'winner_name', 'loser_name', 'winner_mention_share',
              'winner_vote_pct', 'attention_leader_won']].to_string(index=False))


if __name__ == '__main__':
    main()
