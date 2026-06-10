"""
Google Ngrams data collection — presidential candidate mention shares.

Source: Google Books Ngram Viewer (https://books.google.com/ngrams)
Coverage: 1960–2019 (corpus en-2019; 2020/2024 elections not covered)
Methodology: Sum annual frequency (per-million-words) of candidate's full name
             in the book-year matching the election year window.
             Mention share = winner_freq / (winner_freq + loser_freq)
             Window: the 12 calendar months ending on election day (Nov of election year)
             → books data is annual, so we use the election year and year-1.

Disambiguation notes:
- "George Bush" vs "George H.W. Bush" / "George W. Bush": use full names where
  possible; Ngrams doesn't always have the extended form. See edge_cases.md.
- All raw API responses saved as JSON for reproducibility.

Output: data/raw/ngrams/presidential_mention_shares.csv
        data/raw/ngrams/raw_{year}_{candidate}.json   (one per candidate-year)
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.candidates import PRESIDENTIAL

OUT_DIR = Path('data/raw/ngrams')
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG = []


def ngrams_query(terms: list[str], year_start: int, year_end: int,
                 corpus: str = 'en-2019') -> dict:
    """
    Query Google Ngrams JSON API.
    Returns dict: {term: {year: freq_per_million, ...}}
    Saves raw response to disk for auditability.
    """
    content = ','.join(terms)
    params = {
        'content': content,
        'year_start': year_start,
        'year_end': year_end,
        'corpus': corpus,
        'smoothing': 0,
    }
    url = 'https://books.google.com/ngrams/json?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url, headers={'User-Agent': 'attention-wins-elections/1.0 (kaseymarkel@semillanueva.org)'}
    )
    print(f'  GET {url[:120]}...')
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read()
    data = json.loads(raw)

    # Save raw response
    safe = urllib.parse.quote(content, safe='').replace('%', '_')[:60]
    raw_path = OUT_DIR / f'raw_{year_start}_{year_end}_{safe}.json'
    raw_path.write_bytes(raw)

    result = {}
    expected = year_end - year_start + 1
    for series in data:
        name = series['ngram']
        ts = series['timeseries']
        # Ngrams quirk: when year_start == year_end the API ignores the range
        # and returns the full corpus series (1500–2019). Guard against any
        # length mismatch so years never silently misalign with values.
        if len(ts) != expected:
            raise ValueError(
                f'Ngrams returned {len(ts)} points for {name!r} but the '
                f'window {year_start}-{year_end} expects {expected}. '
                f'Use a multi-year window (year_start < year_end).'
            )
        years = list(range(year_start, year_end + 1))
        result[name] = dict(zip(years, ts))
    return result


def get_candidate_freq(name: str, year: int) -> float:
    """
    Return sum of annual Ngrams frequency for `name` in year-1 and year
    (proxy for 12-month pre-election window).
    Returns 0.0 if name not found or outside corpus coverage.
    """
    # Corpus ends ~2019 — skip elections beyond that
    if year > 2019:
        return None

    # Always use a 2-year window (year-1, year). Never let year_start == year_end:
    # the Ngrams API treats a single-year range as "return the whole corpus",
    # which previously zeroed out 1960 (Kennedy/Nixon). The en-2019 corpus
    # covers 1959, so the floor that caused this is removed.
    year_start = year - 1
    year_end = min(2019, year)

    time.sleep(1.0)  # be polite to the API
    try:
        data = ngrams_query([name], year_start, year_end)
    except Exception as e:
        print(f'  ERROR fetching {name!r} for {year}: {e}')
        return None

    freq_series = data.get(name, {})
    if not freq_series:
        print(f'  WARNING: no Ngrams data for {name!r}')
        return 0.0

    # Sum frequency across the window years
    total = sum(freq_series.get(y, 0.0) for y in range(year_start, year_end + 1))
    return total


def main():
    rows = []

    for yr, wn, ln, wp, lp, wv, lv, inc in PRESIDENTIAL:
        print(f'\n=== {yr}: {wn} vs {ln} ===')

        # Ngrams corpus ends at 2019 — skip 2020 and 2024
        if yr > 2019:
            print(f'  Skipping {yr} (beyond Ngrams en-2019 corpus)')
            continue

        winner_freq = get_candidate_freq(wn, yr)
        loser_freq  = get_candidate_freq(ln, yr)

        if winner_freq is None or loser_freq is None:
            print(f'  Skipping {yr} — missing data')
            continue

        total = winner_freq + loser_freq
        if total == 0:
            print(f'  WARNING: both candidates have zero frequency in {yr}')
            winner_share = 0.5
        else:
            winner_share = winner_freq / total

        loser_share = 1.0 - winner_share
        attn_leader_won = int(winner_share > 0.5)

        print(f'  {wn}: freq={winner_freq:.6f}  share={winner_share:.1%}')
        print(f'  {ln}: freq={loser_freq:.6f}  share={loser_share:.1%}')
        print(f'  Attention leader won: {bool(attn_leader_won)} '
              f'(vote: {wv:.1f}% vs {lv:.1f}%)')

        rows.append({
            'year': yr,
            'source': 'Google Ngrams',
            'winner_name': wn,
            'loser_name': ln,
            'winner_party': wp,
            'winner_mention_share': round(winner_share, 6),
            'loser_mention_share': round(loser_share, 6),
            'winner_vote_pct': wv,
            'loser_vote_pct': lv,
            'attention_leader_won': attn_leader_won,
            'winner_raw_freq': winner_freq,
            'loser_raw_freq': loser_freq,
            'corpus': 'en-2019',
            'query_window': f'{yr-1}-{yr}',
        })

        LOG.append({'year': yr, 'source': 'ngrams', 'winner_share': winner_share})

    if not rows:
        print('No data collected!')
        return

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / 'presidential_mention_shares.csv'
    df.to_csv(out_path, index=False)
    print(f'\nSaved {len(df)} rows to {out_path}')

    # Summary stats
    k = df['attention_leader_won'].sum()
    n = len(df)
    print(f'\nH1 (Ngrams): Attention leader won {k}/{n} elections ({k/n:.1%})')
    from scipy import stats
    result = stats.binomtest(k, n, p=0.5, alternative='two-sided')
    print(f'Binomial test p = {result.pvalue:.4f}')

    from scipy.stats import pearsonr
    r, p = pearsonr(df['winner_mention_share'], df['winner_vote_pct'])
    print(f'H2 (Ngrams): Pearson r = {r:.3f}, p = {p:.4f} (n={n})')

    print('\nFull results:')
    print(df[['year', 'winner_name', 'loser_name', 'winner_mention_share',
              'winner_vote_pct', 'attention_leader_won']].to_string(index=False))


if __name__ == '__main__':
    main()
