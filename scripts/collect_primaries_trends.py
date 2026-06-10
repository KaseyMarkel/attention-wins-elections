"""
Presidential PRIMARIES — the purest test of attention vs. outcome.

Primaries are essentially name-recognition contests within a single party, so the
usual "voters like my party" confound is largely removed: every candidate shares
the voter's party. Does the candidate with the most attention win the nomination?

Attention metric: Google Trends search interest, averaged over the "invisible
primary" window — roughly the 7 months BEFORE the first nominating contest
(Jul 1 of the prior year → Jan 31 of the election year). Measuring before any
votes are cast keeps the test predictive rather than circular (we are not just
re-measuring who started winning Iowa).

Each contested race uses up to 5 major candidates (Trends caps a comparison at
5 terms and normalises within the query). The eventual nominee is listed first.

Output: data/raw/trends/primaries_mention_shares.csv
"""

import time
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq
from scipy.stats import binomtest

OUT_DIR = Path('data/raw/trends')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (cycle label, party, [candidates — nominee FIRST], invisible-primary window)
RACES = [
    ('2008', 'D', ['Barack Obama', 'Hillary Clinton', 'John Edwards'],
     '2007-07-01 2008-01-31'),
    ('2008', 'R', ['John McCain', 'Mitt Romney', 'Mike Huckabee', 'Rudy Giuliani'],
     '2007-07-01 2008-01-31'),
    ('2012', 'R', ['Mitt Romney', 'Newt Gingrich', 'Rick Santorum', 'Ron Paul'],
     '2011-07-01 2012-01-31'),
    ('2016', 'R', ['Donald Trump', 'Ted Cruz', 'Marco Rubio', 'Jeb Bush', 'Ben Carson'],
     '2015-07-01 2016-01-31'),
    ('2016', 'D', ['Hillary Clinton', 'Bernie Sanders'],
     '2015-07-01 2016-01-31'),
    ('2020', 'D', ['Joe Biden', 'Bernie Sanders', 'Elizabeth Warren', 'Pete Buttigieg', 'Kamala Harris'],
     '2019-07-01 2020-01-31'),
    ('2024', 'R', ['Donald Trump', 'Ron DeSantis', 'Nikki Haley'],
     '2023-07-01 2024-01-31'),
]


def interest(candidates, timeframe):
    """Mean Google Trends interest per candidate over the window."""
    for attempt in range(4):
        try:
            p = TrendReq(hl='en-US', tz=360)
            p.build_payload(candidates, timeframe=timeframe)
            df = p.interest_over_time()
            if df.empty:
                return None
            df = df.drop(columns=['isPartial'], errors='ignore')
            return df[candidates].mean()
        except Exception as e:
            print(f'    retry ({type(e).__name__}); sleeping')
            time.sleep(15 * (attempt + 1))
    return None


def main():
    rows = []
    for cycle, party, cands, tf in RACES:
        nominee = cands[0]
        print(f'=== {cycle} {party} primary: {", ".join(cands)} ===')
        means = interest(cands, tf)
        time.sleep(3)
        if means is None:
            print('    FAILED — skipping')
            continue
        total = means.sum()
        leader = means.idxmax()
        for c in cands:
            rows.append({
                'cycle': cycle, 'party': party, 'candidate': c,
                'is_nominee': int(c == nominee),
                'search_interest': round(float(means[c]), 2),
                'attention_share': round(float(means[c] / total), 4) if total else 0.0,
                'window': tf,
            })
        nominee_share = means[nominee] / total if total else 0
        print(f'    most-searched: {leader}  |  nominee: {nominee} '
              f'(share {nominee_share:.0%})  |  attention picked nominee: {leader == nominee}')

    df = pd.DataFrame(rows)
    out = OUT_DIR / 'primaries_mention_shares.csv'
    df.to_csv(out, index=False)
    print(f'\nSaved {len(df)} candidate-rows ({df[["cycle","party"]].drop_duplicates().shape[0]} races) to {out}')

    # H1: in how many races did the most-searched candidate become the nominee?
    races = df.groupby(['cycle', 'party'])
    won = 0
    for (cyc, pty), g in races:
        leader = g.loc[g['search_interest'].idxmax(), 'candidate']
        nominee = g.loc[g['is_nominee'] == 1, 'candidate'].iloc[0]
        won += int(leader == nominee)
    n = races.ngroups
    p = binomtest(won, n, 0.5, alternative='two-sided').pvalue
    print(f'\nH1 (primaries): attention leader won the nomination in {won}/{n} '
          f'contested races ({won/n:.0%}), p={p:.4f}')


if __name__ == '__main__':
    main()
