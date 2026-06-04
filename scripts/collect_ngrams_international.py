"""
Google Ngrams — UK general elections + Australian federal elections.

Queries both leaders in one Ngrams request for the 2 years ending on election day.
Winner = whoever formed government (became PM), regardless of vote-share ties.

Output:
  data/raw/ngrams/uk_mention_shares.csv
  data/raw/ngrams/australia_mention_shares.csv
"""

import json, time, urllib.request, urllib.parse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr

OUT_DIR = Path('data/raw/ngrams')
OUT_DIR.mkdir(parents=True, exist_ok=True)

NGRAMS_URL = 'https://books.google.com/ngrams/json'
HEADERS = {'User-Agent': 'attention-wins-elections/1.0 (kaseymarkel@semillanueva.org)'}

# ── UK General Elections ───────────────────────────────────────────────────────
# (year, winner_name, loser_name, winner_party, loser_party,
#  winner_vote_pct, loser_vote_pct)
# winner = whoever became PM; vote_pct = national popular vote share (%).
# Marked with * where vote-share winner ≠ seat winner (FPTP inversion).
UK_ELECTIONS = [
    (1945, 'Clement Attlee',    'Winston Churchill', 'Labour',       'Conservative', 47.7, 39.8),
    (1950, 'Clement Attlee',    'Winston Churchill', 'Labour',       'Conservative', 46.1, 43.5),
    (1951, 'Winston Churchill', 'Clement Attlee',    'Conservative', 'Labour',       48.0, 48.8),  # *vote inversion
    (1955, 'Anthony Eden',      'Clement Attlee',    'Conservative', 'Labour',       49.7, 46.4),
    (1959, 'Harold Macmillan',  'Hugh Gaitskell',    'Conservative', 'Labour',       49.4, 43.8),
    (1964, 'Harold Wilson',     'Alec Douglas-Home', 'Labour',       'Conservative', 44.1, 43.4),
    (1966, 'Harold Wilson',     'Edward Heath',      'Labour',       'Conservative', 47.9, 41.9),
    (1970, 'Edward Heath',      'Harold Wilson',     'Conservative', 'Labour',       46.4, 43.0),
    (1974, 'Harold Wilson',     'Edward Heath',      'Labour',       'Conservative', 37.1, 37.9),  # *Feb 74 hung parliament
    (1979, 'Margaret Thatcher', 'James Callaghan',   'Conservative', 'Labour',       43.9, 36.9),
    (1983, 'Margaret Thatcher', 'Neil Kinnock',      'Conservative', 'Labour',       42.4, 27.6),
    (1987, 'Margaret Thatcher', 'Neil Kinnock',      'Conservative', 'Labour',       42.2, 30.8),
    (1992, 'John Major',        'Neil Kinnock',      'Conservative', 'Labour',       41.9, 34.4),
    (1997, 'Tony Blair',        'John Major',        'Labour',       'Conservative', 43.2, 30.7),
    (2001, 'Tony Blair',        'William Hague',     'Labour',       'Conservative', 40.7, 31.7),
    (2005, 'Tony Blair',        'Michael Howard',    'Labour',       'Conservative', 35.2, 32.4),
    (2010, 'David Cameron',     'Gordon Brown',      'Conservative', 'Labour',       36.1, 29.0),
    (2015, 'David Cameron',     'Ed Miliband',       'Conservative', 'Labour',       36.9, 30.4),
    (2017, 'Theresa May',       'Jeremy Corbyn',     'Conservative', 'Labour',       42.4, 40.0),
    (2019, 'Boris Johnson',     'Jeremy Corbyn',     'Conservative', 'Labour',       43.6, 32.2),
]

# ── Australian Federal Elections ───────────────────────────────────────────────
# winner = PM after election; vote_pct = 2-party-preferred (2PP) %.
# * = seat winner ≠ 2PP winner.
AUS_ELECTIONS = [
    (1949, 'Robert Menzies',   'Ben Chifley',     'Liberal', 'Labor',   51.0, 49.0),
    (1951, 'Robert Menzies',   'Herbert Evatt',   'Liberal', 'Labor',   50.7, 49.3),
    (1954, 'Robert Menzies',   'Herbert Evatt',   'Liberal', 'Labor',   50.7, 49.3),
    (1955, 'Robert Menzies',   'Herbert Evatt',   'Liberal', 'Labor',   54.4, 45.6),
    (1958, 'Robert Menzies',   'Herbert Evatt',   'Liberal', 'Labor',   54.1, 45.9),
    (1961, 'Robert Menzies',   'Arthur Calwell',  'Liberal', 'Labor',   50.5, 49.5),
    (1963, 'Robert Menzies',   'Arthur Calwell',  'Liberal', 'Labor',   52.7, 47.3),
    (1966, 'Harold Holt',      'Arthur Calwell',  'Liberal', 'Labor',   57.0, 43.0),
    (1969, 'John Gorton',      'Gough Whitlam',   'Liberal', 'Labor',   50.2, 49.8),
    (1972, 'Gough Whitlam',    'William McMahon', 'Labor',   'Liberal', 52.7, 47.3),
    (1974, 'Gough Whitlam',    'Billy Snedden',   'Labor',   'Liberal', 51.7, 48.3),
    (1975, 'Malcolm Fraser',   'Gough Whitlam',   'Liberal', 'Labor',   55.7, 44.3),
    (1977, 'Malcolm Fraser',   'Gough Whitlam',   'Liberal', 'Labor',   54.4, 45.6),
    (1980, 'Malcolm Fraser',   'Bill Hayden',     'Liberal', 'Labor',   50.4, 49.6),
    (1983, 'Bob Hawke',        'Malcolm Fraser',  'Labor',   'Liberal', 53.2, 46.8),
    (1984, 'Bob Hawke',        'Andrew Peacock',  'Labor',   'Liberal', 51.8, 48.2),
    (1987, 'Bob Hawke',        'John Howard',     'Labor',   'Liberal', 50.8, 49.2),
    (1990, 'Bob Hawke',        'Andrew Peacock',  'Labor',   'Liberal', 49.9, 50.1),  # *Hawke won seats despite lower 2PP
    (1993, 'Paul Keating',     'John Hewson',     'Labor',   'Liberal', 51.4, 48.6),
    (1996, 'John Howard',      'Paul Keating',    'Liberal', 'Labor',   53.6, 46.4),
    (1998, 'John Howard',      'Kim Beazley',     'Liberal', 'Labor',   49.0, 51.0),  # *vote inversion
    (2001, 'John Howard',      'Kim Beazley',     'Liberal', 'Labor',   51.0, 49.0),
    (2004, 'John Howard',      'Mark Latham',     'Liberal', 'Labor',   52.7, 47.3),
    (2007, 'Kevin Rudd',       'John Howard',     'Labor',   'Liberal', 52.7, 47.3),
    (2010, 'Julia Gillard',    'Tony Abbott',     'Labor',   'Liberal', 50.1, 49.9),
    (2013, 'Tony Abbott',      'Kevin Rudd',      'Liberal', 'Labor',   53.5, 46.5),
    (2016, 'Malcolm Turnbull', 'Bill Shorten',    'Liberal', 'Labor',   50.4, 49.6),
    (2019, 'Scott Morrison',   'Bill Shorten',    'Liberal', 'Labor',   51.5, 48.5),
]


def ngrams_pair(term1, term2, year):
    year_start = max(1960, year - 1)
    year_end   = min(2019, year)
    if year > 2019:
        return None, None

    params = {
        'content': f'{term1},{term2}',
        'year_start': year_start,
        'year_end':   year_end,
        'corpus':     'en-2019',
        'smoothing':  0,
    }
    url = NGRAMS_URL + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            break
        except Exception as e:
            time.sleep(5 * (attempt + 1))
    else:
        return None, None

    freqs = {s['ngram']: sum(s['timeseries']) for s in data}
    return freqs.get(term1, 0.0), freqs.get(term2, 0.0)


def collect(elections, country, label):
    rows = []
    for i, (yr, wn, ln, wp, lp, wv, lv) in enumerate(elections):
        print(f'  [{i+1}/{len(elections)}] {yr} {country}: {wn} vs {ln}')
        f_w, f_l = ngrams_pair(wn, ln, yr)
        time.sleep(1.2)

        if f_w is None:
            status, w_share = 'skip_year', None
        elif f_w == 0 and f_l == 0:
            status, w_share = 'both_zero', None
        elif f_w == 0:
            status, w_share = 'winner_zero', 0.0
        elif f_l == 0:
            status, w_share = 'loser_zero', 1.0
        else:
            status, w_share = 'ok', f_w / (f_w + f_l)

        attn_won = int(w_share > 0.5) if w_share is not None else None
        rows.append(dict(
            year=yr, country=country, source='Google Ngrams',
            winner_name=wn, loser_name=ln,
            winner_party=wp, loser_party=lp,
            winner_vote_pct=wv, loser_vote_pct=lv,
            winner_mention_share=w_share,
            loser_mention_share=1.0 - w_share if w_share is not None else None,
            winner_ngrams_freq=f_w, loser_ngrams_freq=f_l,
            attention_leader_won=attn_won, status=status,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f'{label}_all_races.csv', index=False)
    df_ok = df[df['status'] == 'ok'].copy()
    df_ok.to_csv(OUT_DIR / f'{label}_mention_shares.csv', index=False)

    print(f'\n=== {country.upper()} RESULTS ===')
    print(f'Total: {len(df)} elections, with usable data: {len(df_ok)}')
    if len(df_ok) >= 3:
        k, n = int(df_ok['attention_leader_won'].sum()), len(df_ok)
        res = stats.binomtest(k, n, p=0.5, alternative='two-sided')
        print(f'H1: {k}/{n} ({k/n:.1%}), p={res.pvalue:.4f}')
        r, p = pearsonr(df_ok['winner_mention_share'], df_ok['winner_vote_pct'])
        print(f'H2: r={r:.3f}, p={p:.4f}')
    print(df_ok[['year','winner_name','loser_name','winner_mention_share','winner_vote_pct','attention_leader_won']].to_string(index=False))
    return df_ok


if __name__ == '__main__':
    print('=== UK General Elections ===')
    collect(UK_ELECTIONS, 'UK', 'uk')
    print('\n=== Australian Federal Elections ===')
    collect(AUS_ELECTIONS, 'Australia', 'australia')
    print('\nDone.')
