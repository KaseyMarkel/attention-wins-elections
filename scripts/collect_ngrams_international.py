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

# ── Canadian Federal Elections ─────────────────────────────────────────────────
# winner = whoever became PM (largest party); the two figures are always the
# Liberal vs Progressive-Conservative/Conservative leaders, the only two parties
# that form government. vote_pct = that party's national popular vote share (%).
# The 1993/1997/2000 elections are EXCLUDED: the PC party collapsed and the main
# opposition was Bloc Québécois / Reform / Canadian Alliance, so there is no clean
# Liberal-vs-Conservative two-leader contest (documented exclusion). 2011 excluded
# (NDP, not the Liberals, were the Official Opposition).
# NOTE: vote percentages are from standard Elections Canada / Wikipedia tallies —
# VERIFY before publishing. Winners/leaders confirmed via Wikipedia.
CANADA_ELECTIONS = [
    (1945, 'Mackenzie King',  'John Bracken',     'Liberal',      'PC',           39.8, 27.6),
    (1949, 'Louis St. Laurent','George Drew',     'Liberal',      'PC',           49.2, 29.7),
    (1953, 'Louis St. Laurent','George Drew',     'Liberal',      'PC',           48.8, 31.0),
    (1957, 'John Diefenbaker','Louis St. Laurent','PC',           'Liberal',      38.9, 40.9),  # *PC govt, lower vote
    (1958, 'John Diefenbaker','Lester Pearson',   'PC',           'Liberal',      53.7, 33.5),
    (1962, 'John Diefenbaker','Lester Pearson',   'PC',           'Liberal',      37.2, 37.2),
    (1963, 'Lester Pearson',  'John Diefenbaker', 'Liberal',      'PC',           41.5, 32.7),
    (1965, 'Lester Pearson',  'John Diefenbaker', 'Liberal',      'PC',           40.2, 32.4),
    (1968, 'Pierre Trudeau',  'Robert Stanfield', 'Liberal',      'PC',           45.4, 31.4),
    (1972, 'Pierre Trudeau',  'Robert Stanfield', 'Liberal',      'PC',           38.4, 35.0),
    (1974, 'Pierre Trudeau',  'Robert Stanfield', 'Liberal',      'PC',           43.2, 35.4),
    (1979, 'Joe Clark',       'Pierre Trudeau',   'PC',           'Liberal',      35.9, 40.1),  # *PC govt, lower vote
    (1980, 'Pierre Trudeau',  'Joe Clark',        'Liberal',      'PC',           44.3, 32.5),
    (1984, 'Brian Mulroney',  'John Turner',      'PC',           'Liberal',      50.0, 28.0),
    (1988, 'Brian Mulroney',  'John Turner',      'PC',           'Liberal',      43.0, 31.9),
    (2004, 'Paul Martin',     'Stephen Harper',   'Liberal',      'Conservative', 36.7, 29.6),
    (2006, 'Stephen Harper',  'Paul Martin',      'Conservative', 'Liberal',      36.3, 30.2),
    (2008, 'Stephen Harper',  'Stephane Dion',    'Conservative', 'Liberal',      37.7, 26.3),
    (2015, 'Justin Trudeau',  'Stephen Harper',   'Liberal',      'Conservative', 39.5, 31.9),
    (2019, 'Justin Trudeau',  'Andrew Scheer',    'Liberal',      'Conservative', 33.1, 34.3),  # *Lib govt, lower vote
]

# ── New Zealand General Elections ──────────────────────────────────────────────
# winner = PM after the election; the two figures are the Labour vs National
# leaders. vote_pct = party vote share (%) (party vote from 1996 under MMP).
# NOTE: vote percentages from standard tallies — VERIFY before publishing.
# Many recent NZ leaders are sparse in the (US/UK-dominated) English book corpus,
# so expect heavy attrition to the usable subset.
NZ_ELECTIONS = [
    (1946, 'Peter Fraser',    'Sidney Holland',  'Labour',   'National', 51.3, 48.4),
    (1949, 'Sidney Holland',  'Peter Fraser',    'National', 'Labour',   51.9, 47.2),
    (1951, 'Sidney Holland',  'Walter Nash',     'National', 'Labour',   54.0, 45.8),
    (1954, 'Sidney Holland',  'Walter Nash',     'National', 'Labour',   44.3, 44.1),
    (1957, 'Walter Nash',     'Keith Holyoake',  'Labour',   'National', 48.3, 44.2),
    (1960, 'Keith Holyoake',  'Walter Nash',     'National', 'Labour',   47.6, 43.4),
    (1963, 'Keith Holyoake',  'Arnold Nordmeyer','National', 'Labour',   47.1, 43.7),
    (1966, 'Keith Holyoake',  'Norman Kirk',     'National', 'Labour',   43.6, 41.4),
    (1969, 'Keith Holyoake',  'Norman Kirk',     'National', 'Labour',   45.2, 44.2),
    (1972, 'Norman Kirk',     'Jack Marshall',   'Labour',   'National', 48.4, 41.5),
    (1975, 'Robert Muldoon',  'Bill Rowling',    'National', 'Labour',   47.6, 39.6),
    (1978, 'Robert Muldoon',  'Bill Rowling',    'National', 'Labour',   39.8, 40.4),  # *Nat govt, lower vote
    (1981, 'Robert Muldoon',  'Bill Rowling',    'National', 'Labour',   38.8, 39.0),  # *Nat govt, lower vote
    (1984, 'David Lange',     'Robert Muldoon',  'Labour',   'National', 43.0, 35.9),
    (1987, 'David Lange',     'Jim Bolger',      'Labour',   'National', 48.0, 44.0),
    (1990, 'Jim Bolger',      'Mike Moore',      'National', 'Labour',   47.8, 35.1),
    (1993, 'Jim Bolger',      'Mike Moore',      'National', 'Labour',   35.1, 34.7),
    (1999, 'Helen Clark',     'Jenny Shipley',   'Labour',   'National', 38.7, 30.5),
    (2002, 'Helen Clark',     'Bill English',    'Labour',   'National', 41.3, 20.9),
    (2005, 'Helen Clark',     'Don Brash',       'Labour',   'National', 41.1, 39.1),
    (2008, 'John Key',        'Helen Clark',     'National', 'Labour',   44.9, 34.0),
    (2011, 'John Key',        'Phil Goff',       'National', 'Labour',   47.3, 27.5),
    (2014, 'John Key',        'David Cunliffe',  'National', 'Labour',   47.0, 25.1),
]


def ngrams_pair(term1, term2, year):
    if year > 2019:
        return None, None
    # 2-year window ending on the election year. NEVER floor year_start at 1960:
    # pre-1960 elections would make year_start > year_end (an invalid range), and
    # the Ngrams API then returns an unrelated decade-long series — silently
    # mis-measuring every pre-1960 election. The en-2019 corpus covers back to
    # the 1500s, so year-1 is always valid.
    year_start = year - 1
    year_end   = min(2019, year)
    expected   = year_end - year_start + 1

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

    # Guard against the API returning a mismatched-length series (the quirk that
    # caused the pre-1960 bug): only sum series of the expected window length.
    freqs = {}
    for s in data:
        ts = s['timeseries']
        if len(ts) != expected:
            raise ValueError(
                f'Ngrams returned {len(ts)} pts for {s["ngram"]!r} '
                f'(window {year_start}-{year_end} expects {expected}).')
        freqs[s['ngram']] = sum(ts)
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
    print('\n=== Canadian Federal Elections ===')
    collect(CANADA_ELECTIONS, 'Canada', 'canada')
    print('\n=== New Zealand General Elections ===')
    collect(NZ_ELECTIONS, 'New Zealand', 'new_zealand')
    print('\nDone.')
