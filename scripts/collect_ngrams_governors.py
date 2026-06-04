"""
Google Ngrams — US gubernatorial elections, 1960–2018.

Reads cleaned D-vs-R governor races from data/raw/governors_cleaned.csv
(produced by collect_governors_wikipedia.py) and queries Ngrams for each
winner/loser pair over the 2 years ending on election day.

Output:
  data/raw/ngrams/governors_mention_shares.csv  — status=ok races only
  data/raw/ngrams/governors_all_races.csv       — all races
"""

import json, re, time, urllib.request, urllib.parse
from pathlib import Path
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr

OUT_DIR = Path('data/raw/ngrams')
OUT_DIR.mkdir(parents=True, exist_ok=True)

NGRAMS_URL = 'https://books.google.com/ngrams/json'
HEADERS    = {'User-Agent': 'attention-wins-elections/1.0 (kaseymarkel@semillanueva.org)'}

STATE_PO = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
    'Colorado':'CO','Connecticut':'CT','Delaware':'DE','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS',
    'Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA',
    'Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT',
    'Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM',
    'New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK',
    'Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC',
    'South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT',
    'Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY',
}


def ngram_term(name):
    """Make a queryable Ngrams term from a candidate name.

    'W. P. "Bill" Atkinson' -> 'Bill Atkinson' (nickname + surname);
    'Otto Kerner Jr.' -> 'Otto Kerner' (books rarely print suffixes,
    and the Senate comparator data has no suffixes either).
    """
    s = name.strip()
    s = re.sub(r',?\s+(Jr\.?|Sr\.?|II|III|IV)$', '', s)
    nick = re.search(r'"([^"]+)"', s)
    if nick:
        surname = s.split()[-1]
        return f'{nick.group(1)} {surname}'
    return s


def ngrams_pair(term1, term2, year):
    year_start = max(1960, year - 1)
    year_end   = min(2019, year)
    if year > 2019:
        return None, None
    params = {'content': f'{term1},{term2}', 'year_start': year_start,
              'year_end': year_end, 'corpus': 'en-2019', 'smoothing': 0}
    url = NGRAMS_URL + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            break
        except Exception:
            time.sleep(5 * (attempt + 1))
    else:
        return None, None
    freqs = {s['ngram']: sum(s['timeseries']) for s in data}
    return freqs.get(term1, 0.0), freqs.get(term2, 0.0)


def main():
    gov = pd.read_csv('data/raw/governors_cleaned.csv')
    gov = gov[(gov['year'] >= 1960) & (gov['year'] <= 2018)].copy()
    print(f'{len(gov)} D-vs-R governor races, 1960–2018')

    ckpt_path = OUT_DIR / 'governors_checkpoint.csv'
    done, rows = set(), []
    if ckpt_path.exists():
        ckpt = pd.read_csv(ckpt_path)
        rows = ckpt.to_dict('records')
        done = {(r['year'], r['state']) for r in rows}
        print(f'Resuming: {len(done)} already done')

    gov['state_po'] = gov['state'].map(STATE_PO)
    remaining = gov[~gov.apply(lambda r: (r['year'], r['state_po']) in done, axis=1)]
    print(f'{len(remaining)} races to query')

    for i, (_, row) in enumerate(remaining.iterrows()):
        yr, state = int(row['year']), row['state_po']
        wn, ln = row['winner'], row['loser']
        w_term, l_term = ngram_term(wn), ngram_term(ln)

        if i % 25 == 0:
            print(f'  [{i+1}/{len(remaining)}] {yr} {state}: {wn} vs {ln}', flush=True)

        f_w, f_l = ngrams_pair(w_term, l_term, yr)
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

        rows.append(dict(
            year=yr, state=state, source='Google Ngrams',
            winner=wn, loser=ln,
            winner_party=row['winner_party'],
            winner_vote_pct=row['winner_pct'], loser_vote_pct=row['loser_pct'],
            winner_mention_share=w_share,
            loser_mention_share=1.0 - w_share if w_share is not None else None,
            winner_ngrams_freq=f_w, loser_ngrams_freq=f_l,
            attention_leader_won=int(w_share > 0.5) if w_share is not None else None,
            status=status,
        ))

        if (i + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(ckpt_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'governors_all_races.csv', index=False)
    df_ok = df[df['status'] == 'ok'].copy()
    df_ok.to_csv(OUT_DIR / 'governors_mention_shares.csv', index=False)

    print(f'\n=== GOVERNOR RESULTS ===')
    print(f'Total: {len(df)}, usable: {len(df_ok)}')
    print(df['status'].value_counts().to_string())
    if len(df_ok) >= 5:
        k, n = int(df_ok['attention_leader_won'].sum()), len(df_ok)
        res = stats.binomtest(k, n, p=0.5, alternative='two-sided')
        print(f'H1: {k}/{n} ({k/n:.1%}), p={res.pvalue:.4f}')
        r, p = pearsonr(df_ok['winner_mention_share'], df_ok['winner_vote_pct'])
        print(f'H2: r={r:.3f}, p={p:.4f}')


if __name__ == '__main__':
    main()
