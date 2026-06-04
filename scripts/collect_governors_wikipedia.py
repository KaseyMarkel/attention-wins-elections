"""
US gubernatorial election returns, 1960–2018, from Wikipedia infoboxes.

The MIT MEDSL Dataverse has no candidate-level governor dataset (the file IDs
previously tried returned Senate data). Wikipedia race pages ("1962 California
gubernatorial election") have consistent infoboxes with nominees, parties, and
vote percentages — and properly-cased names, which Ngrams needs.

For each year: list members of "Category:{year} United States gubernatorial
elections", fetch wikitext in batches, parse the infobox, keep races where the
top two candidates are one Democrat and one Republican.

Output:
  data/raw/governors_cleaned.csv — year, state, winner, loser, winner_party,
                                   loser_party, winner_pct, loser_pct
                                   (vote pcts as fractions, matching senate_cleaned.csv)
"""

import json, re, time, urllib.request, urllib.parse
from pathlib import Path
import pandas as pd

API = 'https://en.wikipedia.org/w/api.php'
HEADERS = {'User-Agent': 'attention-wins-elections/1.0 (kaseymarkel@semillanueva.org)'}
TITLE_RE = re.compile(r'^(\d{4}) ([A-Z][A-Za-z ]+?) gubernatorial (?:special )?election$')

TERRITORIES = {'Puerto Rico', 'Puerto Rican', 'Guam', 'Guamanian',
               'United States Virgin Islands', 'American Samoa', 'American Samoan',
               'Northern Mariana Islands', 'District of Columbia'}


def api_get(params):
    params = dict(params, format='json', formatversion=2)
    url = API + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f'API failed: {params}')


def category_members(year):
    """All race pages in the year's gubernatorial elections category."""
    titles, cont = [], {}
    while True:
        d = api_get({'action': 'query', 'list': 'categorymembers',
                     'cmtitle': f'Category:{year} United States gubernatorial elections',
                     'cmlimit': 500, **cont})
        for m in d['query']['categorymembers']:
            t = m['title']
            if TITLE_RE.match(t) and 'lieutenant' not in t.lower():
                titles.append(t)
        if 'continue' in d:
            cont = {'cmcontinue': d['continue']['cmcontinue']}
        else:
            return titles


def fetch_wikitexts(titles):
    """Batch-fetch page wikitext. Returns {title: text}."""
    out = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        d = api_get({'action': 'query', 'prop': 'revisions', 'rvprop': 'content',
                     'rvslots': 'main', 'titles': '|'.join(batch), 'redirects': 1})
        for page in d['query']['pages']:
            revs = page.get('revisions')
            if revs:
                out[page['title']] = revs[0]['slots']['main']['content']
        time.sleep(0.5)
    return out


def clean_name(raw):
    """'[[Bob Graham (politician)|Bob Graham]]' -> 'Bob Graham'."""
    s = raw.strip()
    s = re.sub(r'<ref[^>]*/\s*>', '', s)
    s = re.sub(r'<ref[^>]*>.*?</ref>', '', s, flags=re.S)
    s = re.sub(r'\{\{\s*nowrap\s*\|([^{}]*)\}\}', r'\1', s, flags=re.I)  # unwrap {{nowrap|X}}
    s = re.sub(r'\{\{[^{}]*\}\}', '', s)              # other templates (efn, small, etc.)
    s = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', s)  # wikilinks -> display text
    s = s.replace("'''", '').replace("''", '')
    s = re.sub(r'<[^>]+>.*', '', s)                   # drop <br> and anything after
    s = re.sub(r'\s*\([^)]*\)\s*$', '', s)            # trailing disambiguator
    return s.strip()


def party_letter(raw):
    """Substring match handles state parties ('Florida Democratic Party',
    'Minnesota Democratic–Farmer–Labor Party', 'Democratic-NPL')."""
    p = re.sub(r'[–—]', '-', raw).lower()
    has_d = 'democratic' in p or 'democrat' in p
    has_r = 'republican' in p
    if has_d and not has_r:
        return 'D'
    if has_r and not has_d:
        return 'R'
    return None


def parse_infobox(text):
    """Extract (name, party_letter, pct_fraction) for each infobox candidate."""
    cands = []
    for n in range(1, 6):
        name_m = re.search(rf'\|\s*(?:nominee|candidate){n}\s*=\s*(.+)', text)
        party_m = re.search(rf'\|\s*party{n}\s*=\s*(.+)', text)
        pct_m = re.search(rf'\|\s*percentage{n}\s*=\s*(.+)', text)
        if not (name_m and party_m and pct_m):
            continue
        name = clean_name(name_m.group(1))
        raw_pct = pct_m.group(1)
        # {{percent|votes|total|...}} template
        tmpl = re.search(r'\{\{\s*percent(?:age)?\s*\|([\d,\.]+)\|([\d,\.]+)', raw_pct, re.I)
        if tmpl:
            pct = float(tmpl.group(1).replace(',', '')) / float(tmpl.group(2).replace(',', ''))
        else:
            pct_s = clean_name(raw_pct).replace('%', '').replace(',', '').strip()
            try:
                pct = float(pct_s) / 100.0
            except ValueError:
                continue
        if not name or name.lower() in ('none', 'n/a', 'tbd'):
            continue
        cands.append((name, party_letter(party_m.group(1)), pct))
    return cands


def main():
    rows, skipped = [], []
    for year in range(1960, 2019):
        titles = category_members(year)
        if not titles:
            continue
        texts = fetch_wikitexts(titles)
        print(f'{year}: {len(titles)} races')
        for title, text in texts.items():
            m = TITLE_RE.match(title)
            if not m:
                continue
            state = m.group(2)
            if state in TERRITORIES:
                continue
            cands = parse_infobox(text)
            if len(cands) < 2:
                skipped.append((year, state, 'no_infobox_parse'))
                continue
            cands.sort(key=lambda c: c[2], reverse=True)
            top2 = cands[:2]
            parties = {c[1] for c in top2}
            if parties != {'D', 'R'}:
                skipped.append((year, state, f'top2_not_DvR:{[c[1] for c in top2]}'))
                continue
            w, l = top2
            rows.append(dict(year=year, state=state,
                             winner=w[0], loser=l[0],
                             winner_party=w[1], loser_party=l[1],
                             winner_pct=w[2], loser_pct=l[2]))
        time.sleep(0.3)

    df = pd.DataFrame(rows).sort_values(['year', 'state']).reset_index(drop=True)
    out = Path('data/raw/governors_cleaned.csv')
    df.to_csv(out, index=False)
    print(f'\nSaved {len(df)} D-vs-R governor races to {out}')
    print(f'Skipped {len(skipped)}:')
    for s in skipped:
        print('  ', s)

    # sanity spot checks
    for yr, st, expect in [(1962, 'California', 'Pat Brown'),
                           (1966, 'California', 'Ronald Reagan'),
                           (1978, 'Texas', 'Bill Clements'),
                           (1994, 'Texas', 'George W. Bush')]:
        hit = df[(df.year == yr) & (df.state == st)]
        got = hit.iloc[0]['winner'] if len(hit) else 'MISSING'
        print(f'  check {yr} {st}: {got} (expect {expect})')


if __name__ == '__main__':
    main()
