"""
Canonical presidential candidate reference.
Used by ALL data collection scripts — edit here, changes propagate everywhere.

Each tuple: (year, winner_name, loser_name, winner_party, winner_vote_pct, loser_vote_pct, incumbent_party)
Note 2000: Gore won popular vote; Bush won EC. We use POPULAR vote as outcome per pre-registration.
Note 1992: Three-candidate race; Perot excluded from mention-share denominator (major-party only).
          Perot's vote share (18.9%) logged separately in edge_cases.md.
"""

PRESIDENTIAL = [
    # year  winner                   loser                    wp   lp    wv      lv     inc_pty
    (1960, 'John Kennedy',           'Richard Nixon',         'D', 'R', 49.72,  49.55,  'R'),
    (1964, 'Lyndon Johnson',         'Barry Goldwater',       'D', 'R', 61.05,  38.47,  'D'),
    (1968, 'Richard Nixon',          'Hubert Humphrey',       'R', 'D', 43.42,  42.72,  'D'),
    (1972, 'Richard Nixon',          'George McGovern',       'R', 'D', 60.67,  37.52,  'R'),
    (1976, 'Jimmy Carter',           'Gerald Ford',           'D', 'R', 50.08,  48.02,  'R'),
    (1980, 'Ronald Reagan',          'Jimmy Carter',          'R', 'D', 50.75,  41.01,  'D'),
    (1984, 'Ronald Reagan',          'Walter Mondale',        'R', 'D', 58.77,  40.56,  'R'),
    (1988, 'George Bush',            'Michael Dukakis',       'R', 'D', 53.37,  45.65,  'R'),
    (1992, 'Bill Clinton',           'George Bush',           'D', 'R', 43.01,  37.45,  'R'),  # Perot 18.9%
    (1996, 'Bill Clinton',           'Bob Dole',              'D', 'R', 49.24,  40.71,  'D'),  # Perot 8.4%
    (2000, 'Al Gore',                'George W. Bush',        'D', 'R', 48.38,  47.87,  'D'),  # Gore won pop. vote
    (2004, 'George W. Bush',         'John Kerry',            'R', 'D', 50.73,  48.27,  'R'),
    (2008, 'Barack Obama',           'John McCain',           'D', 'R', 52.93,  45.65,  'R'),
    (2012, 'Barack Obama',           'Mitt Romney',           'D', 'R', 51.06,  47.20,  'D'),
    (2016, 'Hillary Clinton',        'Donald Trump',          'D', 'R', 48.18,  46.09,  'D'),  # Clinton won pop. vote
    (2020, 'Joe Biden',              'Donald Trump',          'D', 'R', 51.31,  46.86,  'R'),
    (2024, 'Donald Trump',           'Kamala Harris',         'R', 'D', 49.84,  48.36,  'D'),
]

# Search terms for each candidate — handles disambiguation
# Format: (year, winner_search_term, loser_search_term)
# Note: some candidates appear in multiple elections; queries should be year-specific
# where needed (the 12-month window handles most disambiguation naturally)
SEARCH_TERMS = {
    # For Ngrams and text search — use full name for disambiguation
    'ngrams': {
        yr: (w, l) for yr, w, l, *_ in [
            (1960, 'John Kennedy',           'Richard Nixon'),
            (1964, 'Lyndon Johnson',         'Barry Goldwater'),
            (1968, 'Richard Nixon',          'Hubert Humphrey'),
            (1972, 'Richard Nixon',          'George McGovern'),
            (1976, 'Jimmy Carter',           'Gerald Ford'),
            (1980, 'Ronald Reagan',          'Jimmy Carter'),
            (1984, 'Ronald Reagan',          'Walter Mondale'),
            (1988, 'George Bush',            'Michael Dukakis'),
            (1992, 'Bill Clinton',           'George Bush'),
            (1996, 'Bill Clinton',           'Bob Dole'),
            (2000, 'Al Gore',                'George W. Bush'),
            (2004, 'George W. Bush',         'John Kerry'),
            (2008, 'Barack Obama',           'John McCain'),
            (2012, 'Barack Obama',           'Mitt Romney'),
            (2016, 'Hillary Clinton',        'Donald Trump'),
            (2020, 'Joe Biden',              'Donald Trump'),
            (2024, 'Donald Trump',           'Kamala Harris'),
        ]
    },
}

# Official popular vote results for validation
# Source: Dave Leip's Atlas of U.S. Presidential Elections
OFFICIAL_VOTE_SHARES = {
    yr: {'winner': wv, 'loser': lv, 'winner_name': wn, 'loser_name': ln, 'winner_party': wp}
    for yr, wn, ln, wp, lp, wv, lv, _ in PRESIDENTIAL
}
