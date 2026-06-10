"""
UK House of Commons party-aggregate dataset (analogue of the US House figure).

Mirrors the US House analysis: instead of individual constituency races (whose
candidates are far too obscure for the Ngrams book corpus — the same reason the
US House is analysed at party level), we ask whether the major party whose
LEADER dominates national book coverage wins the larger share of Commons seats.

  Conservative attention share = Con-leader mentions / (Con + Lab leader mentions)
  Conservative seat share       = Con seats / (Con + Lab seats)   [two-party]

Leader mention shares are reused from data/raw/ngrams/uk_mention_shares.csv
(already collected). Seat counts are the official Commons results for each
general election (House of Commons Library / Craig, British Electoral Facts).

Output: data/raw/ngrams/uk_commons.csv
"""

from pathlib import Path
import pandas as pd
from scipy.stats import binomtest, pearsonr

OUT_DIR = Path('data/raw/ngrams')

# Official Conservative / Labour seats won at each general election.
# (Feb 1974 used for 1974; National Liberals counted with the Conservatives.)
COMMONS_SEATS = {
    1945: (197, 393), 1950: (298, 315), 1951: (321, 295), 1955: (345, 277),
    1959: (365, 258), 1966: (253, 364), 1970: (330, 288), 1974: (297, 301),
    1983: (397, 209), 1987: (376, 229), 1992: (336, 271), 1997: (165, 418),
    2001: (166, 412), 2005: (198, 355), 2015: (330, 232), 2017: (317, 262),
    2019: (365, 202),
}  # {year: (conservative_seats, labour_seats)}


def main():
    uk = pd.read_csv(OUT_DIR / 'uk_mention_shares.csv')

    rows = []
    for _, r in uk.iterrows():
        yr = int(r['year'])
        if yr not in COMMONS_SEATS:
            continue
        # Conservative leader's attention share for this election
        if r['winner_party'] == 'Conservative':
            con_attn = r['winner_mention_share']
        else:
            con_attn = r['loser_mention_share']

        con_seats, lab_seats = COMMONS_SEATS[yr]
        con_seat_share = con_seats / (con_seats + lab_seats)
        majority = 'Con' if con_seats > lab_seats else 'Lab'
        attn_majority = 'Con' if con_attn > 0.5 else 'Lab'

        rows.append({
            'year': yr,
            'source': 'Google Ngrams',
            'con_attention_share': round(float(con_attn), 6),
            'lab_attention_share': round(1 - float(con_attn), 6),
            'con_seats': con_seats, 'lab_seats': lab_seats,
            'con_seat_share': round(con_seat_share, 6),
            'majority': majority,
            'attn_majority': attn_majority,
            'attn_predicts_majority': int(majority == attn_majority),
        })

    df = pd.DataFrame(rows).sort_values('year')
    out = OUT_DIR / 'uk_commons.csv'
    df.to_csv(out, index=False)
    print(f'Saved {len(df)} UK general elections to {out}\n')

    k, n = int(df['attn_predicts_majority'].sum()), len(df)
    p = binomtest(k, n, 0.5, alternative='two-sided').pvalue
    r, pr = pearsonr(df['con_attention_share'], df['con_seat_share'])
    print(f'H1 (Commons): attention-leading party won more seats {k}/{n} ({k/n:.0%}), p={p:.4f}')
    print(f'H2 (Commons): Con attention share vs Con seat share  r={r:.3f}, p={pr:.4f}')
    print(df[['year', 'con_attention_share', 'con_seat_share',
              'majority', 'attn_predicts_majority']].to_string(index=False))


if __name__ == '__main__':
    main()
