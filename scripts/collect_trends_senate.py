"""
Google Trends — Senate races (state-level geo comparison).

For each D-vs-R Senate race (2016–2020), queries both candidate names together
with geo restricted to the race's state for a 52-week pre-election window.

Resumes from checkpoint if present; skips already-collected races.

Output:
  data/raw/trends/senate_mention_shares.csv      — final output
  data/raw/trends/senate_checkpoint.csv          — running checkpoint
"""

import time
import sys
from pathlib import Path
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT_DIR = Path('data/raw/trends')
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUT_DIR / 'senate_checkpoint.csv'
FINAL_PATH      = OUT_DIR / 'senate_mention_shares.csv'


def collect_one(winner: str, loser: str, state_po: str, year: int) -> dict | None:
    from pytrends.request import TrendReq
    import datetime

    # Election day approximations
    EDAYS = {2016: '2016-11-08', 2018: '2018-11-06', 2020: '2020-11-03'}
    ed_str = EDAYS.get(year)
    if not ed_str:
        return None

    ed = datetime.date.fromisoformat(ed_str)
    start = (ed - datetime.timedelta(days=365)).isoformat()
    timeframe = f'{start} {ed_str}'
    geo = f'US-{state_po}'

    pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25), retries=2, backoff_factor=1.0)

    for attempt in range(3):
        try:
            pytrends.build_payload([winner, loser], timeframe=timeframe, geo=geo, gprop='')
            df = pytrends.interest_over_time()
            break
        except Exception as e:
            print(f'    attempt {attempt+1} failed: {e}')
            time.sleep(20 * (attempt + 1))
    else:
        return None

    if df.empty:
        return None

    df = df.drop(columns=['isPartial'], errors='ignore')

    w_sum = df[winner].sum() if winner in df.columns else 0
    l_sum = df[loser].sum()  if loser  in df.columns else 0
    total = w_sum + l_sum

    if total == 0:
        return {'w_sum': 0, 'l_sum': 0, 'w_share': None, 'timeframe': timeframe}

    return {
        'w_sum': int(w_sum), 'l_sum': int(l_sum),
        'w_share': w_sum / total,
        'timeframe': timeframe,
    }


def main():
    senate = pd.read_csv('data/raw/senate_cleaned.csv')
    senate = senate[senate['year'].isin([2016, 2018, 2020])].copy()
    print(f'{len(senate)} Senate races (2016–2020)')

    # Load existing checkpoint (the old one from the prior run)
    old_ckpt_path = OUT_DIR / 'senate_mention_shares_checkpoint.csv'
    done_keys = set()
    existing_rows = []

    if CHECKPOINT_PATH.exists():
        ckpt = pd.read_csv(CHECKPOINT_PATH)
        existing_rows = ckpt.to_dict('records')
        done_keys = {(r['year'], r['state']) for r in existing_rows}
        print(f'Resuming: {len(done_keys)} races already done')
    elif old_ckpt_path.exists():
        ckpt = pd.read_csv(old_ckpt_path)
        existing_rows = ckpt.to_dict('records')
        done_keys = {(r['year'], r['state']) for r in existing_rows}
        print(f'Loaded old checkpoint: {len(done_keys)} races already done')

    all_rows = list(existing_rows)
    remaining = senate[~senate.apply(lambda r: (r['year'], r['state_po']) in done_keys, axis=1)]
    print(f'{len(remaining)} races to collect')

    for i, (_, row) in enumerate(remaining.iterrows()):
        yr = int(row['year'])
        state = row['state_po']
        winner = row['winner']
        loser  = row['loser']

        print(f'[{i+1}/{len(remaining)}] {yr} {state}: {winner} vs {loser}')

        result = collect_one(winner, loser, state, yr)
        time.sleep(8)

        if result is None or result['w_share'] is None:
            status = 'skip'
            w_share = None
            attn_won = None
        else:
            status = 'ok'
            w_share = result['w_share']
            attn_won = int(w_share > 0.5)

        all_rows.append({
            'year': yr, 'state': state, 'source': 'Google Trends',
            'winner': winner, 'loser': loser,
            'winner_party': row['winner_party'],
            'winner_mention_share': round(w_share, 6) if w_share is not None else None,
            'loser_mention_share': round(1 - w_share, 6) if w_share is not None else None,
            'winner_vote_pct': row['winner_pct'],
            'loser_vote_pct': row['loser_pct'],
            'attention_leader_won': attn_won,
            'winner_trends_sum': result['w_sum'] if result else None,
            'loser_trends_sum':  result['l_sum'] if result else None,
            'geo': f'US-{state}',
            'timeframe': result['timeframe'] if result else None,
            'status': status,
        })

        # Checkpoint every 10 races
        if (i + 1) % 10 == 0:
            pd.DataFrame(all_rows).to_csv(CHECKPOINT_PATH, index=False)
            print(f'  Checkpoint saved ({len(all_rows)} total)')

    df = pd.DataFrame(all_rows)
    df.to_csv(CHECKPOINT_PATH, index=False)

    df_ok = df[df['status'] == 'ok'].copy()
    df_ok.to_csv(FINAL_PATH, index=False)
    print(f'\nSaved {len(df_ok)} rows with data to {FINAL_PATH}')

    if len(df_ok) >= 5:
        k = df_ok['attention_leader_won'].sum()
        n = len(df_ok)
        result = stats.binomtest(int(k), n, p=0.5, alternative='two-sided')
        print(f'H1: {k}/{n} ({k/n:.1%}), p={result.pvalue:.4f}')
        r, p = pearsonr(df_ok['winner_mention_share'], df_ok['winner_vote_pct'])
        print(f'H2: r={r:.3f}, p={p:.4f}')


if __name__ == '__main__':
    main()
