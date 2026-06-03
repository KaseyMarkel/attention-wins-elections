"""
Compile mention share data from all sources into a single analysis-ready file.

Reads from data/raw/{source}/presidential_mention_shares.csv
Writes to data/processed/mention_shares_by_source.csv

Also writes summary stats and flags data quality issues.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
import json

SOURCE_FILES = {
    'Google Ngrams':       Path('data/raw/ngrams/presidential_mention_shares.csv'),
    'Google Trends':       Path('data/raw/trends/presidential_mention_shares.csv'),
    'Wikipedia Pageviews': Path('data/raw/wikipedia/presidential_mention_shares.csv'),
    'GDELT News':          Path('data/raw/gdelt_news/presidential_mention_shares.csv'),
    'GDELT TV':            Path('data/raw/gdelt_tv/presidential_mention_shares.csv'),
    'Reddit':              Path('data/raw/reddit/presidential_mention_shares.csv'),
    'MediaCloud':          Path('data/raw/mediacloud/presidential_mention_shares.csv'),
}

PROCESSED_DIR = Path('data/processed')
PROCESSED_DIR.mkdir(exist_ok=True)


def main():
    all_dfs = []
    summary = []

    print('=== COMPILING MENTION SHARE DATA FROM ALL SOURCES ===\n')

    for source_name, path in SOURCE_FILES.items():
        if not path.exists():
            print(f'[MISSING] {source_name}: {path}')
            summary.append({'source': source_name, 'status': 'missing', 'n': 0})
            continue

        df = pd.read_csv(path)
        if df.empty:
            print(f'[EMPTY]   {source_name}: {path}')
            summary.append({'source': source_name, 'status': 'empty', 'n': 0})
            continue

        required_cols = ['year', 'winner_mention_share', 'winner_vote_pct', 'attention_leader_won']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f'[ERROR]   {source_name}: missing columns {missing_cols}')
            continue

        n = len(df)
        k = df['attention_leader_won'].sum()
        win_rate = k / n

        try:
            binom = stats.binomtest(k, n, p=0.5, alternative='two-sided')
            p_val = binom.pvalue
            ci_lo = binom.proportion_ci().low
            ci_hi = binom.proportion_ci().high
        except Exception:
            p_val, ci_lo, ci_hi = float('nan'), float('nan'), float('nan')

        try:
            r, r_p = pearsonr(df['winner_mention_share'], df['winner_vote_pct'])
        except Exception:
            r, r_p = float('nan'), float('nan')

        print(f'[OK]      {source_name}: n={n} elections')
        print(f'          H1 win rate: {win_rate:.1%} ({k}/{n}), p={p_val:.4f}, CI=[{ci_lo:.2f},{ci_hi:.2f}]')
        print(f'          H2 Pearson r: {r:.3f}, p={r_p:.4f}')
        print()

        df['source'] = source_name
        all_dfs.append(df)

        summary.append({
            'source': source_name,
            'status': 'ok',
            'n': n,
            'h1_win_rate': round(win_rate, 4),
            'h1_k': int(k),
            'h1_p': round(p_val, 5),
            'h1_ci_lo': round(ci_lo, 4),
            'h1_ci_hi': round(ci_hi, 4),
            'h2_r': round(r, 4),
            'h2_p': round(r_p, 5),
        })

    if not all_dfs:
        print('No data found! Run collection scripts first.')
        return

    # Merge all sources
    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = PROCESSED_DIR / 'mention_shares_by_source.csv'
    combined.to_csv(out_path, index=False)
    print(f'Saved combined data: {out_path} ({len(combined)} rows)')

    # Summary table
    summary_df = pd.DataFrame(summary)
    summary_path = PROCESSED_DIR / 'collection_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'Saved summary: {summary_path}')

    # Also save summary as JSON for easy reading
    json_path = PROCESSED_DIR / 'collection_summary.json'
    json_path.write_text(json.dumps(summary, indent=2))

    # Print summary table
    print('\n=== SUMMARY TABLE ===')
    ok = summary_df[summary_df['status'] == 'ok']
    if not ok.empty:
        print(ok[['source', 'n', 'h1_win_rate', 'h1_p', 'h2_r', 'h2_p']].to_string(index=False))

    # Cross-source consistency check
    print('\n=== CROSS-SOURCE CONSISTENCY ===')
    for year in sorted(combined['year'].unique()):
        year_data = combined[combined['year'] == year]
        if len(year_data) < 2:
            continue
        shares = year_data['winner_mention_share']
        if shares.max() - shares.min() > 0.15:
            print(f'  FLAGGED {year}: winner share ranges {shares.min():.1%}–{shares.max():.1%} across sources')
            for _, row in year_data.iterrows():
                print(f'    {row["source"]}: {row["winner_mention_share"]:.1%}')


if __name__ == '__main__':
    main()
