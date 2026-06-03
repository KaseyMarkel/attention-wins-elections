"""
Real-data figures for 'Is Attention All You Need (to Win Elections)?'

Real data used everywhere we have a complete dataset.
Figures without complete real data show a blank "pending" placeholder —
no synthetic data is mixed into any figure.

Figure inventory:
  fig1  — H1 presidential win rate          [REAL: Ngrams presidential]
  fig2  — H2 scatter pres + senate          [REAL: Ngrams pres + senate]
  fig3  — Attention gap vs margin           [REAL: Ngrams presidential]
  fig4  — Cross-level win rate comparison   [REAL: Ngrams pres + senate + house]
  fig5  — House party attention vs seats    [REAL: Ngrams house]
  fig6  — Gap distribution box              [REAL: Ngrams pres + senate]
  fig7  — Multi-source forest plot          [REAL: 5 sources, 2 pending]
  figS1 — By media era                      [REAL: Ngrams pres, n limited in Social Media era]
  figS2 — Window sensitivity                [PENDING — requires multi-window collection]
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, binomtest
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

Path('data/processed').mkdir(exist_ok=True)
Path('figures').mkdir(exist_ok=True)

FONT     = 'Georgia, serif'
BG, GRID = '#FAFAFA', '#E8E8E8'
D_COLOR  = '#1060C8'
R_COLOR  = '#C82010'
NAVY     = '#1a1a2e'
GRAY     = '#CCCCCC'
UPSET    = '#E05050'
SRC_PALETTE = ['#1a1a2e','#2d4a7a','#3d6b9e','#5a8ec4','#7fb3d9','#a8cceb','#c5dff0']

# ── Helpers ────────────────────────────────────────────────────────────────────

def style(fig, title, subtitle=None, w=820, h=520):
    txt = title + (f'<br><sup style="color:#555">{subtitle}</sup>' if subtitle else '')
    fig.update_layout(
        title=dict(text=txt, font=dict(family=FONT, size=17, color='#111'), x=0),
        font=dict(family=FONT, size=13),
        plot_bgcolor=BG, paper_bgcolor='white',
        width=w, height=h,
        margin=dict(l=72, r=50, t=88, b=64),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig

def save(fig, path):
    fig.write_image(path, scale=2)
    print(f'  {path}')

def pct_above_ci(fig, rates, ci_hi_errs, x_positions, color=NAVY, size=14, row=None, col=None):
    kw = dict(row=row, col=col) if row else {}
    for x, r, hi in zip(x_positions, rates, ci_hi_errs):
        fig.add_annotation(
            x=x, y=r + hi + 0.03,
            text=f'<b>{r:.0%}</b>',
            showarrow=False,
            font=dict(size=size, color=color),
            yanchor='bottom', xanchor='center',
            **kw,
        )

def pearson_r_ci(r, n, alpha=0.05):
    if abs(r) >= 0.999 or n <= 3:
        return -1.0, 1.0
    z, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))

def blank_fig(title, reason, w=820, h=520):
    """Return a styled blank placeholder figure."""
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5, xref='paper', yref='paper',
        text=f'<b>Data collection pending</b><br><span style="color:#777;font-size:13px">{reason}</span>',
        showarrow=False, font=dict(family=FONT, size=16, color='#444'),
        align='center',
    )
    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT, size=17, color='#111'), x=0),
        font=dict(family=FONT, size=13),
        plot_bgcolor='#F0F0F0', paper_bgcolor='white',
        width=w, height=h,
        margin=dict(l=72, r=50, t=88, b=64),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LOAD REAL DATA
# ══════════════════════════════════════════════════════════════════════════════

# Presidential Ngrams — vote_pct in percentage form (e.g., 49.72)
pres_ng = pd.read_csv('data/raw/ngrams/presidential_mention_shares.csv')
pres_ng['winner_vote_frac'] = pres_ng['winner_vote_pct'] / 100
pres_ng['loser_vote_frac']  = pres_ng['loser_vote_pct']  / 100
pres_ng['loser_mention_share'] = 1.0 - pres_ng['winner_mention_share']
pres_ng['gap'] = pres_ng['winner_mention_share'] / pres_ng['loser_mention_share']
pres_ng['margin'] = pres_ng['winner_vote_frac'] - pres_ng['loser_vote_frac']
pres_ng['upset'] = pres_ng['gap'] < 1

# Senate Ngrams — vote_pct already as fraction
senate_ng = pd.read_csv('data/raw/ngrams/senate_mention_shares.csv')
senate_ng['loser_mention_share'] = 1.0 - senate_ng['winner_mention_share']
senate_ng['loser_party'] = senate_ng['winner_party'].map({'D':'R','R':'D'})

# House Ngrams
house_ng = pd.read_csv('data/raw/ngrams/house_attention_shares.csv')

# Multi-source presidential data
trends_pres   = pd.read_csv('data/raw/trends/presidential_mention_shares.csv')
wiki_pres     = pd.read_csv('data/raw/wikipedia/presidential_mention_shares.csv')
gdelt_pres    = pd.read_csv('data/raw/gdelt_news/presidential_mention_shares.csv')
reddit_pres   = pd.read_csv('data/raw/reddit/presidential_mention_shares.csv')

# Convert vote pcts to fractions for non-Ngrams sources
for df in [trends_pres, wiki_pres, gdelt_pres, reddit_pres]:
    df['winner_vote_frac'] = df['winner_vote_pct'] / 100
    df['loser_vote_frac']  = df['loser_vote_pct']  / 100


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE STATS
# ══════════════════════════════════════════════════════════════════════════════

def h1_stats(df, col='attention_leader_won'):
    k, n = int(df[col].sum()), len(df)
    bi = binomtest(k, n, p=0.5, alternative='two-sided')
    ci = bi.proportion_ci()
    return k, n, k/n, ci.low, ci.high, bi.pvalue

def h2_stats(ms_col, vs_col, df):
    r, p = pearsonr(df[ms_col], df[vs_col])
    lo, hi = pearson_r_ci(r, len(df))
    return r, p, lo, hi

# Presidential Ngrams
k_p, n_p, rate_p, ci_p_lo, ci_p_hi, p_p = h1_stats(pres_ng)

# Build long-form pres scatter data for H2
pres_scatter = pd.concat([
    pres_ng[['year','winner_party','winner_mention_share','winner_vote_frac','winner_name']].rename(
        columns={'winner_party':'party','winner_mention_share':'ms','winner_vote_frac':'vs','winner_name':'name'}),
    pres_ng[['year','winner_party','loser_mention_share','loser_vote_frac','loser_name']].assign(
        winner_party=pres_ng['winner_party'].map({'D':'R','R':'D'})).rename(
        columns={'winner_party':'party','loser_mention_share':'ms','loser_vote_frac':'vs','loser_name':'name'}),
], ignore_index=True)
r_p, p_p_h2, r_p_lo, r_p_hi = h2_stats('ms', 'vs', pres_scatter)

# Senate Ngrams
k_s, n_s, rate_s, ci_s_lo, ci_s_hi, p_s = h1_stats(senate_ng)
senate_scatter = pd.concat([
    senate_ng[['year','winner_party','winner_mention_share','winner_vote_pct','winner']].rename(
        columns={'winner_party':'party','winner_mention_share':'ms','winner_vote_pct':'vs','winner':'name'}),
    senate_ng[['year','loser_party','loser_mention_share','loser_vote_pct','loser']].rename(
        columns={'loser_party':'party','loser_mention_share':'ms','loser_vote_pct':'vs','loser':'name'}),
], ignore_index=True)
r_s, p_s_h2, r_s_lo, r_s_hi = h2_stats('ms', 'vs', senate_scatter)

# House Ngrams
k_h, n_h, rate_h, ci_h_lo, ci_h_hi, p_h = h1_stats(house_ng, 'attn_predicts_majority')
r_h, p_h_h2, r_h_lo, r_h_hi = h2_stats('d_attention_share', 'd_seat_share', house_ng)

print(f'Presidential Ngrams:  H1={k_p}/{n_p} ({rate_p:.0%}), p={p_p:.4f}  H2: r={r_p:.3f}')
print(f'Senate Ngrams:        H1={k_s}/{n_s} ({rate_s:.0%}), p={p_s:.4f}  H2: r={r_s:.3f}')
print(f'House Ngrams:         H1={k_h}/{n_h} ({rate_h:.0%}), p={p_h:.4f}  H2: r={r_h:.3f}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating figures...')

# ── Figure 1: H1 presidential win rate (Ngrams) ───────────────────────────────
p1s = f'p = {p_p:.3f}' if p_p >= 0.001 else 'p < 0.001'
fig = go.Figure()
fig.add_trace(go.Bar(
    x=['Attention leader', 'Chance (50%)'],
    y=[rate_p, 0.5],
    error_y=dict(type='data',
                 array=[ci_p_hi - rate_p, 0],
                 arrayminus=[rate_p - ci_p_lo, 0],
                 thickness=2),
    marker_color=[NAVY, GRAY], marker_line_width=0, width=0.4,
))
pct_above_ci(fig, [rate_p, 0.5], [ci_p_hi - rate_p, 0], ['Attention leader', 'Chance (50%)'])
fig.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5)
fig.update_yaxes(range=[0, 1.18], tickformat='.0%', title='Win rate')
style(fig,
      f'Figure 1. Attention leader wins popular vote: {k_p}/{n_p} elections ({rate_p:.0%})',
      f'Google Ngrams · 1960–2016 · Binomial test vs. 50% — {p1s} — 95% CI [{ci_p_lo:.0%}, {ci_p_hi:.0%}]',
      w=560, h=500)
save(fig, 'figures/fig1_h1_win_rate.png')


# ── Figure 2: H2 scatter — Presidential + Senate ──────────────────────────────
fig = make_subplots(rows=1, cols=2,
    subplot_titles=[
        f'(A) Presidential elections (n={len(pres_scatter)})',
        f'(B) Senate races (n={len(senate_scatter):,})',
    ],
    horizontal_spacing=0.12)

# Panel A — Presidential Ngrams
for party, color, label in [('D', D_COLOR, 'Democrat'), ('R', R_COLOR, 'Republican')]:
    sub = pres_scatter[pres_scatter['party'] == party]
    fig.add_trace(go.Scatter(
        x=sub['ms'], y=sub['vs'], mode='markers',
        customdata=sub['name'] + " '" + sub['year'].astype(str).str[-2:],
        hovertemplate='%{customdata}<br>Mention: %{x:.1%} · Vote: %{y:.1%}<extra></extra>',
        marker=dict(color=color, size=10, line=dict(color='white', width=1.5)),
        name=label, legendgroup=label,
    ), row=1, col=1)

# Label notable points
notable = pres_scatter[
    (pres_scatter['ms'] > 0.72) | (pres_scatter['ms'] < 0.28) |
    (pres_scatter['vs'] > 0.58) | (pres_scatter['vs'] < 0.38)
]
fig.add_trace(go.Scatter(
    x=notable['ms'], y=notable['vs'], mode='text',
    text=notable['name'].str.split().str[-1] + " '" + notable['year'].astype(str).str[-2:],
    textposition='top center', textfont=dict(size=8, color='#444'),
    showlegend=False, hoverinfo='skip',
), row=1, col=1)

m_p, b_p = np.polyfit(pres_scatter['ms'], pres_scatter['vs'], 1)
xr_p = np.linspace(pres_scatter['ms'].min() - 0.02, pres_scatter['ms'].max() + 0.02, 200)
fig.add_trace(go.Scatter(x=xr_p, y=m_p * xr_p + b_p, mode='lines',
    line=dict(color='#555', dash='dash', width=1.5),
    name=f'OLS (r={r_p:.2f})',
), row=1, col=1)

# Panel B — Senate Ngrams
for party, color, label in [('D', D_COLOR, 'Democrat'), ('R', R_COLOR, 'Republican')]:
    sub = senate_scatter[senate_scatter['party'] == party]
    fig.add_trace(go.Scatter(
        x=sub['ms'], y=sub['vs'], mode='markers',
        marker=dict(color=color, size=4, opacity=0.35, line=dict(width=0)),
        name=label, legendgroup=label, showlegend=False,
    ), row=1, col=2)

m_s2, b_s2 = np.polyfit(senate_scatter['ms'], senate_scatter['vs'], 1)
xr_s = np.linspace(senate_scatter['ms'].min() - 0.01, senate_scatter['ms'].max() + 0.01, 200)
fig.add_trace(go.Scatter(x=xr_s, y=m_s2 * xr_s + b_s2, mode='lines',
    line=dict(color='#222', dash='dash', width=2),
    name=f'OLS (r={r_s:.2f})',
), row=1, col=2)

fig.update_xaxes(tickformat='.0%', title_text='Mention share (12 months pre-election)', row=1, col=1)
fig.update_xaxes(tickformat='.0%', title_text='Mention share (12 months pre-election)', row=1, col=2)
fig.update_yaxes(tickformat='.0%', title_text='Popular vote share', row=1, col=1)
fig.update_yaxes(tickformat='.0%', title_text='Vote share', row=1, col=2)
fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
style(fig,
      'Figure 2. Mention share vs. vote share — presidential and Senate',
      f'Google Ngrams · Pres r = {r_p:.2f} · Senate r = {r_s:.2f}',
      w=1100, h=540)
save(fig, 'figures/fig2_h2_scatter.png')


# ── Figure 3: Attention gap vs vote margin + timeline ─────────────────────────
h3 = pres_ng[['year','winner_name','loser_name','gap','margin','upset']].sort_values('year')
r3, _ = pearsonr(h3['gap'], h3['margin'])
m3, b3 = np.polyfit(h3['gap'], h3['margin'], 1)
xr3 = np.linspace(h3['gap'].min() - 0.05, h3['gap'].max() + 0.1, 200)

fig = make_subplots(rows=1, cols=2,
    subplot_titles=['(A) Attention gap vs. vote margin', '(B) Attention ratio by election year'],
    horizontal_spacing=0.12)

for upset, color, label in [(False, NAVY, 'Attention → winner'), (True, UPSET, 'Attention upset')]:
    sub = h3[h3['upset'] == upset]
    fig.add_trace(go.Scatter(
        x=sub['gap'], y=sub['margin'], mode='markers+text',
        text=sub['year'].astype(str), textposition='top center', textfont=dict(size=8),
        marker=dict(color=color, size=10, line=dict(color='white', width=1.5)), name=label,
    ), row=1, col=1)
fig.add_trace(go.Scatter(x=xr3, y=m3 * xr3 + b3, mode='lines',
    line=dict(color='#888', dash='dash', width=1.5), name=f'OLS (r={r3:.2f})',
), row=1, col=1)
fig.add_vline(x=1, line_dash='dot', line_color='#aaa', line_width=1.5, row=1, col=1)
fig.add_hline(y=0, line_dash='dot', line_color='#aaa', line_width=1, row=1, col=1)

bar_colors = [UPSET if u else NAVY for u in h3['upset']]
winner_labels = h3['winner_name'].str.split().str[-1] + ' ' + h3['gap'].map('{:.2f}×'.format)
fig.add_trace(go.Bar(
    x=h3['year'], y=h3['gap'],
    text=winner_labels,
    textposition='outside', textfont=dict(size=8),
    marker_color=bar_colors, marker_line_width=0, width=3,
    showlegend=False,
), row=1, col=2)
fig.add_hline(y=1, line_dash='dot', line_color='#aaa', line_width=1.5,
    annotation_text='Equal (1×)', annotation_position='top left', row=1, col=2)

fig.update_xaxes(title_text='Mention ratio (winner ÷ loser)', row=1, col=1)
fig.update_yaxes(tickformat='.1%', title_text='Popular vote margin', row=1, col=1)
fig.update_xaxes(title_text='Election year', tickmode='array',
    tickvals=h3['year'].tolist(), tickangle=-55, tickfont=dict(size=10), row=1, col=2)
fig.update_yaxes(title_text='Attention ratio (winner ÷ loser)',
    range=[0, h3['gap'].max() * 1.25], row=1, col=2)
fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
style(fig,
      'Figure 3. Attention gap: does bigger coverage lead predict bigger win?',
      f'Google Ngrams · 1960–2016 · Red = elections where attention underdog won · r={r3:.2f}',
      w=1100, h=520)
save(fig, 'figures/fig3_gap_and_timeline.png')


# ── Figure 4: Cross-level win rate comparison ─────────────────────────────────
rates4 = [rate_p, rate_s, rate_h]
hi4    = [ci_p_hi - rate_p, ci_s_hi - rate_s, ci_h_hi - rate_h]
lo4    = [rate_p - ci_p_lo, rate_s - ci_s_lo, rate_h - ci_h_lo]
p4     = [p_p, p_s, p_h]
xlabs4 = [
    f'Presidential<br>(n={n_p} elections)',
    f'Senate<br>(n={n_s:,} races)',
    f'House cycles<br>(n={n_h} cycles)',
]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=xlabs4, y=rates4,
    error_y=dict(type='data', array=hi4, arrayminus=lo4, thickness=2),
    marker_color=[NAVY, '#2d4a7a', '#3d6b9e'], marker_line_width=0, width=0.45,
    customdata=[f'{"p < 0.001" if pv < 0.001 else f"p = {pv:.3f}"}  n={n:,}'
                for pv, n in zip(p4, [n_p, n_s, n_h])],
    hovertemplate='%{x}<br>Win rate: %{y:.1%}<br>%{customdata}<extra></extra>',
))
pct_above_ci(fig, rates4, hi4, xlabs4)
fig.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5,
    annotation_text='Chance (50%)', annotation_position='right')
fig.update_yaxes(range=[0, 1.18], tickformat='.0%', title='Attention leader win rate')
p4l = ['p < 0.001' if pv < 0.001 else f'p = {pv:.3f}' for pv in p4]
style(fig,
      'Figure 4. Attention leader win rate at every level of government',
      f'Google Ngrams · Presidential {p4l[0]} · Senate {p4l[1]} · House {p4l[2]}',
      w=700, h=540)
save(fig, 'figures/fig4_win_rate_comparison.png')


# ── Figure 5: House party attention share vs. seat share ──────────────────────
r7 = r_h
fig = go.Figure()
fig.add_shape(type='rect', x0=0.5, x1=0.68, y0=0.5, y1=0.75,
    fillcolor='rgba(16,96,200,0.05)', line_width=0)
fig.add_shape(type='rect', x0=0.32, x1=0.5, y0=0.25, y1=0.5,
    fillcolor='rgba(200,32,16,0.05)', line_width=0)
for maj, color, label in [('D', D_COLOR, 'Dem majority'), ('R', R_COLOR, 'Rep majority')]:
    sub = house_ng[house_ng['majority'] == maj]
    fig.add_trace(go.Scatter(
        x=sub['d_attention_share'], y=sub['d_seat_share'],
        mode='markers+text', text=sub['year'].astype(str),
        textposition='top center', textfont=dict(size=9),
        marker=dict(color=color, size=10, line=dict(color='white', width=1.5)), name=label,
        customdata=sub[['year', 'd_seats']].values,
        hovertemplate='%{customdata[0]}<br>D attention: %{x:.1%}<br>D seats: %{customdata[1]}<extra></extra>',
    ))
m7, b7 = np.polyfit(house_ng['d_attention_share'], house_ng['d_seat_share'], 1)
xr7 = np.linspace(house_ng['d_attention_share'].min() - 0.01,
                   house_ng['d_attention_share'].max() + 0.01, 200)
fig.add_trace(go.Scatter(x=xr7, y=m7 * xr7 + b7, mode='lines',
    line=dict(color='#666', dash='dash', width=1.5), name=f'OLS (r={r7:.2f})'))
fig.add_vline(x=0.5, line_dash='dot', line_color='#aaa', line_width=1.2)
fig.add_hline(y=0.5, line_dash='dot', line_color='#aaa', line_width=1.2)
p7s = 'p < 0.001' if p_h_h2 < 0.001 else f'p = {p_h_h2:.3f}'
fig.update_xaxes(tickformat='.0%', title='Democratic attention share (that cycle)', range=[0.33, 0.67])
fig.update_yaxes(tickformat='.0%', title='Democratic seat share (out of 435)', range=[0.38, 0.72])
style(fig,
      f'Figure 5. House: does the party dominating coverage win more seats? (r = {r7:.2f})',
      f'Google Ngrams · 1962–2024 · {p7s}',
      w=820, h=600)
save(fig, 'figures/fig5_house_seat_share.png')


# ── Figure 6: Attention gap distribution ─────────────────────────────────────
gaps_p = pres_ng[['year', 'gap']].copy()
gaps_s = senate_ng.assign(gap=senate_ng['winner_mention_share'] / senate_ng['loser_mention_share'])[['year', 'gap']]

fig = go.Figure()
fig.add_trace(go.Box(
    y=gaps_s['gap'], name=f'Senate (n={len(gaps_s):,})',
    boxpoints=False,
    marker_color='#3d6b9e',
    line=dict(color='#2d4a7a', width=2),
    fillcolor='rgba(61,107,158,0.18)',
    boxmean='sd',
    width=0.35,
))
fig.add_trace(go.Box(
    y=gaps_p['gap'], name=f'Presidential (n={len(gaps_p)})',
    boxpoints='all',
    jitter=0.35,
    pointpos=0,
    marker=dict(color=NAVY, size=8, opacity=0.85, line=dict(color='white', width=1.5)),
    line=dict(color=NAVY, width=2),
    fillcolor='rgba(26,26,46,0.15)',
    boxmean='sd',
    width=0.35,
))
fig.add_hline(y=1, line_dash='dot', line_color='#555', line_width=2,
    annotation_text='Equal attention (1×)', annotation_position='top right',
    annotation_font=dict(size=11))
fig.update_yaxes(title='Mention ratio (winner ÷ loser)', range=[0, None])
style(fig,
      'Figure 6. Distribution of attention gaps by race type',
      'Google Ngrams · Values > 1× = winner received more coverage · Box: median, IQR, ±1 SD',
      w=680, h=560)
save(fig, 'figures/fig6_gap_distribution.png')


# ── Figure 7: Multi-source forest plot ────────────────────────────────────────
# Build source stats from real data (presidential only for cross-source comparison)
def source_entry(name, df, color, desc):
    k, n, rate, lo, hi, pval = h1_stats(df)
    ms = df['winner_mention_share']
    vs = df['winner_vote_frac'] if 'winner_vote_frac' in df.columns else df['winner_vote_pct'] / 100
    if n >= 3:
        r, _, r_lo, r_hi = h2_stats('winner_mention_share',
                                     'winner_vote_frac' if 'winner_vote_frac' in df.columns else 'winner_vote_pct',
                                     df)
        if 'winner_vote_frac' not in df.columns:
            r, _, r_lo, r_hi = pearsonr(ms, vs / 100 if vs.max() > 2 else vs), None, None, None
            r, p2 = pearsonr(ms, df['winner_vote_pct'] / 100)
            r_lo, r_hi = pearson_r_ci(r, n)
    else:
        r, r_lo, r_hi = float('nan'), float('nan'), float('nan')
    return dict(name=name, color=color, desc=desc,
                n=n, rate=rate, lo=lo, hi=hi, pval=pval,
                r=r, r_lo=r_lo, r_hi=r_hi)

sources_real = [
    source_entry('Google Ngrams',       pres_ng,    SRC_PALETTE[1], 'Book corpus · 1960–2016'),
    source_entry('Google Trends',       trends_pres, SRC_PALETTE[2], 'Search volume · 2004–2024'),
    source_entry('Wikipedia Pageviews', wiki_pres,  SRC_PALETTE[3], 'Article views · 2016–2024'),
    source_entry('GDELT News',          gdelt_pres, SRC_PALETTE[0], 'News articles · 2020–2024'),
    source_entry('Reddit',              reddit_pres,SRC_PALETTE[5], 'r/politics posts · 2008–2012'),
]

# Sort by H1 win rate
src_df = pd.DataFrame(sources_real).sort_values('rate', ascending=False)

PENDING = [
    ('GDELT TV',    SRC_PALETTE[4], 'Broadcast transcripts · collection pending'),
    ('MediaCloud',  SRC_PALETTE[6], 'Academic news index · collection pending'),
]

fig = make_subplots(rows=1, cols=2,
    subplot_titles=['(A) H1: attention leader win rate', '(B) H2: mention share → vote share (r)'],
    horizontal_spacing=0.18)

all_names = list(src_df['name']) + [p[0] for p in PENDING]

for _, s in src_df.iterrows():
    n_str = f"n={s['n']} elections"
    fig.add_trace(go.Scatter(
        x=[s['rate']], y=[s['name']],
        error_x=dict(type='data', symmetric=False,
                     array=[s['hi'] - s['rate']], arrayminus=[s['rate'] - s['lo']],
                     thickness=2, width=6),
        mode='markers+text',
        text=[n_str], textposition='middle right', textfont=dict(size=9, color='#777'),
        marker=dict(color=s['color'], size=13, symbol='circle',
                    line=dict(color='white', width=1.5)),
        name=s['name'], showlegend=False,
        hovertemplate=f"{s['name']}<br>Win rate: %{{x:.0%}}<br>{n_str}<extra></extra>",
    ), row=1, col=1)

    if not np.isnan(s['r']):
        r_lo = s['r_lo'] if not np.isnan(s['r_lo']) else s['r']
        r_hi = s['r_hi'] if not np.isnan(s['r_hi']) else s['r']
        fig.add_trace(go.Scatter(
            x=[s['r']], y=[s['name']],
            error_x=dict(type='data', symmetric=False,
                         array=[r_hi - s['r']], arrayminus=[s['r'] - r_lo],
                         thickness=2, width=6),
            mode='markers',
            marker=dict(color=s['color'], size=13, symbol='circle',
                        line=dict(color='white', width=1.5)),
            showlegend=False,
            hovertemplate=f"{s['name']}<br>r = %{{x:.2f}}<br>{n_str}<extra></extra>",
        ), row=1, col=2)

# Pending rows (hollow markers)
for pname, pcolor, pdesc in PENDING:
    for col, xval in [(1, 0.5), (2, 0.0)]:
        fig.add_trace(go.Scatter(
            x=[xval], y=[pname],
            mode='markers+text',
            text=['pending'] if col == 1 else [''],
            textposition='middle right', textfont=dict(size=9, color='#bbb'),
            marker=dict(color='white', size=13, symbol='circle',
                        line=dict(color='#bbb', width=2)),
            showlegend=False,
            hovertemplate=f'{pname}<br>{pdesc}<extra></extra>',
        ), row=1, col=col)

fig.add_vline(x=0.5, line_dash='dot', line_color='#aaa', line_width=1.5,
    annotation_text='Chance', annotation_position='bottom', row=1, col=1)
fig.add_vline(x=0.0, line_dash='dot', line_color='#aaa', line_width=1.5,
    annotation_text='r = 0', annotation_position='bottom', row=1, col=2)

fig.update_xaxes(tickformat='.0%', range=[0, 1.35], title_text='Win rate', row=1, col=1)
fig.update_xaxes(tickformat='.2f', range=[-0.6, 1.1], title_text='Pearson r', row=1, col=2)
fig.update_yaxes(categoryorder='array', categoryarray=all_names[::-1])
fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
fig.update_yaxes(showgrid=False)
style(fig,
      'Figure 7. Does the finding replicate across 7 independent data sources?',
      'Presidential elections · n varies by source coverage · 2 sources collection pending',
      w=1050, h=500)
save(fig, 'figures/fig7_multi_source_forest.png')


# ── Figure S1: By media era (Ngrams presidential, 1960–2016) ──────────────────
ERAS = [
    ('Print / Radio\n1960–1972', [1960, 1964, 1968, 1972]),
    ('Network TV\n1976–1988',    [1976, 1980, 1984, 1988]),
    ('Cable / Web\n1992–2008',   [1992, 1996, 2000, 2004, 2008]),
    ('Social Media\n2012–2016*', [2012, 2016]),
]
era_r, era_hi, era_lo, era_ns = [], [], [], []
for _, yrs in ERAS:
    sub = pres_ng[pres_ng['year'].isin(yrs)]
    k_e, n_e = int(sub['attention_leader_won'].sum()), len(sub)
    bi_e = binomtest(k_e, n=n_e, p=0.5)
    ci_e = bi_e.proportion_ci()
    era_r.append(k_e / n_e)
    era_hi.append(ci_e.high - k_e / n_e)
    era_lo.append(k_e / n_e - ci_e.low)
    era_ns.append(n_e)

era_labs = [e[0] for e in ERAS]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=era_labs, y=era_r, mode='markers+lines',
    error_y=dict(type='data', array=era_hi, arrayminus=era_lo, thickness=2),
    marker=dict(color=[SRC_PALETTE[i] for i in range(4)], size=14,
                line=dict(color='white', width=2)),
    line=dict(color='#aaa', dash='dot', width=1.5),
    showlegend=False,
    customdata=[f'n={n}' for n in era_ns],
    hovertemplate='%{x}<br>%{y:.0%}<br>%{customdata}<extra></extra>',
))
for xl, r, hi, n in zip(era_labs, era_r, era_hi, era_ns):
    fig.add_annotation(
        x=xl, y=r + hi + 0.03,
        text=f'<b>{r:.0%}</b><br><span style="font-size:10px">n={n}</span>',
        showarrow=False, font=dict(size=12, color=NAVY),
        yanchor='bottom', xanchor='center')
fig.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5,
    annotation_text='Chance (50%)', annotation_position='right')
fig.update_yaxes(range=[0, 1.2], tickformat='.0%', title='Attention leader win rate')
style(fig,
      'Figure S1. Does the attention–outcome link vary by media era?',
      'Google Ngrams · 1960–2016 · *Social Media era limited to 2012–2016 (Ngrams corpus ends 2019)',
      w=720, h=520)
save(fig, 'figures/figS1_by_era.png')


# ── Figure S2: PENDING (multi-window real collection required) ────────────────
fig = blank_fig(
    'Figure S2. Sensitivity: does the measurement window length matter?',
    'Requires collecting 3-month and 6-month pre-election windows from each source.<br>'
    'Only the 12-month window has been collected so far.',
    w=640, h=500)
save(fig, 'figures/figS2_window_sensitivity.png')


# ── Table S1: Real presidential data ─────────────────────────────────────────
tbl = pres_ng[['year', 'winner_name', 'winner_party', 'winner_mention_share',
               'winner_vote_frac', 'loser_name', 'loser_mention_share',
               'loser_vote_frac', 'attention_leader_won']].copy()
tbl.columns = ['Year', 'Winner', 'Party', 'Winner mention share',
               'Winner vote share', 'Loser', 'Loser mention share',
               'Loser vote share', 'Attention leader won']
tbl['Winner mention share'] = tbl['Winner mention share'].map('{:.1%}'.format)
tbl['Loser mention share']  = tbl['Loser mention share'].map('{:.1%}'.format)
tbl['Winner vote share']    = tbl['Winner vote share'].map('{:.1%}'.format)
tbl['Loser vote share']     = tbl['Loser vote share'].map('{:.1%}'.format)
tbl['Attention leader won'] = tbl['Attention leader won'].map({1: '✓', 0: ''})
tbl.to_csv('data/processed/table_s1_presidential_real.csv', index=False)
print('  data/processed/table_s1_presidential_real.csv')

print('\nAll done. ✓')
