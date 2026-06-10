"""
Real-data figures for 'Is Attention All You Need (to Win Elections)?'

Real data used everywhere we have a complete dataset.
Figures without complete real data show a blank "pending" placeholder —
no synthetic data is mixed into any figure.

Figure inventory:
  fig1  — H1 presidential win rate          [REAL: Ngrams presidential]
  fig2  — H2 scatter pres + senate          [REAL: Ngrams pres + senate]
  fig3  — Attention gap vs margin           [REAL: Ngrams presidential]
  fig4  — Cross-level win rate comparison   [REAL: Ngrams pres + senate + governor + house]
  fig5  — House party attention vs seats    [REAL: Ngrams house]
  fig6  — Gap distribution box              [REAL: Ngrams pres + senate + governor + house]
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
D_COLOR  = '#1060C8'   # Democratic party blue
R_COLOR  = '#C82010'   # Republican party red
NAVY     = '#1a1a2e'
GRAY     = '#CCCCCC'
# Attention outcome colors — deliberately different from D/R party colors
WINNER_CLR = '#2c7873'   # teal  — attention → winner
UPSET_CLR  = '#f4a261'   # amber — attention underdog won
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

def ols_label(fig, r, xdata, ydata, row=None, col=None, color='#444', dx=0, dy=0):
    """Annotate OLS line near its midpoint with r value."""
    x_mid = float(np.percentile(xdata, 60))
    m, b  = np.polyfit(xdata, ydata, 1)
    y_mid = m * x_mid + b
    kw = dict(row=row, col=col) if row else {}
    fig.add_annotation(
        x=x_mid + dx, y=y_mid + dy,
        text=f'<i>r</i> = {r:.2f}',
        showarrow=False,
        font=dict(size=11, color=color),
        bgcolor='rgba(255,255,255,0.75)',
        borderpad=2,
        xanchor='left', yanchor='bottom',
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

# Governor Ngrams — vote_pct as fraction (matches Senate convention)
gov_ng = pd.read_csv('data/raw/ngrams/governors_mention_shares.csv')
gov_ng['margin'] = gov_ng['winner_vote_pct'] - gov_ng['loser_vote_pct']

senate_ng['margin'] = senate_ng['winner_vote_pct'] - senate_ng['loser_vote_pct']

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

# Governor Ngrams
k_g, n_g, rate_g, ci_g_lo, ci_g_hi, p_g = h1_stats(gov_ng)
r_g, p_g_h2, r_g_lo, r_g_hi = h2_stats('winner_mention_share', 'winner_vote_pct', gov_ng)

# House Ngrams
k_h, n_h, rate_h, ci_h_lo, ci_h_hi, p_h = h1_stats(house_ng, 'attn_predicts_majority')
r_h, p_h_h2, r_h_lo, r_h_hi = h2_stats('d_attention_share', 'd_seat_share', house_ng)

print(f'Presidential Ngrams:  H1={k_p}/{n_p} ({rate_p:.0%}), p={p_p:.4f}  H2: r={r_p:.3f}')
print(f'Senate Ngrams:        H1={k_s}/{n_s} ({rate_s:.0%}), p={p_s:.4f}  H2: r={r_s:.3f}')
print(f'Governor Ngrams:      H1={k_g}/{n_g} ({rate_g:.0%}), p={p_g:.4f}  H2: r={r_g:.3f}')
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
    showlegend=False,
), row=1, col=1)
ols_label(fig, r_p, pres_scatter['ms'].values, pres_scatter['vs'].values, row=1, col=1)

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
    showlegend=False,
), row=1, col=2)
ols_label(fig, r_s, senate_scatter['ms'].values, senate_scatter['vs'].values, row=1, col=2)

senate_pct_extreme = (senate_scatter['ms'] > 0.95).mean() + (senate_scatter['ms'] < 0.05).mean()
fig.update_xaxes(tickformat='.0%', title_text='Mention share (12 months pre-election)', row=1, col=1)
fig.update_xaxes(tickformat='.0%', title_text='Mention share (12 months pre-election)', row=1, col=2)
fig.update_yaxes(tickformat='.0%', title_text='Popular vote share', row=1, col=1)
fig.update_yaxes(tickformat='.0%', title_text='Vote share', row=1, col=2)
fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
style(fig,
      'Figure 2. Mention share vs. vote share — presidential and Senate',
      f'Google Ngrams · Pres r = {r_p:.2f} · Senate r = {r_s:.2f} · '
      f'{senate_pct_extreme:.0%} of Senate races have one candidate at <5% or >95% share '
      f'(famous incumbents vs. unknown challengers)',
      w=1100, h=540)
save(fig, 'figures/fig2_h2_scatter.png')


# ── Figure 3: Attention gap vs vote margin + timeline ─────────────────────────
h3 = pres_ng[['year','winner_name','loser_name','gap','margin','upset']].sort_values('year')
r3, _ = pearsonr(h3['gap'], h3['margin'])
m3, b3 = np.polyfit(h3['gap'], h3['margin'], 1)
xr3 = np.linspace(h3['gap'].min() - 0.05, h3['gap'].max() + 0.1, 200)

# Senate gap vs margin
senate_ng['gap'] = senate_ng['winner_mention_share'] / (1 - senate_ng['winner_mention_share'])
senate_ng['margin'] = senate_ng['winner_vote_pct'] - senate_ng['loser_vote_pct']
senate_ng['upset'] = senate_ng['gap'] < 1

# Use log(gap) for OLS since gap is right-skewed
from scipy.stats import pearsonr as _pr
r3s_log, _ = _pr(np.log(senate_ng['gap']), senate_ng['margin'])
s_sub = senate_ng[senate_ng['gap'] <= senate_ng['gap'].quantile(0.95)]
m3s, b3s = np.polyfit(s_sub['gap'], s_sub['margin'], 1)

fig = make_subplots(rows=1, cols=2,
    subplot_titles=[
        f'(A) Presidential — gap vs. vote margin (n={len(h3)})',
        f'(B) Senate — gap vs. vote margin (n={len(senate_ng):,})',
    ],
    horizontal_spacing=0.12)

# Panel A — Presidential (teal/amber, no party colors)
for upset, color, label in [
    (False, WINNER_CLR, 'Attention → winner'),
    (True,  UPSET_CLR,  'Attention upset'),
]:
    sub = h3[h3['upset'] == upset]
    fig.add_trace(go.Scatter(
        x=sub['gap'], y=sub['margin'], mode='markers+text',
        text=sub['year'].astype(str), textposition='top center', textfont=dict(size=8),
        marker=dict(color=color, size=10, line=dict(color='white', width=1.5)), name=label,
    ), row=1, col=1)
fig.add_trace(go.Scatter(x=xr3, y=m3 * xr3 + b3, mode='lines',
    line=dict(color='#888', dash='dash', width=1.5), showlegend=False,
), row=1, col=1)
ols_label(fig, r3, h3['gap'], h3['margin'], row=1, col=1)
fig.add_vline(x=1, line_dash='dot', line_color='#aaa', line_width=1.5, row=1, col=1)
fig.add_hline(y=0, line_dash='dot', line_color='#aaa', line_width=1, row=1, col=1)

# Panel B — Senate (teal/amber, log x-axis)
for upset, color, label in [
    (False, WINNER_CLR, 'Attention → winner'),
    (True,  UPSET_CLR,  'Attention upset'),
]:
    sub = senate_ng[senate_ng['upset'] == upset]
    fig.add_trace(go.Scatter(
        x=sub['gap'], y=sub['margin'], mode='markers',
        marker=dict(color=color, size=4, opacity=0.45, line=dict(width=0)),
        name=label, legendgroup=label, showlegend=False,
        hovertemplate='%{x:.1f}× gap · %{y:.1%} margin<extra></extra>',
    ), row=1, col=2)

log_gap_s = np.log10(senate_ng['gap'].clip(lower=0.001))
m3s_log, b3s_log = np.polyfit(log_gap_s, senate_ng['margin'], 1)
xlog_range = np.linspace(log_gap_s.min(), log_gap_s.quantile(0.99), 200)
fig.add_trace(go.Scatter(
    x=10**xlog_range, y=m3s_log * xlog_range + b3s_log, mode='lines',
    line=dict(color='#444', dash='dash', width=2), showlegend=False,
), row=1, col=2)
# r label near midpoint of log-scale line
x_mid_log = 10 ** float(np.percentile(log_gap_s, 60))
y_mid_log  = m3s_log * np.log10(x_mid_log) + b3s_log
fig.add_annotation(x=x_mid_log, y=y_mid_log + 0.01,
    text=f'<i>r</i>(log) = {r3s_log:.2f}', showarrow=False,
    font=dict(size=11, color='#444'), bgcolor='rgba(255,255,255,0.75)',
    borderpad=2, xanchor='left', yanchor='bottom', row=1, col=2)
fig.add_vline(x=1, line_dash='dot', line_color='#aaa', line_width=1.5, row=1, col=2)
fig.add_hline(y=0, line_dash='dot', line_color='#aaa', line_width=1, row=1, col=2)

fig.update_xaxes(title_text='Mention ratio (winner ÷ loser)', row=1, col=1)
fig.update_xaxes(title_text='Mention ratio (winner ÷ loser, log scale)',
    type='log', row=1, col=2)
fig.update_yaxes(tickformat='.1%', title_text='Popular vote margin', row=1, col=1)
fig.update_yaxes(tickformat='.1%', title_text='Popular vote margin', row=1, col=2)
fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
style(fig,
      'Figure 3. Attention gap vs. vote margin — presidential and Senate',
      f'Google Ngrams · Pres r={r3:.2f} · Senate r(log)={r3s_log:.2f} · '
      f'Amber = attention underdog won',
      w=1100, h=520)
save(fig, 'figures/fig3_gap_and_timeline.png')


# ── Figure 4: Win rate + vote margin by level — 2 panels ─────────────────────
# Note: House is party-level (29 election cycles, 'Democrats' vs 'Republicans'
# in Ngrams). Individual House candidates are too obscure for Ngrams.

rates4 = [rate_p, rate_s, rate_g, rate_h]
hi4    = [ci_p_hi - rate_p, ci_s_hi - rate_s, ci_g_hi - rate_g, ci_h_hi - rate_h]
lo4    = [rate_p - ci_p_lo, rate_s - ci_s_lo, rate_g - ci_g_lo, rate_h - ci_h_lo]
p4     = [p_p, p_s, p_g, p_h]
xlabs4 = ['Presidential', 'Senate', 'Governor', 'House\n(party-level)']
bar_colors4 = [NAVY, '#2d4a7a', '#3d6b9e', '#5a8ec4']

# Attention leader's own outcome share per race (continuous analogue of "did
# they win"): vote share for candidate levels, winning-party seat share for the
# House. Clusters above 50% exactly when the attention leader tends to win.
pres_ng['leader_share'] = np.where(pres_ng['attention_leader_won'] == 1,
                                   pres_ng['winner_vote_frac'], pres_ng['loser_vote_frac'])
senate_ng['leader_share'] = np.where(senate_ng['attention_leader_won'] == 1,
                                     senate_ng['winner_vote_pct'], senate_ng['loser_vote_pct'])
gov_ng['leader_share'] = np.where(gov_ng['attention_leader_won'] == 1,
                                  gov_ng['winner_vote_pct'], gov_ng['loser_vote_pct'])
house_ng['leader_share'] = np.where(house_ng['attention_majority'] == 'D',
                                    house_ng['d_seat_share'], house_ng['r_seat_share'])

fig = make_subplots(rows=1, cols=3,
    subplot_titles=[
        '(A) H1: attention leader win rate',
        '(B) Attention leader vote / seat share',
        '(C) Vote margin: leaders vs underdogs',
    ],
    horizontal_spacing=0.075)

# Panel A: bars + raw data jittered
rng4 = np.random.default_rng(99)
# Numeric x positions (0,1,2,3) so the jittered scatter overlays align with
# the bars without polluting a category axis with their jitter values.
for xi, (xlab, rate, hi, lo, bc, pv) in enumerate(zip(xlabs4, rates4, hi4, lo4, bar_colors4, p4)):
    fig.add_trace(go.Bar(
        x=[xi], y=[rate],
        error_y=dict(type='data', array=[hi], arrayminus=[lo], thickness=2),
        marker_color=bc, marker_line_width=0, width=0.45,
        hovertemplate=f'{xlab}<br>Win rate: {rate:.1%}<br>'
                      f'{"p < 0.001" if pv < 0.001 else f"p = {pv:.3f}"}<extra></extra>',
        showlegend=False,
    ), row=1, col=1)

# Raw outcome points overlaid (jittered x)
# Presidential
pres_outcomes = pres_ng['attention_leader_won'].values.astype(float)
fig.add_trace(go.Scatter(
    x=rng4.uniform(-0.15, 0.15, len(pres_outcomes)) + 0,
    y=pres_outcomes + rng4.uniform(-0.02, 0.02, len(pres_outcomes)),
    xaxis='x', yaxis='y',
    mode='markers',
    marker=dict(color=WINNER_CLR, size=7, opacity=0.7, line=dict(color='white', width=1)),
    showlegend=False, hoverinfo='skip',
), row=1, col=1)
# Senate (too many to show all — sample 200)
senate_samp = senate_ng.sample(min(200, len(senate_ng)), random_state=42)
senate_outcomes = senate_samp['attention_leader_won'].values.astype(float)
fig.add_trace(go.Scatter(
    x=rng4.uniform(-0.18, 0.18, len(senate_outcomes)) + 1,
    y=senate_outcomes + rng4.uniform(-0.02, 0.02, len(senate_outcomes)),
    mode='markers',
    marker=dict(color=WINNER_CLR, size=4, opacity=0.4, line=dict(width=0)),
    showlegend=False, hoverinfo='skip',
), row=1, col=1)
# Governor (too many to show all — sample 200)
gov_samp = gov_ng.sample(min(200, len(gov_ng)), random_state=42)
gov_outcomes = gov_samp['attention_leader_won'].values.astype(float)
fig.add_trace(go.Scatter(
    x=rng4.uniform(-0.18, 0.18, len(gov_outcomes)) + 2,
    y=gov_outcomes + rng4.uniform(-0.02, 0.02, len(gov_outcomes)),
    mode='markers',
    marker=dict(color=WINNER_CLR, size=4, opacity=0.4, line=dict(width=0)),
    showlegend=False, hoverinfo='skip',
), row=1, col=1)
# House
house_outcomes = house_ng['attn_predicts_majority'].values.astype(float)
fig.add_trace(go.Scatter(
    x=rng4.uniform(-0.12, 0.12, len(house_outcomes)) + 3,
    y=house_outcomes + rng4.uniform(-0.02, 0.02, len(house_outcomes)),
    mode='markers',
    marker=dict(color=WINNER_CLR, size=7, opacity=0.7, line=dict(color='white', width=1)),
    showlegend=False, hoverinfo='skip',
), row=1, col=1)

pct_above_ci(fig, rates4, hi4, list(range(len(xlabs4))), row=1, col=1)
fig.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5,
    annotation_text='Chance', annotation_position='right', row=1, col=1)
fig.add_hline(y=1.0, line_dash='dot', line_color='#ccc', line_width=1, row=1, col=1)
fig.add_hline(y=0.0, line_dash='dot', line_color='#ccc', line_width=1, row=1, col=1)
fig.update_yaxes(range=[-0.15, 1.25], tickformat='.0%',
    title='Attention leader win rate', row=1, col=1)
fig.update_xaxes(row=1, col=1, tickmode='array',
    tickvals=list(range(len(xlabs4))),
    ticktext=[x.replace('\n', '<br>') for x in xlabs4],
    range=[-0.5, len(xlabs4) - 0.5])

# ── Panel B: attention leader's vote/seat share — box + all raw points ────────
B_LEVELS = [
    (pres_ng,   'Presidential',            8, None),
    (senate_ng, 'Senate',                  3, 250),
    (gov_ng,    'Governor',                3, 250),
    (house_ng,  'House<br>(party-level)',  6, None),
]
for sub, xlab, ptsize, samp in B_LEVELS:
    vals = sub['leader_share']
    if samp and len(vals) > samp:
        vals = vals.sample(samp, random_state=7)
    fig.add_trace(go.Box(
        x=[xlab] * len(vals), y=vals,
        name=xlab, showlegend=False,
        marker_color=WINNER_CLR, line_color='#1f5a55',
        fillcolor='rgba(44,120,115,0.15)',
        boxpoints='all', jitter=0.4, pointpos=0,
        marker=dict(size=ptsize, opacity=0.45, line=dict(color='white', width=0.5)),
        width=0.45,
    ), row=1, col=2)
fig.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5,
    annotation_text='50%', annotation_position='right', row=1, col=2)
fig.update_yaxes(tickformat='.0%', title='Attention leader vote / seat share',
                 range=[0, 1], row=1, col=2)
fig.update_xaxes(categoryorder='array',
    categoryarray=['Presidential', 'Senate', 'Governor', 'House<br>(party-level)'], row=1, col=2)

# ── Panel C: vote margin distribution by level, split by attention outcome ─────
for attn_won, color, label, show_leg in [
    (1, WINNER_CLR, 'Attention leader won', True),
    (0, UPSET_CLR,  'Attention underdog won', True),
]:
    sub_p = pres_ng[pres_ng['attention_leader_won'] == attn_won]
    sub_s = senate_ng[senate_ng['attention_leader_won'] == attn_won]
    sub_g = gov_ng[gov_ng['attention_leader_won'] == attn_won]
    sub_h = house_ng[house_ng['attn_predicts_majority'] == attn_won]

    for xi, (sub, xlab) in enumerate([(sub_p, 'Presidential'),
                                       (sub_s, 'Senate'),
                                       (sub_g, 'Governor'),
                                       (sub_h, 'House<br>(party-level)')]):
        margin_col = 'margin' if 'margin' in sub.columns else \
                     sub.assign(margin=abs(sub['d_seat_share'] - 0.5) * 2)['margin'] \
                     if 'd_seat_share' in sub.columns else None
        if margin_col is None:
            continue
        y_vals = sub[margin_col] if isinstance(margin_col, str) else margin_col

        fig.add_trace(go.Box(
            x=[xlab] * len(y_vals), y=y_vals,
            name=label, legendgroup=label, showlegend=(show_leg and xi == 0),
            marker_color=color, line_color=color,
            fillcolor=color.replace(')', ', 0.15)').replace('rgb', 'rgba') if 'rgb' in color
                else f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)',
            boxpoints='all', jitter=0.4, pointpos=0,
            marker=dict(size=5 if xlab not in ('Senate', 'Governor') else 3, opacity=0.5),
            width=0.35,
        ), row=1, col=3)

fig.update_yaxes(tickformat='.1%', title='Vote margin (winner − loser)', row=1, col=3)
fig.update_xaxes(categoryorder='array',
    categoryarray=['Presidential', 'Senate', 'Governor', 'House<br>(party-level)'], row=1, col=3)
fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
p4l = ['p < 0.001' if pv < 0.001 else f'p = {pv:.3f}' for pv in p4]
style(fig,
      'Figure 4. Attention leader win rate, vote share, and margin at every level of government',
      f'Google Ngrams · Presidential {p4l[0]} · Senate {p4l[1]} · Governor {p4l[2]} · House {p4l[3]}',
      w=1480, h=560)
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
    line=dict(color='#666', dash='dash', width=1.5), showlegend=False))
ols_label(fig, r7, house_ng['d_attention_share'].values, house_ng['d_seat_share'].values)
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

# House: winner party attn share / loser party attn share
house_ng['winner_attn'] = house_ng.apply(
    lambda r: r['d_attention_share'] if r['majority'] == 'D' else r['r_attention_share'], axis=1)
house_ng['loser_attn']  = 1.0 - house_ng['winner_attn']
gaps_h = house_ng.assign(gap=house_ng['winner_attn'] / house_ng['loser_attn'])[['year', 'gap']]
gaps_g = gov_ng.assign(gap=gov_ng['winner_mention_share'] / gov_ng['loser_mention_share'])[['year', 'gap']]

# Plot log10(gap) on a linear axis (with ×-style tick labels) so the violin
# KDE is computed in log space, where attention ratios are roughly symmetric
# and the density reads cleanly. Each trace overlays violin + inner box + all
# raw points (the box chart's data, kept).
BOX_CONFIGS = [
    (gaps_s, f'Senate (n={len(gaps_s):,})',      '#3d6b9e', 'rgba(61,107,158,0.18)',   '#2d4a7a', 3),
    (gaps_g, f'Governor (n={len(gaps_g):,})',    '#4a7ab0', 'rgba(74,122,176,0.18)',   '#33567e', 3),
    (gaps_h, f'House cycles (n={len(gaps_h)})',  '#5a8ec4', 'rgba(90,142,196,0.18)',   '#3d6b9e', 6),
    (gaps_p, f'Presidential (n={len(gaps_p)})',  NAVY,      'rgba(26,26,46,0.15)',      NAVY,      8),
]

fig = go.Figure()
for gaps, name, mcolor, fill, lcolor, ptsize in BOX_CONFIGS:
    log_gap = np.log10(gaps['gap'].clip(lower=1e-3))
    fig.add_trace(go.Violin(
        y=log_gap, name=name,
        points='all', jitter=0.3, pointpos=0,
        box_visible=True, meanline_visible=False,
        scalemode='width', width=0.8, bandwidth=0.18, spanmode='hard',
        marker=dict(color=mcolor, size=ptsize, opacity=0.45,
                    line=dict(color='white', width=0.6)),
        line=dict(color=lcolor, width=1.8),
        fillcolor=fill,
        hoveron='violins',
        hovertemplate=f'{name}<extra></extra>',
    ))

# Equal-attention reference line at log10(1)=0; label kept on the LEFT so it
# does not sit on top of the (rightmost) presidential distribution.
fig.add_hline(y=0, line_dash='dot', line_color='#555', line_width=2,
    annotation_text='Equal attention (1×)', annotation_position='top left',
    annotation_font=dict(size=11))
fig.update_yaxes(
    title='Mention ratio (winner ÷ loser, log scale)',
    tickvals=[-1, -0.523, 0, 0.477, 1, 1.477, 2, 2.477, 3],
    ticktext=['0.1×', '0.3×', '1×', '3×', '10×', '30×', '100×', '300×', '1000×'],
)
style(fig,
      'Figure 6. Distribution of attention gaps by race type',
      f'Google Ngrams · Log scale · Violin + box + all raw data · '
      f'Senate median {gaps_s["gap"].median():.0f}×, Governor median {gaps_g["gap"].median():.1f}×, '
      f'House median {gaps_h["gap"].median():.1f}×, Pres median {gaps_p["gap"].median():.1f}×',
      w=820, h=600)
fig.update_layout(showlegend=True)
save(fig, 'figures/fig6_gap_distribution.png')


# ── Figure 7: Multi-source × multi-level forest plot ─────────────────────────
# 4 panels: (A) H1 win rate [sources × levels as offset dots on numeric y]
#           (B) H2 Presidential, (C) H2 Senate, (D) H2 House
# Each H2 panel is a standard forest plot — one dot per source row — so CIs
# are unambiguous.

def level_entry(df, level, vote_col, is_pct=False):
    """Compute H1 and H2 for one source × level combination."""
    df = df[df['attention_leader_won'].notna()].copy()
    if len(df) < 2:
        return None
    k, n, rate, lo, hi, pval = h1_stats(df)
    if n >= 3:
        vs_vals = df[vote_col] / 100 if is_pct else df[vote_col]
        r2, _ = pearsonr(df['winner_mention_share'], vs_vals)
        r_lo2, r_hi2 = pearson_r_ci(r2, n)
        r, r_lo, r_hi = r2, r_lo2, r_hi2
    else:
        r, r_lo, r_hi = float('nan'), float('nan'), float('nan')
    return dict(level=level, n=n, rate=rate, lo=lo, hi=hi, pval=pval,
                r=r, r_lo=r_lo, r_hi=r_hi)

def house_level_entry():
    k, n, rate, lo, hi, pval = h1_stats(house_ng, 'attn_predicts_majority')
    r2, _, r_lo2, r_hi2 = h2_stats('d_attention_share', 'd_seat_share', house_ng)
    return dict(level='House', n=n, rate=rate, lo=lo, hi=hi, pval=pval,
                r=r2, r_lo=r_lo2, r_hi=r_hi2)

senate_trends = pd.read_csv('data/raw/trends/senate_mention_shares.csv')
senate_trends = senate_trends[senate_trends['attention_leader_won'].notna()].copy()
senate_wiki   = pd.read_csv('data/raw/wikipedia/senate_mention_shares.csv')

# pres sources use winner_vote_pct as percentage; senate CSVs have it as fraction already
SOURCE_LEVELS = [
    ('Google Ngrams',       SRC_PALETTE[1], [
        level_entry(pres_ng,      'Presidential', 'winner_vote_pct', is_pct=True),
        level_entry(senate_ng,    'Senate',        'winner_vote_pct', is_pct=False),
        level_entry(gov_ng,       'Governor',      'winner_vote_pct', is_pct=False),
        house_level_entry(),
    ]),
    ('Google Trends',       SRC_PALETTE[2], [
        level_entry(trends_pres,   'Presidential', 'winner_vote_pct', is_pct=True),
        level_entry(senate_trends, 'Senate',        'winner_vote_pct', is_pct=False),
    ]),
    ('Wikipedia Pageviews', SRC_PALETTE[3], [
        level_entry(wiki_pres,   'Presidential', 'winner_vote_pct', is_pct=True),
        level_entry(senate_wiki, 'Senate',        'winner_vote_pct', is_pct=False),
    ]),
    ('GDELT News',  SRC_PALETTE[0], [level_entry(gdelt_pres,  'Presidential', 'winner_vote_pct', is_pct=True)]),
    ('Reddit',      SRC_PALETTE[5], [level_entry(reddit_pres, 'Presidential', 'winner_vote_pct', is_pct=True)]),
]
# ── H1 forest: sources on a numeric y-axis, race levels offset within each row ─
# Ngrams (4 levels, large n) sits at the top; single-level sources below.
# Markers for each source are centred on its row and spread by how many levels
# it actually has, so a lone presidential dot is never left floating.
SOURCE_ORDER = [s[0] for s in SOURCE_LEVELS]
src_y = {name: len(SOURCE_ORDER) - 1 - i for i, name in enumerate(SOURCE_ORDER)}  # Ngrams on top
LEVEL_COLORS  = {'Presidential': NAVY,  'Senate': '#3d6b9e', 'Governor': '#5a8ec4', 'House': '#7fb3d9'}
LEVEL_SYMBOLS = {'Presidential': 'circle', 'Senate': 'square', 'Governor': 'triangle-up', 'House': 'diamond'}
LEVEL_RANK    = {'Presidential': 0, 'Senate': 1, 'Governor': 2, 'House': 3}

fig = go.Figure()
legend_added = set()
for sname, scolor, levels in SOURCE_LEVELS:
    present = [e for e in levels if e is not None]
    present.sort(key=lambda e: LEVEL_RANK[e['level']])
    m = len(present)
    # symmetric vertical offsets around the row centre (top level highest)
    span = 0.30 if m > 1 else 0.0
    offs = [span * (((m - 1) / 2) - j) for j in range(m)]
    for entry, off in zip(present, offs):
        lvl = entry['level']
        ypos = src_y[sname] + off
        small_n = entry['n'] < 5          # tiny samples: keep but de-emphasise
        show_leg = lvl not in legend_added
        legend_added.add(lvl)
        fig.add_trace(go.Scatter(
            x=[entry['rate']], y=[ypos],
            error_x=dict(type='data', symmetric=False,
                         array=[entry['hi'] - entry['rate']],
                         arrayminus=[entry['rate'] - entry['lo']],
                         thickness=1 if small_n else 1.6,
                         width=4, color='#ccc' if small_n else None),
            mode='markers',
            marker=dict(color=LEVEL_COLORS[lvl], size=11,
                        symbol=LEVEL_SYMBOLS[lvl],
                        opacity=0.5 if small_n else 1.0,
                        line=dict(color='white', width=1.5)),
            name=lvl, legendgroup=lvl, showlegend=show_leg,
            hovertemplate=f'{sname} ({lvl})<br>Win rate: %{{x:.0%}}<br>n={entry["n"]:,}<extra></extra>',
        ))
        # n label to the right of each point so coverage/precision is explicit
        fig.add_annotation(
            x=max(entry['hi'], entry['rate']) + 0.015, y=ypos,
            text=f'<span style="font-size:9px;color:#999">n={entry["n"]:,}</span>',
            showarrow=False, xanchor='left', yanchor='middle')

fig.add_vline(x=0.5, line_dash='dot', line_color='#999', line_width=1)
fig.update_xaxes(tickformat='.0%', range=[0, 1.18], title_text='H1 win rate', showgrid=True, gridcolor=GRID)
fig.update_yaxes(
    tickmode='array',
    tickvals=list(src_y.values()),
    ticktext=list(src_y.keys()),
    range=[-0.7, max(src_y.values()) + 0.7],
    showgrid=False,
)
style(fig,
      'Figure 7. Attention leader win rate across data sources and race levels',
      'H1 win rate by source · ● Presidential  ■ Senate  ▲ Governor  ◆ House · '
      'faded markers = small samples (n&lt;5) · GDELT TV &amp; MediaCloud not yet collected',
      w=820, h=540)
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


# ── Figure S2: Window-length sensitivity (Ngrams presidential) ────────────────
# Google Ngrams is annual book data, so the pre-registered "window" robustness
# check (Step 6) is implemented as the number of years averaged ending on the
# election year: 1-year (election year only), 2-year (primary measure), 3-year.
# Data from scripts/collect_ngrams_windows.py.
windows = pd.read_csv('data/raw/ngrams/presidential_windows.csv')
WIN_LABELS = {1: '1 year\n(election year)', 2: '2 years\n(primary)', 3: '3 years'}
win_order = [1, 2, 3]
w_rate, w_hi, w_lo, w_r, w_rhi, w_rlo, w_n = [], [], [], [], [], [], []
for ny in win_order:
    sub = windows[windows['window_years'] == ny]
    k, n, rate, lo, hi, _ = h1_stats(sub)
    r2, _, rlo2, rhi2 = h2_stats('winner_mention_share', 'winner_vote_pct', sub)
    w_rate.append(rate); w_hi.append(hi - rate); w_lo.append(rate - lo)
    w_r.append(r2); w_rhi.append(rhi2 - r2); w_rlo.append(r2 - rlo2); w_n.append(n)

win_labs = [WIN_LABELS[w].replace('\n', '<br>') for w in win_order]
fig = make_subplots(rows=1, cols=2,
    subplot_titles=['(A) H1 win rate by window', '(B) H2 correlation by window'],
    horizontal_spacing=0.16)

# Panel A — H1 win rate
fig.add_trace(go.Scatter(
    x=win_labs, y=w_rate, mode='markers+lines',
    error_y=dict(type='data', array=w_hi, arrayminus=w_lo, thickness=2, width=6),
    marker=dict(color=NAVY, size=14, line=dict(color='white', width=2)),
    line=dict(color='#bbb', dash='dot', width=1.5), showlegend=False,
), row=1, col=1)
for xl, r, hi in zip(win_labs, w_rate, w_hi):
    fig.add_annotation(x=xl, y=r + hi + 0.04, text=f'<b>{r:.0%}</b>',
        showarrow=False, font=dict(size=12, color=NAVY),
        yanchor='bottom', xanchor='center', row=1, col=1)
fig.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5,
    annotation_text='Chance', annotation_position='right', row=1, col=1)
fig.update_yaxes(range=[0, 1.2], tickformat='.0%', title='Attention leader win rate',
                 row=1, col=1)

# Panel B — H2 Pearson r
fig.add_trace(go.Scatter(
    x=win_labs, y=w_r, mode='markers+lines',
    error_y=dict(type='data', array=w_rhi, arrayminus=w_rlo, thickness=2, width=6),
    marker=dict(color='#3d6b9e', size=14, line=dict(color='white', width=2)),
    line=dict(color='#bbb', dash='dot', width=1.5), showlegend=False,
), row=1, col=2)
for xl, r in zip(win_labs, w_r):
    fig.add_annotation(x=xl, y=r + 0.06, text=f'<b>r = {r:.2f}</b>',
        showarrow=False, font=dict(size=12, color='#2d4a7a'),
        yanchor='bottom', xanchor='center', row=1, col=2)
fig.add_hline(y=0, line_dash='dot', line_color='#999', line_width=1.5, row=1, col=2)
fig.update_yaxes(range=[-0.2, 0.8], title='Pearson r (mention vs. vote share)', row=1, col=2)
fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
style(fig,
      'Figure S2. Sensitivity: does the measurement window length matter?',
      f'Google Ngrams presidential · n={w_n[0]} elections · '
      f'H1 is identical across 1–3 year windows; H2 is stable (r = {min(w_r):.2f}–{max(w_r):.2f})',
      w=1000, h=520)
save(fig, 'figures/figS2_window_sensitivity.png')


# ══════════════════════════════════════════════════════════════════════════════
# INTERNATIONAL FIGURES (Google Ngrams — UK, Australia, Canada, New Zealand)
# ══════════════════════════════════════════════════════════════════════════════

uk_ng  = pd.read_csv('data/raw/ngrams/uk_mention_shares.csv')
aus_ng = pd.read_csv('data/raw/ngrams/australia_mention_shares.csv')
can_ng = pd.read_csv('data/raw/ngrams/canada_mention_shares.csv')
nz_ng  = pd.read_csv('data/raw/ngrams/new_zealand_mention_shares.csv')

# (display name, dataframe, accent colour, span)
INTL = [
    ('United Kingdom',  uk_ng,  '#8B0000', '1945–2019'),
    ('Australia',       aus_ng, '#00614A', '1949–2019'),
    ('Canada',          can_ng, '#C8102E', '1945–2019'),
    ('New Zealand',     nz_ng,  '#1f3a93', '1946–2014'),
]
GRID_RC = [(1, 1), (1, 2), (2, 1), (2, 2)]

# Pooled H1 across all four countries
pool_k = sum(int(d['attention_leader_won'].sum()) for _, d, _, _ in INTL)
pool_n = sum(len(d) for _, d, _, _ in INTL)
pool_p = binomtest(pool_k, pool_n, 0.5, alternative='two-sided').pvalue
pool_p_s = 'p < 0.001' if pool_p < 0.001 else f'p = {pool_p:.3f}'

# ── Figure I1: H1 win rate — four countries (2×2) ─────────────────────────────
fig = make_subplots(rows=2, cols=2,
    subplot_titles=[f'{n} (n={len(d)})' for n, d, _, _ in INTL],
    horizontal_spacing=0.13, vertical_spacing=0.16)

for (country, df, color, span), (rr, cc) in zip(INTL, GRID_RC):
    k, n, rate, lo, hi, pval = h1_stats(df)
    err_lo, err_hi = rate - lo, hi - rate
    fig.add_trace(go.Bar(
        x=[country], y=[rate],
        error_y=dict(type='data', array=[err_hi], arrayminus=[err_lo], thickness=2, width=10),
        marker_color=color, showlegend=False, width=0.5,
        hovertemplate=f'{rate:.0%}<br>n={n}<br>p={pval:.3f}<extra></extra>',
    ), row=rr, col=cc)
    ps = 'p < 0.001' if pval < 0.001 else f'p = {pval:.3f}'
    fig.add_annotation(x=country, y=rate + err_hi + 0.05,
        text=f'<b>{rate:.0%}</b> · <span style="font-size:10px">{ps}</span>',
        showarrow=False, font=dict(size=12, color=color),
        yanchor='bottom', xanchor='center', row=rr, col=cc)
    fig.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5, row=rr, col=cc)
    fig.update_yaxes(tickformat='.0%', range=[0, 1.18], title_text='Win rate', row=rr, col=cc)
    fig.update_xaxes(showticklabels=False, row=rr, col=cc)

style(fig, 'Figure I1. International replication: does the attention leader win?',
      f'Google Ngrams · 4 Anglosphere democracies · '
      f'pooled {pool_k}/{pool_n} = {pool_k/pool_n:.0%} ({pool_p_s})',
      w=920, h=720)
save(fig, 'figures/figI1_international_h1.png')


# ── Figure I2: H2 scatter — four countries (2×2) ──────────────────────────────
# BOTH candidates per election (winner + loser), like the US H2 (Fig 2).
# Plotting winners only conditions on winning — a selection/collider effect that
# spuriously flips the correlation negative (famous losing leaders such as
# Churchill 1945 sit at high mention share but get excluded). With both
# candidates the relationship is positive in every country, matching the US.
# Centre-right bloc (Conservative/Liberal/National) = blue; centre-left = red.
INTL_BLOC = {'Conservative': 'right', 'Liberal': 'right', 'PC': 'right', 'National': 'right',
             'Labour': 'left', 'Labor': 'left'}
BLOC_CLR  = {'right': '#1560BD', 'left': '#E4003B'}
BLOC_NAME = {'right': 'Centre-right (Con/Lib/Nat)', 'left': 'Centre-left (Lab)'}

def intl_both(df):
    w = df[['year', 'winner_party', 'winner_mention_share', 'winner_vote_pct', 'winner_name']].rename(
        columns={'winner_party': 'party', 'winner_mention_share': 'ms',
                 'winner_vote_pct': 'vs', 'winner_name': 'name'})
    l = df[['year', 'loser_party', 'loser_mention_share', 'loser_vote_pct', 'loser_name']].rename(
        columns={'loser_party': 'party', 'loser_mention_share': 'ms',
                 'loser_vote_pct': 'vs', 'loser_name': 'name'})
    return pd.concat([w, l], ignore_index=True)

fig = make_subplots(rows=2, cols=2,
    subplot_titles=[f'{n}' for n, _, _, _ in INTL],
    horizontal_spacing=0.12, vertical_spacing=0.14)

intl_r, pooled_ms, pooled_vs = {}, [], []
legend_added = set()
for (country, df, _, _), (rr, cc) in zip(INTL, GRID_RC):
    sc = intl_both(df)
    sc = sc.assign(bloc=sc['party'].map(INTL_BLOC).fillna('left'))
    r_i, p_i = pearsonr(sc['ms'], sc['vs'])
    intl_r[country] = (r_i, p_i)
    pooled_ms += list(sc['ms']); pooled_vs += list(sc['vs'])
    for bloc in ['left', 'right']:
        sub = sc[sc['bloc'] == bloc]
        if sub.empty:
            continue
        show = bloc not in legend_added
        legend_added.add(bloc)
        fig.add_trace(go.Scatter(
            x=sub['ms'], y=sub['vs'], mode='markers',
            marker=dict(color=BLOC_CLR[bloc], size=8, line=dict(color='white', width=1.1)),
            name=BLOC_NAME[bloc], legendgroup=bloc, showlegend=show,
            text=sub['name'].str.split().str[-1] + " '" + sub['year'].astype(str).str[-2:],
            hovertemplate='%{text}<br>Mention: %{x:.1%}<br>Vote: %{y:.1f}%<extra></extra>',
        ), row=rr, col=cc)
    m_i, b_i = np.polyfit(sc['ms'], sc['vs'], 1)
    xr = np.linspace(sc['ms'].min() - 0.02, sc['ms'].max() + 0.02, 200)
    fig.add_trace(go.Scatter(x=xr, y=m_i * xr + b_i, mode='lines',
        line=dict(color='#555', dash='dash', width=1.5), showlegend=False), row=rr, col=cc)
    ols_label(fig, r_i, sc['ms'].values, sc['vs'].values, row=rr, col=cc)
    fig.update_xaxes(tickformat='.0%', title_text='Mention share', row=rr, col=cc)
    fig.update_yaxes(title_text='Vote share (%)', row=rr, col=cc)

pooled_r, pooled_rp = pearsonr(pooled_ms, pooled_vs)
fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
style(fig, 'Figure I2. Mention share vs. vote share — four countries',
      f'Google Ngrams · both major candidates per election · '
      f'positive in every country · pooled r = {pooled_r:.2f} (n={len(pooled_ms)})',
      w=1000, h=760)
save(fig, 'figures/figI2_international_h2.png')


# ── Figure I3: UK House of Commons — party attention vs. seat share ───────────
# Down-ballot UK analogue of the US House figure (Fig 5): does the major party
# whose leader dominates book coverage win the larger share of Commons seats?
uk_commons = pd.read_csv('data/raw/ngrams/uk_commons.csv')
CON_CLR, LAB_CLR = '#0087DC', '#E4003B'   # UK party colours (Tory blue, Labour red)
r_uc, p_uc = pearsonr(uk_commons['con_attention_share'], uk_commons['con_seat_share'])
k_uc = int(uk_commons['attn_predicts_majority'].sum()); n_uc = len(uk_commons)

fig = go.Figure()
fig.add_shape(type='rect', x0=0.5, x1=1.0, y0=0.5, y1=0.75,
    fillcolor='rgba(0,135,220,0.05)', line_width=0)
fig.add_shape(type='rect', x0=0.0, x1=0.5, y0=0.25, y1=0.5,
    fillcolor='rgba(228,0,59,0.05)', line_width=0)
for maj, color, label in [('Con', CON_CLR, 'Conservative majority'),
                          ('Lab', LAB_CLR, 'Labour majority')]:
    sub = uk_commons[uk_commons['majority'] == maj]
    fig.add_trace(go.Scatter(
        x=sub['con_attention_share'], y=sub['con_seat_share'],
        mode='markers+text', text=sub['year'].astype(str),
        textposition='top center', textfont=dict(size=9),
        marker=dict(color=color, size=10, line=dict(color='white', width=1.5)),
        name=label,
        customdata=sub[['con_seats', 'lab_seats']].values,
        hovertemplate='%{text}<br>Con attention: %{x:.1%}<br>'
                      'Con seats: %{customdata[0]} · Lab seats: %{customdata[1]}<extra></extra>',
    ))
m_uc, b_uc = np.polyfit(uk_commons['con_attention_share'], uk_commons['con_seat_share'], 1)
xr_uc = np.linspace(uk_commons['con_attention_share'].min() - 0.02,
                    uk_commons['con_attention_share'].max() + 0.02, 200)
fig.add_trace(go.Scatter(x=xr_uc, y=m_uc * xr_uc + b_uc, mode='lines',
    line=dict(color='#666', dash='dash', width=1.5), showlegend=False))
ols_label(fig, r_uc, uk_commons['con_attention_share'].values,
          uk_commons['con_seat_share'].values)
fig.add_vline(x=0.5, line_dash='dot', line_color='#aaa', line_width=1.2)
fig.add_hline(y=0.5, line_dash='dot', line_color='#aaa', line_width=1.2)
p_uc_s = 'p < 0.001' if p_uc < 0.001 else f'p = {p_uc:.3f}'
fig.update_xaxes(tickformat='.0%', title='Conservative leader attention share', range=[0, 1])
fig.update_yaxes(tickformat='.0%', title='Conservative seat share (of Con+Lab)', range=[0.2, 0.8])
style(fig,
      f'Figure I3. UK House of Commons: does the party with the more-covered leader win more seats? (r = {r_uc:.2f})',
      f'Google Ngrams · {n_uc} UK general elections 1945–2019 · {p_uc_s} · '
      f'attention-leading party won the seat count in {k_uc}/{n_uc} elections',
      w=860, h=600)
save(fig, 'figures/figI3_uk_commons.png')


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
