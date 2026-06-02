"""
Synthetic data and figures for 'Is Attention All You Need (to Win Elections)?'
All candidate names are Zorblaxian placeholders.
D/R colors preserved for real data substitution.

Data model: logit-normal mention shares (no hard clipping) +
margin-of-victory vote shares (no boundary spikes).
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

rng = np.random.default_rng(42)
Path('data/processed').mkdir(exist_ok=True)
Path('figures').mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
FONT      = 'Georgia, serif'
BG, GRID  = '#FAFAFA', '#E8E8E8'
D_COLOR   = '#1060C8'
R_COLOR   = '#C82010'
NAVY      = '#1a1a2e'
GRAY      = '#CCCCCC'
UPSET     = '#E05050'
SRC_PALETTE = ['#1a1a2e','#2d4a7a','#3d6b9e','#5a8ec4','#7fb3d9','#a8cceb','#c5dff0']

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
    """Place % labels above CI error bar tops — no overlap guaranteed."""
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
    """95% CI for Pearson r via Fisher z-transform."""
    if abs(r) >= 0.999 or n <= 3:
        return -1.0, 1.0
    z, se = np.arctanh(r), 1.0 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Presidential (Zorblaxian names, real vote shares) ─────────────────────────

PRES = [
    # year  winner               loser                wp   lp   inc  wv      lv
    (1960,'Zorbax Krenvel',    'Thrumble Quist',    'D','R', 0, 0.4972,0.4955),
    (1964,'Glornak Venturis',  'Frixel Stombard',   'D','R', 1, 0.6131,0.3858),
    (1968,'Quist II',          'Yarvok Drensch',    'R','D', 0, 0.4342,0.4298),
    (1972,'Quist II',          'Blexor Marvine',    'R','D', 1, 0.6077,0.3780),
    (1976,'Wumple Crondix',    'Garfax Trellen',    'D','R', 0, 0.5008,0.4796),
    (1980,'Splorg Vanthorn',   'Wumple Crondix',    'R','D', 0, 0.5075,0.4107),
    (1984,'Splorg Vanthorn',   'Zeldrick Munvale',  'R','D', 1, 0.5849,0.4083),
    (1988,'Borrax Grundle',    'Quelvin Droskip',   'R','D', 0, 0.5337,0.4585),
    (1992,'Mirval Prondex',    'Borrax Grundle',    'D','R', 0, 0.4300,0.3739),
    (1996,'Mirval Prondex',    'Flubb Ranstorm',    'D','R', 1, 0.4927,0.4072),
    (2000,'Xenthox Quivel',    'Drembix Faltor',    'R','D', 0, 0.4763,0.4838),
    (2004,'Xenthox Quivel',    'Clorvis Stemple',   'R','D', 1, 0.5073,0.4823),
    (2008,'Zumbrix Halvore',   'Skrendle Mactorvish','D','R',0, 0.5259,0.4573),
    (2012,'Zumbrix Halvore',   'Prumdex Vorthalack','D','R', 1, 0.5106,0.4706),
    (2016,'Grumvox Spralton',  'Clindivar Norquess','R','D', 0, 0.4609,0.4818),
    (2020,'Blengle Arvondex',  'Grumvox Spralton',  'D','R', 0, 0.5135,0.4685),
    (2024,'Grumvox Spralton',  'Harrinda Vex',      'R','D', 0, 0.4977,0.4823),
]

pres_rows = []
for yr, wn, ln, wp, lp, inc, wv, lv in PRES:
    # Use the real vote margin as the latent "candidate quality" signal.
    # Winner gets positive logit signal, loser negative — so mention share
    # correlates positively with vote share by construction.
    real_margin = wv - lv          # e.g. +0.03 for close, +0.15 for blowout
    signal = real_margin * 3.5     # scale to logit space
    ms_w = float(expit(signal + rng.normal(0, 0.55)))
    ms_l = 1.0 - ms_w
    attn = int(ms_w > ms_l)
    pres_rows += [
        dict(year=yr,candidate=wn,party=wp,mention_share=ms_w,
             popular_vote_share=wv,popular_vote_winner=1,
             attention_leader=attn,incumbent_running=inc),
        dict(year=yr,candidate=ln,party=lp,mention_share=ms_l,
             popular_vote_share=lv,popular_vote_winner=0,
             attention_leader=1-attn,incumbent_running=0),
    ]
df_pres = pd.DataFrame(pres_rows)

# ── Senate (logit-normal mentions + margin-of-victory votes) ──────────────────
# No hard clipping anywhere — expit tapers naturally, margin draws from half-normal.

ZORB_F = ['Zorbax','Thrumble','Glornak','Frixel','Yarvok','Blexor','Wumple',
          'Garfax','Splorg','Zeldrick','Borrax','Quelvin','Mirval','Flubb',
          'Xenthox','Drembix','Clorvis','Zumbrix','Skrendle','Prumdex',
          'Grumvox','Clindivar','Blengle','Harrinda','Thumlek','Plorbex']
ZORB_L = ['Vrenlik','Quorvast','Drensch','Marvine','Stombard','Krenvel',
          'Venturis','Crondix','Grundle','Droskip','Ranstorm','Stemple',
          'Halvore','Norquess','Blendrix','Gorvalak','Threxis','Smundrel',
          'Quarvox','Frundax','Glorbik','Wrendle','Thumlek','Vex','Prondex']
def zorb(): return f'{rng.choice(ZORB_F)} {rng.choice(ZORB_L)}'

senate_rows = []
for cycle in range(1960, 2025, 2):
    for _ in range(33):
        # Shared latent quality drives both attention and vote share.
        # Half-normal: winner always has positive quality advantage; noise can flip attention.
        quality = abs(rng.normal(0, 0.6))   # winner's advantage in logit space
        ms_w = float(expit(quality + rng.normal(0, 0.40)))
        ms_l = 1.0 - ms_w
        # Vote margin: half-normal shifted by quality signal
        margin = abs(rng.normal(0.04 + 0.08 * quality, 0.04))
        margin = np.clip(margin, 0.001, 0.49)
        vs_w, vs_l = 0.5 + margin, 0.5 - margin
        pw, pl = ('D','R') if rng.random() < 0.5 else ('R','D')
        attn = int(ms_w > ms_l)
        senate_rows += [
            dict(year=cycle,candidate=zorb(),party=pw,mention_share=ms_w,
                 popular_vote_share=vs_w,popular_vote_winner=1,attention_leader=attn),
            dict(year=cycle,candidate=zorb(),party=pl,mention_share=ms_l,
                 popular_vote_share=vs_l,popular_vote_winner=0,attention_leader=1-attn),
        ]
df_senate = pd.DataFrame(senate_rows)

# ── House aggregate ────────────────────────────────────────────────────────────
HOUSE_D = {1960:263,1962:258,1964:295,1966:248,1968:243,1970:255,1972:242,
           1974:291,1976:292,1978:277,1980:243,1982:269,1984:253,1986:258,
           1988:260,1990:267,1992:258,1994:204,1996:207,1998:211,2000:212,
           2002:204,2004:202,2006:233,2008:257,2010:193,2012:201,2014:188,
           2016:194,2018:235,2020:222,2022:213,2024:215}
house_rows = []
for cy, ds in HOUSE_D.items():
    dss = ds / 435
    da = float(np.clip(rng.normal(0.48 + 0.18*(dss-0.5), 0.04), 0.35, 0.65))
    house_rows.append(dict(year=cy,d_attention_share=da,r_attention_share=1-da,
        d_seat_share=dss,r_seat_share=1-dss,d_seats=ds,
        majority='D' if ds>217 else 'R',
        attention_majority='D' if da>0.5 else 'R'))
df_house = pd.DataFrame(house_rows)
df_house['attn_correct'] = (df_house['majority']==df_house['attention_majority']).astype(int)

# ── Multi-source data ──────────────────────────────────────────────────────────
# Each source measures the same latent phenomenon (who is politically salient)
# but via a different channel, with its own noise characteristics.

SOURCES = [
    # name                start  spread  color             description
    ('GDELT News',        1979,  0.95,  SRC_PALETTE[0], '~800 global news outlets'),
    ('Google Ngrams',     1960,  0.70,  SRC_PALETTE[1], 'Book corpus frequency'),
    ('Google Trends',     2004,  1.05,  SRC_PALETTE[2], 'Search query volume'),
    ('Wikipedia Views',   2008,  1.00,  SRC_PALETTE[3], 'Article pageview counts'),
    ('GDELT TV',          2009,  0.85,  SRC_PALETTE[4], 'Broadcast transcripts'),
    ('Reddit Posts',      2007,  1.15,  SRC_PALETTE[5], 'Political subreddit posts'),
    ('MediaCloud',        2010,  0.90,  SRC_PALETTE[6], 'Academic news index'),
]

PRES_FOR_SRC = [(yr, wv, lv) for yr,_,_,_,_,_,wv,lv in PRES]

source_stats = []
for sname, start, spread, color, desc in SOURCES:
    elig = [(yr,wv,lv) for yr,wv,lv in PRES_FOR_SRC if yr >= start]
    ms_all, vs_all, h1_wins = [], [], 0
    for yr, wv, lv in elig:
        ms_w = float(expit(rng.normal(0, spread)))
        ms_l = 1.0 - ms_w
        h1_wins += int(ms_w > ms_l)
        ms_all += [ms_w, ms_l]; vs_all += [wv, lv]
    n_e = len(elig)
    bi = stats.binomtest(h1_wins, n=n_e, p=0.5, alternative='two-sided')
    ci1 = bi.proportion_ci()
    r2, _ = stats.pearsonr(ms_all, vs_all)
    r2_lo, r2_hi = pearson_r_ci(r2, len(ms_all))
    source_stats.append(dict(
        name=sname, start=start, n=n_e, color=color, desc=desc,
        h1=h1_wins/n_e, h1_lo=ci1.low, h1_hi=ci1.high, h1_p=bi.pvalue,
        r2=r2, r2_lo=r2_lo, r2_hi=r2_hi,
    ))
src_df = pd.DataFrame(source_stats).sort_values('h1', ascending=False)

# ── Compute shared stats ───────────────────────────────────────────────────────
per_pres = (df_pres.groupby('year')
    .apply(lambda g: int((g.loc[g['attention_leader']==1,'popular_vote_winner']==1).any()))
    .reset_index(name='won'))
k_p, n_p = per_pres['won'].sum(), len(per_pres)
bi_p = stats.binomtest(k_p, n=n_p, p=0.5); ci_p = bi_p.proportion_ci()

sp2 = df_senate.copy(); sp2['rid'] = sp2.groupby('year').cumcount()//2
aw_s = sp2.groupby(['year','rid']).apply(
    lambda g: int((g.loc[g['attention_leader']==1,'popular_vote_winner']==1).any()))
k_s, n_s = aw_s.sum(), len(aw_s)
bi_s = stats.binomtest(k_s, n=n_s, p=0.5); ci_s = bi_s.proportion_ci()

k_h, n_h = df_house['attn_correct'].sum(), len(df_house)
bi_h = stats.binomtest(k_h, n=n_h, p=0.5); ci_h = bi_h.proportion_ci()

r_p, _ = stats.pearsonr(df_pres['mention_share'], df_pres['popular_vote_share'])
r_s, _ = stats.pearsonr(df_senate['mention_share'], df_senate['popular_vote_share'])

w_p = df_pres[df_pres['popular_vote_winner']==1][['year','candidate','mention_share','popular_vote_share']]
l_p = df_pres[df_pres['popular_vote_winner']==0][['year','candidate','mention_share','popular_vote_share']]
h3 = w_p.merge(l_p, on='year', suffixes=('_w','_l'))
h3['gap']    = h3['mention_share_w'] / h3['mention_share_l']
h3['margin'] = h3['popular_vote_share_w'] - h3['popular_vote_share_l']
h3['upset']  = h3['gap'] < 1

print(f'Pres H1: {k_p}/{n_p} ({k_p/n_p:.0%})  Senate H1: {k_s}/{n_s} ({k_s/n_s:.0%})')
print(f'Pres r={r_p:.2f}  Senate r={r_s:.2f}')

# ── Gap distribution helpers ───────────────────────────────────────────────────
def race_gaps(df):
    df2 = df.copy(); df2['rid'] = df2.groupby('year').cumcount()//2
    rows = []
    for (yr,rid), g in df2.groupby(['year','rid']):
        w = g[g['popular_vote_winner']==1]; l = g[g['popular_vote_winner']==0]
        if len(w) and len(l):
            rows.append(dict(year=yr, gap=w['mention_share'].values[0]/l['mention_share'].values[0]))
    return pd.DataFrame(rows)

gaps_p = race_gaps(df_pres)
gaps_s = race_gaps(df_senate)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print('\nGenerating figures...')

# ── Figure 1: H1 presidential win rate ───────────────────────────────────────
p1s = f'p = {bi_p.pvalue:.3f}' if bi_p.pvalue >= 0.001 else 'p < 0.001'
fig = go.Figure()
fig.add_trace(go.Bar(
    x=['Attention leader','Chance (50%)'], y=[k_p/n_p, 0.5],
    error_y=dict(type='data', array=[ci_p.high-k_p/n_p, 0],
                 arrayminus=[k_p/n_p-ci_p.low, 0], thickness=2),
    marker_color=[NAVY, GRAY], marker_line_width=0, width=0.4,
))
pct_above_ci(fig, [k_p/n_p, 0.5], [ci_p.high-k_p/n_p, 0],
             ['Attention leader','Chance (50%)'])
fig.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5)
fig.update_yaxes(range=[0,1.18], tickformat='.0%', title='Win rate')
style(fig, f'Figure 1. Attention leader wins popular vote: {k_p}/{n_p} elections ({k_p/n_p:.0%})',
      f'Binomial test vs. 50% chance — {p1s} — 95% CI [{ci_p.low:.0%}, {ci_p.high:.0%}] — <i>synthetic data</i>',
      w=560, h=500)
save(fig, 'figures/fig1_h1_win_rate.png')


# ── Figure 2: H2 scatter — 2 panels (Presidential + Senate) ─────────────────
fig = make_subplots(rows=1, cols=2,
    subplot_titles=['(A) Presidential elections (n=34)', f'(B) Senate races (n={len(df_senate):,})'],
    horizontal_spacing=0.12)

# Panel A — Presidential
m_p, b_p = np.polyfit(df_pres['mention_share'], df_pres['popular_vote_share'], 1)
xr_p = np.linspace(df_pres['mention_share'].min()-0.02, df_pres['mention_share'].max()+0.02, 200)
for party, color, label in [('D',D_COLOR,'Democrat'),('R',R_COLOR,'Republican')]:
    sub = df_pres[df_pres['party']==party]
    fig.add_trace(go.Scatter(
        x=sub['mention_share'], y=sub['popular_vote_share'], mode='markers',
        customdata=sub['candidate']+" '"+sub['year'].astype(str).str[-2:],
        hovertemplate='%{customdata}<br>Mention: %{x:.1%} · Vote: %{y:.1%}<extra></extra>',
        marker=dict(color=color,size=10,line=dict(color='white',width=1.5)),
        name=label, legendgroup=label,
    ), row=1, col=1)
notable = df_pres[(df_pres['mention_share']>0.67)|(df_pres['mention_share']<0.32)|
                  (df_pres['popular_vote_share']>0.58)|(df_pres['popular_vote_share']<0.40)]
fig.add_trace(go.Scatter(
    x=notable['mention_share'], y=notable['popular_vote_share'], mode='text',
    text=notable['candidate'].str.split().str[0]+" '"+notable['year'].astype(str).str[-2:],
    textposition='top center', textfont=dict(size=8,color='#444'),
    showlegend=False, hoverinfo='skip',
), row=1, col=1)
fig.add_trace(go.Scatter(x=xr_p, y=m_p*xr_p+b_p, mode='lines',
    line=dict(color='#555',dash='dash',width=1.5), name=f'OLS (r={r_p:.2f})',
    legendgroup='ols_p',
), row=1, col=1)

# Panel B — Senate
m_s, b_s = np.polyfit(df_senate['mention_share'], df_senate['popular_vote_share'], 1)
xr_s = np.linspace(df_senate['mention_share'].min()-0.01, df_senate['mention_share'].max()+0.01, 200)
for party, color, label in [('D',D_COLOR,'Democrat'),('R',R_COLOR,'Republican')]:
    sub = df_senate[df_senate['party']==party]
    fig.add_trace(go.Scatter(
        x=sub['mention_share'], y=sub['popular_vote_share'], mode='markers',
        marker=dict(color=color,size=4,opacity=0.35,line=dict(width=0)),
        name=label, legendgroup=label, showlegend=False,
    ), row=1, col=2)
fig.add_trace(go.Scatter(x=xr_s, y=m_s*xr_s+b_s, mode='lines',
    line=dict(color='#222',dash='dash',width=2), name=f'OLS (r={r_s:.2f})',
    legendgroup='ols_s',
), row=1, col=2)

fig.update_xaxes(tickformat='.0%', title_text='Mention share (12 months pre-election)', row=1, col=1)
fig.update_xaxes(tickformat='.0%', title_text='Mention share (12 months pre-election)', row=1, col=2)
fig.update_yaxes(tickformat='.0%', title_text='Popular vote share', row=1, col=1)
fig.update_yaxes(tickformat='.0%', title_text='Vote share', row=1, col=2)
fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
style(fig, 'Figure 2. Mention share vs. vote share — presidential and Senate',
      f'Pres r = {r_p:.2f} · Senate r = {r_s:.2f} · <i>synthetic data</i>', w=1100, h=540)
save(fig, 'figures/fig2_h2_scatter.png')


# ── Figure 3: 2 panels — gap vs margin + timeline ────────────────────────────
h3s = h3.sort_values('year')
r3, _ = stats.pearsonr(h3['gap'], h3['margin'])
m3, b3 = np.polyfit(h3['gap'], h3['margin'], 1)
xr3 = np.linspace(h3['gap'].min()-0.05, h3['gap'].max()+0.1, 200)

fig = make_subplots(rows=1, cols=2,
    subplot_titles=['(A) Attention gap vs. vote margin', '(B) Attention ratio by election year'],
    horizontal_spacing=0.12)

# Panel A
for upset, color, label in [(False,NAVY,'Attention → winner'),(True,UPSET,'Attention upset')]:
    sub = h3[h3['upset']==upset]
    fig.add_trace(go.Scatter(
        x=sub['gap'], y=sub['margin'], mode='markers+text',
        text=sub['year'].astype(str), textposition='top center', textfont=dict(size=8),
        marker=dict(color=color,size=10,line=dict(color='white',width=1.5)), name=label,
    ), row=1, col=1)
fig.add_trace(go.Scatter(x=xr3, y=m3*xr3+b3, mode='lines',
    line=dict(color='#888',dash='dash',width=1.5), name=f'OLS (r={r3:.2f})', showlegend=True,
), row=1, col=1)
fig.add_vline(x=1, line_dash='dot', line_color='#aaa', line_width=1.5, row=1, col=1)
fig.add_hline(y=0, line_dash='dot', line_color='#aaa', line_width=1, row=1, col=1)

# Panel B — timeline
bar_colors = [UPSET if u else NAVY for u in h3s['upset']]
fig.add_trace(go.Bar(
    x=h3s['year'], y=h3s['gap'],
    text=[f"{r['candidate_w'].split()[0]} {r['gap']:.2f}×" for _,r in h3s.iterrows()],
    textposition='outside', textfont=dict(size=8),
    marker_color=bar_colors, marker_line_width=0, width=3,
    showlegend=False,
), row=1, col=2)
fig.add_hline(y=1, line_dash='dot', line_color='#aaa', line_width=1.5,
    annotation_text='Equal (1×)', annotation_position='top left', row=1, col=2)

fig.update_xaxes(title_text='Mention ratio (winner ÷ loser)', row=1, col=1)
fig.update_yaxes(tickformat='.1%', title_text='Popular vote margin', row=1, col=1)
fig.update_xaxes(title_text='Election year', tickmode='array',
    tickvals=h3s['year'].tolist(), tickangle=-55, tickfont=dict(size=10), row=1, col=2)
fig.update_yaxes(title_text='Attention ratio (winner ÷ loser)',
    range=[0, h3s['gap'].max()*1.25], row=1, col=2)
fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
style(fig, 'Figure 3. Attention gap: does bigger coverage lead predict bigger win?',
      f'Exploratory — red = elections where attention underdog won — <i>synthetic data</i>',
      w=1100, h=520)
save(fig, 'figures/fig3_gap_and_timeline.png')


# ── Figure 4: Cross-level win rate comparison ─────────────────────────────────
p4 = [bi_p.pvalue, bi_s.pvalue, bi_h.pvalue]
rates4  = [k_p/n_p, k_s/n_s, k_h/n_h]
hi4     = [ci_p.high-k_p/n_p, ci_s.high-k_s/n_s, ci_h.high-k_h/n_h]
lo4     = [k_p/n_p-ci_p.low,  k_s/n_s-ci_s.low,  k_h/n_h-ci_h.low]
xlabs4  = [f'Presidential<br>({n_p} elections)', f'Senate<br>({n_s:,} races)',
           f'House cycles<br>({n_h} cycles)']

fig = go.Figure()
fig.add_trace(go.Bar(
    x=xlabs4, y=rates4,
    error_y=dict(type='data', array=hi4, arrayminus=lo4, thickness=2),
    marker_color=[NAVY,'#2d4a7a','#3d6b9e'], marker_line_width=0, width=0.45,
    customdata=[f'{"p < 0.001" if pv<0.001 else f"p = {pv:.3f}"}  n={n:,}'
                for pv,n in zip(p4,[n_p,n_s,n_h])],
    hovertemplate='%{x}<br>Win rate: %{y:.1%}<br>%{customdata}<extra></extra>',
))
pct_above_ci(fig, rates4, hi4, xlabs4)
fig.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5,
    annotation_text='Chance (50%)', annotation_position='right')
fig.update_yaxes(range=[0, 1.18], tickformat='.0%', title='Attention leader win rate')
p4l = ['p < 0.001' if pv<0.001 else f'p = {pv:.3f}' for pv in p4]
style(fig, 'Figure 4. Attention leader win rate at every level of government',
      f'Presidential {p4l[0]} · Senate {p4l[1]} · House {p4l[2]} — <i>synthetic data</i>',
      w=700, h=540)
save(fig, 'figures/fig4_win_rate_comparison.png')


# ── Figure 5: House party attention share vs. seat share ──────────────────────
r7, _ = stats.pearsonr(df_house['d_attention_share'], df_house['d_seat_share'])
m7, b7 = np.polyfit(df_house['d_attention_share'], df_house['d_seat_share'], 1)
xr7 = np.linspace(df_house['d_attention_share'].min()-0.01,
                   df_house['d_attention_share'].max()+0.01, 200)

fig = go.Figure()
fig.add_shape(type='rect', x0=0.5,x1=0.68,y0=0.5,y1=0.75,
    fillcolor='rgba(16,96,200,0.05)', line_width=0)
fig.add_shape(type='rect', x0=0.32,x1=0.5,y0=0.25,y1=0.5,
    fillcolor='rgba(200,32,16,0.05)', line_width=0)
for maj, color, label in [('D',D_COLOR,'Dem majority'),('R',R_COLOR,'Rep majority')]:
    sub = df_house[df_house['majority']==maj]
    fig.add_trace(go.Scatter(
        x=sub['d_attention_share'], y=sub['d_seat_share'],
        mode='markers+text', text=sub['year'].astype(str),
        textposition='top center', textfont=dict(size=9),
        marker=dict(color=color,size=10,line=dict(color='white',width=1.5)), name=label,
        customdata=sub[['year','d_seats']].values,
        hovertemplate='%{customdata[0]}<br>D attention: %{x:.1%}<br>D seats: %{customdata[1]}<extra></extra>',
    ))
fig.add_trace(go.Scatter(x=xr7, y=m7*xr7+b7, mode='lines',
    line=dict(color='#666',dash='dash',width=1.5), name=f'OLS (r={r7:.2f})'))
fig.add_vline(x=0.5, line_dash='dot', line_color='#aaa', line_width=1.2)
fig.add_hline(y=0.5, line_dash='dot', line_color='#aaa', line_width=1.2)
p7s = 'p < 0.001' if _ < 0.001 else f'p = {_:.3f}'
fig.update_xaxes(tickformat='.0%', title='Democratic attention share (that cycle)', range=[0.33,0.67])
fig.update_yaxes(tickformat='.0%', title='Democratic seat share (out of 435)', range=[0.38,0.72])
style(fig, f'Figure 5. House: does the party dominating coverage win more seats? (r = {r7:.2f})',
      'Each point = one election cycle, 1960–2024 — <i>synthetic data</i>', w=820, h=600)
save(fig, 'figures/fig5_house_seat_share.png')


# ── Figure 6: Attention gap distribution — box + centered points ──────────────
fig = go.Figure()

# Senate: box only (n too large for individual points, would obscure)
fig.add_trace(go.Box(
    y=gaps_s['gap'], name=f'Senate (n={len(gaps_s):,})',
    boxpoints=False,
    marker_color='#3d6b9e',
    line=dict(color='#2d4a7a', width=2),
    fillcolor='rgba(61,107,158,0.18)',
    boxmean='sd',
    width=0.35,
))

# Presidential: box + all points centered on the box
fig.add_trace(go.Box(
    y=gaps_p['gap'], name=f'Presidential (n={len(gaps_p)})',
    boxpoints='all',
    jitter=0.35,
    pointpos=0,          # points centered on box, not offset to side
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
style(fig, 'Figure 6. Distribution of attention gaps by race type',
      'Values > 1× = winner received more coverage · Box: median, IQR, ±1 SD — <i>synthetic data</i>',
      w=680, h=560)
save(fig, 'figures/fig6_gap_distribution.png')


# ── Figure 7: Multi-source forest plot ────────────────────────────────────────
# 2 panels: (A) H1 win rate by source, (B) H2 Pearson r by source
# Each source is an independent data pipeline — if they all agree, the finding is robust.

fig = make_subplots(rows=1, cols=2,
    subplot_titles=[
        '(A) H1: attention leader win rate',
        '(B) H2: mention share → vote share (r)',
    ],
    horizontal_spacing=0.18)

y_labels = list(src_df['name'])

for _, s in src_df.iterrows():
    n_str = f"n={s['n']} elections"
    # Panel A: H1 forest
    fig.add_trace(go.Scatter(
        x=[s['h1']], y=[s['name']],
        error_x=dict(type='data', symmetric=False,
                     array=[s['h1_hi']-s['h1']], arrayminus=[s['h1']-s['h1_lo']],
                     thickness=2, width=6),
        mode='markers+text',
        text=[n_str], textposition='middle right', textfont=dict(size=9, color='#777'),
        marker=dict(color=s['color'], size=13, symbol='circle',
                    line=dict(color='white', width=1.5)),
        name=s['name'], showlegend=False,
        hovertemplate=f"{s['name']}<br>Win rate: %{{x:.0%}}<br>{n_str}<extra></extra>",
    ), row=1, col=1)

    # Panel B: H2 forest
    fig.add_trace(go.Scatter(
        x=[s['r2']], y=[s['name']],
        error_x=dict(type='data', symmetric=False,
                     array=[s['r2_hi']-s['r2']], arrayminus=[s['r2']-s['r2_lo']],
                     thickness=2, width=6),
        mode='markers',
        marker=dict(color=s['color'], size=13, symbol='circle',
                    line=dict(color='white', width=1.5)),
        showlegend=False,
        hovertemplate=f"{s['name']}<br>r = %{{x:.2f}}<br>{n_str}<extra></extra>",
    ), row=1, col=2)

fig.add_vline(x=0.5, line_dash='dot', line_color='#aaa', line_width=1.5,
    annotation_text='Chance', annotation_position='bottom', row=1, col=1)
fig.add_vline(x=0.0, line_dash='dot', line_color='#aaa', line_width=1.5,
    annotation_text='r = 0', annotation_position='bottom', row=1, col=2)

fig.update_xaxes(tickformat='.0%', range=[0, 1.25], title_text='Win rate', row=1, col=1)
fig.update_xaxes(tickformat='.2f', range=[-0.6, 1.1], title_text='Pearson r', row=1, col=2)
fig.update_yaxes(categoryorder='array', categoryarray=y_labels[::-1])
fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
fig.update_yaxes(showgrid=False)
style(fig,
      'Figure 7. Does the finding replicate across 7 independent data sources?',
      'Presidential elections — n varies by source coverage year — <i>synthetic data</i>',
      w=1050, h=500)
save(fig, 'figures/fig7_multi_source_forest.png')


# ── Figure S1: By media era ────────────────────────────────────────────────────
ERAS = [
    ('Print / Radio\n1960–1972', [1960,1964,1968,1972]),
    ('Network TV\n1976–1988',    [1976,1980,1984,1988]),
    ('Cable / Web\n1992–2008',   [1992,1996,2000,2004,2008]),
    ('Social Media\n2012–2024',  [2012,2016,2020,2024]),
]
era_r, era_hi, era_lo, era_ns = [], [], [], []
for _, yrs in ERAS:
    sub = per_pres[per_pres['year'].isin(yrs)]
    k_e, n_e = sub['won'].sum(), len(sub)
    bi_e = stats.binomtest(k_e, n=n_e, p=0.5)
    ci_e = bi_e.proportion_ci()
    era_r.append(k_e/n_e); era_hi.append(ci_e.high-k_e/n_e)
    era_lo.append(k_e/n_e-ci_e.low); era_ns.append(n_e)

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
# Add n labels above points
for xl, r, hi, n in zip(era_labs, era_r, era_hi, era_ns):
    fig.add_annotation(x=xl, y=r+hi+0.03, text=f'<b>{r:.0%}</b><br><span style="font-size:10px">n={n}</span>',
        showarrow=False, font=dict(size=12, color=NAVY), yanchor='bottom', xanchor='center')
fig.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5,
    annotation_text='Chance (50%)', annotation_position='right')
fig.update_yaxes(range=[0, 1.2], tickformat='.0%', title='Attention leader win rate')
style(fig, 'Figure S1. Does the attention–outcome link vary by media era?',
      'Wide CIs reflect small n per era — <i>synthetic data</i>', w=720, h=520)
save(fig, 'figures/figS1_by_era.png')


# ── Figure S2: Window sensitivity ─────────────────────────────────────────────
w_labs  = ['3-month\nwindow', '6-month\nwindow', '12-month\nwindow\n(primary)']
w_noise = [0.38, 0.20, 0.0]
wr, w_hi, w_lo = [], [], []
for noise in w_noise:
    correct = 0
    for yr, grp in df_pres.groupby('year'):
        g2 = grp.copy()
        g2['ms_n'] = (g2['mention_share'] + rng.normal(0, noise, len(g2))).clip(0.01, 0.99)
        g2['ms_n'] /= g2['ms_n'].sum()
        correct += int(g2.loc[g2['ms_n'].idxmax(),'party'] ==
                       g2.loc[g2['popular_vote_winner']==1,'party'].values[0])
    bi_ = stats.binomtest(correct, n=n_p, p=0.5)
    ci_ = bi_.proportion_ci()
    wr.append(correct/n_p); w_hi.append(ci_.high-correct/n_p); w_lo.append(correct/n_p-ci_.low)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=w_labs, y=wr,
    error_y=dict(type='data', array=w_hi, arrayminus=w_lo, thickness=2),
    marker_color=[NAVY,'#2d4a7a','#3d6b9e'], marker_line_width=0, width=0.45,
))
pct_above_ci(fig, wr, w_hi, w_labs)
fig.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5,
    annotation_text='Chance (50%)', annotation_position='right')
fig.update_yaxes(range=[0, 1.18], tickformat='.0%', title='Attention leader win rate')
style(fig, 'Figure S2. Sensitivity: does the measurement window length matter?',
      'Shorter windows modeled by adding noise to mention shares — <i>synthetic data</i>',
      w=640, h=500)
save(fig, 'figures/figS2_window_sensitivity.png')


# ── Table S1 ──────────────────────────────────────────────────────────────────
tbl = df_pres.copy()
tbl['Candidate'] = tbl['candidate'] + ' (' + tbl['party'] + ')'
tbl['Mention share'] = tbl['mention_share'].map('{:.1%}'.format)
tbl['Vote share']    = tbl['popular_vote_share'].map('{:.1%}'.format)
tbl['Winner']        = tbl['popular_vote_winner'].map({1:'✓',0:''})
tbl['Attn leader']   = tbl['attention_leader'].map({1:'✓',0:''})
(tbl[['year','Candidate','Mention share','Vote share','Winner','Attn leader']]
 .rename(columns={'year':'Year'}).sort_values('Year')
 .to_csv('data/processed/table_s1_presidential_FAKE.csv', index=False))
print('  data/processed/table_s1_presidential_FAKE.csv')

print('\nAll done. ✓')
