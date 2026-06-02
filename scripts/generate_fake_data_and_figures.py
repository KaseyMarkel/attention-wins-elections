"""
Generates synthetic mention data for visual prototyping.
All candidate names are fictional Zorblaxian placeholders.
D/R party colors are preserved for real data substitution.

Data model: latent quality score per candidate drives both mention share
and vote share with independent noise, so distributions are continuous
and realistic rather than bimodally split around 50%.

Usage: python scripts/generate_fake_data_and_figures.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

np.random.seed(42)
Path('data/processed').mkdir(exist_ok=True)
Path('figures').mkdir(exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────

FONT    = 'Georgia, serif'
BG      = '#FAFAFA'
GRID    = '#E8E8E8'
D_COLOR = '#1060C8'
R_COLOR = '#C82010'
NAVY    = '#1a1a2e'
UPSET   = '#E05050'
GRAY    = '#CCCCCC'
ERA_COLORS = ['#1a1a2e','#2d4a7a','#3d6b9e','#5a8ec4']

def style(fig, title, subtitle=None, w=820, h=520):
    fig.update_layout(
        title=dict(
            text=title + (f'<br><sup style="color:#555">{subtitle}</sup>' if subtitle else ''),
            font=dict(family=FONT, size=17, color='#111'), x=0,
        ),
        font=dict(family=FONT, size=13),
        plot_bgcolor=BG, paper_bgcolor='white',
        width=w, height=h,
        margin=dict(l=72, r=44, t=84, b=64),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig

def save(fig, path):
    fig.write_image(path, scale=2)
    print(f'  {path}')


# ══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION — latent quality model
# ══════════════════════════════════════════════════════════════════════════════
#
# For each race, two candidates are assigned latent quality scores qa, qb.
# Mention share is derived from quality + independent noise (how newsworthy
# they are is correlated with but not identical to electability).
# Vote share is derived from quality + separate noise.
# This produces a continuous joint distribution without artificial gaps.

rng = np.random.default_rng(99)

def generate_race(mention_spread=0.8, vote_noise=0.30, corr=0.45,
                  ms_clip=(0.20, 0.80), vs_clip=(0.36, 0.64)):
    """
    Logit-normal model. Returns (ms_winner, ms_loser, vs_winner, vs_loser).

    mention_spread: SD of mention log-odds
      0.7 → ~95% of mention shares in [20%, 80%]  (Senate)
      1.0 → ~95% of mention shares in [12%, 88%]  (Presidential)
    vote_noise: SD of independent noise on vote logit (kept tight — real vote
      shares in two-candidate races rarely go below 36% or above 64%).
    """
    while True:
        logit_m = rng.normal(0, mention_spread)
        ms_a = float(np.clip(expit(logit_m), *ms_clip))
        logit_v = logit_m * corr + rng.normal(0, vote_noise)
        vs_a = float(np.clip(expit(logit_v), *vs_clip))
        if abs(vs_a - 0.5) > 0.003:
            break
    ms_b, vs_b = 1.0 - ms_a, 1.0 - vs_a
    if vs_a > vs_b:
        return ms_a, ms_b, vs_a, vs_b
    else:
        return ms_b, ms_a, vs_b, vs_a


# ── Presidential (Zorblaxian names) ──────────────────────────────────────────

PRES = [
    # year  winner                loser                 wp   lp   inc
    (1960, 'Zorbax Krenvel',     'Thrumble Quist',     'D', 'R', 0),
    (1964, 'Glornak Venturis',   'Frixel Stombard',    'D', 'R', 1),
    (1968, 'Quist II',           'Yarvok Drensch',     'R', 'D', 0),
    (1972, 'Quist II',           'Blexor Marvine',     'R', 'D', 1),
    (1976, 'Wumple Crondix',     'Garfax Trellen',     'D', 'R', 0),
    (1980, 'Splorg Vanthorn',    'Wumple Crondix',     'R', 'D', 0),
    (1984, 'Splorg Vanthorn',    'Zeldrick Munvale',   'R', 'D', 1),
    (1988, 'Borrax Grundle',     'Quelvin Droskip',    'R', 'D', 0),
    (1992, 'Mirval Prondex',     'Borrax Grundle',     'D', 'R', 0),
    (1996, 'Mirval Prondex',     'Flubb Ranstorm',     'D', 'R', 1),
    (2000, 'Xenthox Quivel',     'Drembix Faltor',     'R', 'D', 0),
    (2004, 'Xenthox Quivel',     'Clorvis Stemple',    'R', 'D', 1),
    (2008, 'Zumbrix Halvore',    'Skrendle Mactorvish','D', 'R', 0),
    (2012, 'Zumbrix Halvore',    'Prumdex Vorthalack', 'D', 'R', 1),
    (2016, 'Grumvox Spralton',   'Clindivar Norquess', 'R', 'D', 0),
    (2020, 'Blengle Arvondex',   'Grumvox Spralton',   'D', 'R', 0),
    (2024, 'Grumvox Spralton',   'Harrinda Vex',       'R', 'D', 0),
]
REAL_VOTE = {  # actual popular vote shares from election_results.csv
    (1960,'D'):(0.4972,1),(1960,'R'):(0.4955,0),
    (1964,'D'):(0.6131,1),(1964,'R'):(0.3858,0),
    (1968,'R'):(0.4342,1),(1968,'D'):(0.4298,0),
    (1972,'R'):(0.6077,1),(1972,'D'):(0.3780,0),
    (1976,'D'):(0.5008,1),(1976,'R'):(0.4796,0),
    (1980,'R'):(0.5075,1),(1980,'D'):(0.4107,0),
    (1984,'R'):(0.5849,1),(1984,'D'):(0.4083,0),
    (1988,'R'):(0.5337,1),(1988,'D'):(0.4585,0),
    (1992,'D'):(0.4300,1),(1992,'R'):(0.3739,0),
    (1996,'D'):(0.4927,1),(1996,'R'):(0.4072,0),
    (2000,'R'):(0.4763,1),(2000,'D'):(0.4838,0),
    (2004,'R'):(0.5073,1),(2004,'D'):(0.4823,0),
    (2008,'D'):(0.5259,1),(2008,'R'):(0.4573,0),
    (2012,'D'):(0.5106,1),(2012,'R'):(0.4706,0),
    (2016,'R'):(0.4609,1),(2016,'D'):(0.4818,0),
    (2020,'D'):(0.5135,1),(2020,'R'):(0.4685,0),
    (2024,'R'):(0.4977,1),(2024,'D'):(0.4823,0),
}

pres_rows = []
for year, wn, ln, wp, lp, inc in PRES:
    ms_w, ms_l, _, _ = generate_race(mention_spread=1.0, vote_noise=0.30, corr=0.5,
                                      ms_clip=(0.15, 0.85), vs_clip=(0.37, 0.63))
    wv, wwin = REAL_VOTE[(year, wp)]
    lv, lwin = REAL_VOTE[(year, lp)]
    attn_w = int(ms_w > ms_l)
    pres_rows += [
        dict(level='Presidential', year=year, candidate=wn, party=wp,
             mention_share=ms_w, popular_vote_share=wv,
             popular_vote_winner=1, attention_leader=attn_w, incumbent_running=inc),
        dict(level='Presidential', year=year, candidate=ln, party=lp,
             mention_share=ms_l, popular_vote_share=lv,
             popular_vote_winner=0, attention_leader=1-attn_w, incumbent_running=0),
    ]
df_pres = pd.DataFrame(pres_rows)

ZORB_F = ['Zorbax','Thrumble','Glornak','Frixel','Yarvok','Blexor','Wumple','Garfax',
          'Splorg','Zeldrick','Borrax','Quelvin','Mirval','Flubb','Xenthox','Drembix',
          'Clorvis','Zumbrix','Skrendle','Prumdex','Grumvox','Clindivar','Blengle',
          'Harrinda','Vrenlik','Quorvast','Thumlek','Plorbex','Stelzik','Drovnik',
          'Frundax','Glorbik','Wrendle','Quarvox','Smundrel','Threxis','Gorvalak']
ZORB_L = ['Vrenlik','Quorvast','Drensch','Marvine','Stombard','Krenvel','Venturis',
          'Crondix','Grundle','Droskip','Ranstorm','Stemple','Halvore','Norquess',
          'Blendrix','Gorvalak','Threxis','Smundrel','Quarvox','Plorbex','Stelzik',
          'Drovnik','Frundax','Glorbik','Wrendle','Thumlek','Vex','Prondex','Faltor',
          'Spralton','Arvondex','Vanthorn','Mactorvish','Grumvox','Zorblax','Blendrik']

def zorb():
    return f'{rng.choice(ZORB_F)} {rng.choice(ZORB_L)}'

# ── Senate ────────────────────────────────────────────────────────────────────

senate_rows = []
for cycle in range(1960, 2025, 2):
    for _ in range(33):
        ms_w, ms_l, vs_w, vs_l = generate_race(mention_spread=0.7, vote_noise=0.65, corr=0.40)
        party_w, party_l = rng.choice([('D','R'),('R','D')])
        attn_w = int(ms_w > ms_l)
        senate_rows += [
            dict(level='Senate', year=cycle, candidate=zorb(), party=party_w,
                 mention_share=ms_w, popular_vote_share=vs_w,
                 popular_vote_winner=1, attention_leader=attn_w),
            dict(level='Senate', year=cycle, candidate=zorb(), party=party_l,
                 mention_share=ms_l, popular_vote_share=vs_l,
                 popular_vote_winner=0, attention_leader=1-attn_w),
        ]
df_senate = pd.DataFrame(senate_rows)

# ── House (aggregate per cycle) ───────────────────────────────────────────────

HOUSE_D_SEATS = {
    1960:263,1962:258,1964:295,1966:248,1968:243,1970:255,1972:242,
    1974:291,1976:292,1978:277,1980:243,1982:269,1984:253,1986:258,
    1988:260,1990:267,1992:258,1994:204,1996:207,1998:211,2000:212,
    2002:204,2004:202,2006:233,2008:257,2010:193,2012:201,2014:188,
    2016:194,2018:235,2020:222,2022:213,2024:215,
}
house_rows = []
for cycle, d_seats in HOUSE_D_SEATS.items():
    d_seat_share = d_seats / 435
    d_attn = float(np.clip(rng.normal(0.48 + 0.18*(d_seat_share-0.5), 0.04), 0.35, 0.65))
    house_rows.append(dict(
        year=cycle, d_attention_share=d_attn, r_attention_share=1-d_attn,
        d_seat_share=d_seat_share, r_seat_share=1-d_seat_share, d_seats=d_seats,
        majority='D' if d_seats>217 else 'R',
        attention_majority='D' if d_attn>0.5 else 'R',
    ))
df_house = pd.DataFrame(house_rows)
df_house['attention_predicted_majority'] = (df_house['majority']==df_house['attention_majority']).astype(int)

# Save
df_pres.to_csv('data/processed/mention_share_FAKE.csv', index=False)
print(f'Data: {len(df_pres)} presidential rows, {len(df_senate)} senate rows, {len(df_house)} house cycles\n')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print('Generating figures...')

# ── Figure 1: H1 Win rate — Presidential ─────────────────────────────────────

per_pres = (df_pres.groupby('year')
    .apply(lambda g: int((g.loc[g['attention_leader']==1,'popular_vote_winner']==1).any()))
    .reset_index(name='won'))
k_p, n_p = per_pres['won'].sum(), len(per_pres)
bi_p = stats.binomtest(k_p, n=n_p, p=0.5, alternative='two-sided')
ci_p = bi_p.proportion_ci()

fig = go.Figure()
fig.add_trace(go.Bar(
    x=['Attention leader','Chance (50%)'], y=[k_p/n_p, 0.5],
    error_y=dict(type='data', array=[ci_p.high-k_p/n_p, 0],
                 arrayminus=[k_p/n_p-ci_p.low, 0], thickness=2),
    marker_color=[NAVY, GRAY], marker_line_width=0, width=0.4,
    text=[f'<b>{k_p/n_p:.0%}</b>','50%'], textposition='outside', textfont=dict(size=15),
))
fig.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5)
fig.update_yaxes(range=[0,1.15], tickformat='.0%', title='Win rate')
p_p = f'p = {bi_p.pvalue:.3f}' if bi_p.pvalue >= 0.001 else 'p < 0.001'
style(fig, f'Figure 1. Attention leader wins popular vote: {k_p}/{n_p} elections ({k_p/n_p:.0%})',
      f'Binomial test vs. 50% chance — {p_p} — 95% CI [{ci_p.low:.0%}, {ci_p.high:.0%}] — <i>synthetic data</i>',
      w=560, h=480)
save(fig, 'figures/fig1_h1_win_rate.png')

# ── Figure 2: H2 Scatter — Presidential ──────────────────────────────────────

r2, p2 = stats.pearsonr(df_pres['mention_share'], df_pres['popular_vote_share'])
m2, b2 = np.polyfit(df_pres['mention_share'], df_pres['popular_vote_share'], 1)
xr = np.linspace(df_pres['mention_share'].min()-0.02, df_pres['mention_share'].max()+0.02, 200)

fig = go.Figure()
for party, color, label in [('D',D_COLOR,'Democrat'),('R',R_COLOR,'Republican')]:
    sub = df_pres[df_pres['party']==party]
    fig.add_trace(go.Scatter(
        x=sub['mention_share'], y=sub['popular_vote_share'], mode='markers',
        customdata=sub['candidate']+" '"+sub['year'].astype(str).str[-2:],
        hovertemplate='%{customdata}<br>Mention: %{x:.1%} · Vote: %{y:.1%}<extra></extra>',
        marker=dict(color=color, size=11, line=dict(color='white', width=1.5)), name=label,
    ))
notable = df_pres[
    (df_pres['mention_share']>0.65)|(df_pres['mention_share']<0.33)|
    (df_pres['popular_vote_share']>0.58)|(df_pres['popular_vote_share']<0.40)
]
fig.add_trace(go.Scatter(
    x=notable['mention_share'], y=notable['popular_vote_share'], mode='text',
    text=notable['candidate']+" '"+notable['year'].astype(str).str[-2:],
    textposition='top center', textfont=dict(size=9, color='#444'),
    showlegend=False, hoverinfo='skip',
))
fig.add_trace(go.Scatter(x=xr, y=m2*xr+b2, mode='lines',
    line=dict(color='#555', dash='dash', width=1.5), name=f'OLS (r = {r2:.2f})'))
fig.update_xaxes(tickformat='.0%', title='Mention share (12 months pre-election day)')
fig.update_yaxes(tickformat='.0%', title='Popular vote share')
p2s = 'p < 0.001' if p2<0.001 else f'p = {p2:.3f}'
style(fig, f'Figure 2. Mention share vs. popular vote share (r = {r2:.2f}, {p2s})',
      f'n = {len(df_pres)} candidate-elections, 1960–2024 — <i>synthetic data</i>', w=820, h=580)
save(fig, 'figures/fig2_h2_scatter.png')

# ── Figure 3: H3 Gap vs. margin — Presidential ───────────────────────────────

w_p = df_pres[df_pres['popular_vote_winner']==1][['year','candidate','mention_share','popular_vote_share']]
l_p = df_pres[df_pres['popular_vote_winner']==0][['year','candidate','mention_share','popular_vote_share']]
h3 = w_p.merge(l_p, on='year', suffixes=('_w','_l'))
h3['mention_gap'] = h3['mention_share_w'] / h3['mention_share_l']
h3['vote_margin'] = h3['popular_vote_share_w'] - h3['popular_vote_share_l']
h3['upset'] = h3['mention_gap'] < 1
r3, p3 = stats.pearsonr(h3['mention_gap'], h3['vote_margin'])

fig = go.Figure()
for upset, color, label in [(False,NAVY,'Attention predicted winner'),(True,UPSET,'Attention upset')]:
    sub = h3[h3['upset']==upset]
    fig.add_trace(go.Scatter(
        x=sub['mention_gap'], y=sub['vote_margin'], mode='markers+text',
        text=sub['year'].astype(str), textposition='top center', textfont=dict(size=9),
        marker=dict(color=color, size=11, line=dict(color='white', width=1.5)), name=label,
    ))
xr3 = np.linspace(h3['mention_gap'].min()-0.05, h3['mention_gap'].max()+0.05, 200)
m3, b3 = np.polyfit(h3['mention_gap'], h3['vote_margin'], 1)
fig.add_trace(go.Scatter(x=xr3, y=m3*xr3+b3, mode='lines',
    line=dict(color='#888', dash='dash', width=1.5), name=f'OLS (r = {r3:.2f})'))
fig.add_vline(x=1, line_dash='dot', line_color='#aaa', line_width=1.5,
    annotation_text='Equal attention', annotation_position='top right')
fig.add_hline(y=0, line_dash='dot', line_color='#aaa', line_width=1)
fig.update_xaxes(title='Mention ratio (winner ÷ loser)')
fig.update_yaxes(tickformat='.1%', title='Popular vote margin (winner − loser)')
p3s = 'p < 0.001' if p3<0.001 else f'p = {p3:.3f}'
style(fig, f'Figure 3. Attention gap vs. vote margin (r = {r3:.2f}, {p3s})',
      f'Exploratory — red = elections where attention underdog won — <i>synthetic data</i>', w=820, h=560)
save(fig, 'figures/fig3_h3_gap_margin.png')

# ── Figure 4: Timeline ────────────────────────────────────────────────────────

h3s = h3.sort_values('year')
fig = go.Figure()
fig.add_trace(go.Bar(
    x=h3s['year'], y=h3s['mention_gap'],
    text=[f"{r['candidate_w'].split()[0]} {r['mention_gap']:.2f}×" for _,r in h3s.iterrows()],
    textposition='outside', textfont=dict(size=9),
    marker_color=[UPSET if u else NAVY for u in h3s['upset']],
    marker_line_width=0, width=3,
))
fig.add_hline(y=1, line_dash='dot', line_color='#aaa', line_width=1.5,
    annotation_text='Equal attention (1×)', annotation_position='top left')
fig.update_xaxes(title='Election Year', tickmode='array',
    tickvals=h3s['year'].tolist(), tickangle=-45)
fig.update_yaxes(title='Mention ratio (winner ÷ loser)', range=[0, h3s['mention_gap'].max()*1.25])
style(fig, 'Figure 4. Presidential attention gap by election year',
      'Dark = attention predicted winner · Red = attention upset — <i>synthetic data</i>', w=900, h=500)
save(fig, 'figures/fig4_timeline.png')

# ── Figure 5: Win rate comparison — all levels ───────────────────────────────

# Senate
senate_pairs = df_senate.copy()
senate_pairs['race_id'] = senate_pairs.groupby('year').cumcount() // 2
attn_won_s = (senate_pairs.groupby(['year','race_id'])
    .apply(lambda g: int((g.loc[g['attention_leader']==1,'popular_vote_winner']==1).any())))
k_s, n_s = attn_won_s.sum(), len(attn_won_s)
bi_s = stats.binomtest(k_s, n=n_s, p=0.5, alternative='two-sided')
ci_s = bi_s.proportion_ci()

# House
k_h, n_h = df_house['attention_predicted_majority'].sum(), len(df_house)
bi_h = stats.binomtest(k_h, n=n_h, p=0.5, alternative='two-sided')
ci_h = bi_h.proportion_ci()

rates  = [k_p/n_p, k_s/n_s, k_h/n_h]
hi_err = [ci_p.high-k_p/n_p, ci_s.high-k_s/n_s, ci_h.high-k_h/n_h]
lo_err = [k_p/n_p-ci_p.low,  k_s/n_s-ci_s.low,  k_h/n_h-ci_h.low]
pvals  = [bi_p.pvalue, bi_s.pvalue, bi_h.pvalue]
ns_    = [n_p, n_s, n_h]
xlabs  = [f'Presidential<br>({n_p} elections)', f'Senate<br>({n_s:,} races)', f'House cycles<br>({n_h} cycles)']

fig = go.Figure()
fig.add_trace(go.Bar(
    x=xlabs, y=rates,
    error_y=dict(type='data', array=hi_err, arrayminus=lo_err, thickness=2),
    marker_color=[NAVY,'#2d4a7a','#3d6b9e'], marker_line_width=0, width=0.45,
    text=[f'<b>{r:.0%}</b>' for r in rates], textposition='outside', textfont=dict(size=14),
    customdata=[f'{"p < 0.001" if pv<0.001 else f"p = {pv:.3f}"}  n={n:,}' for pv,n in zip(pvals,ns_)],
    hovertemplate='%{x}<br>Win rate: %{y:.1%}<br>%{customdata}<extra></extra>',
))
fig.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5,
    annotation_text='Chance (50%)', annotation_position='right')
fig.update_yaxes(range=[0,1.12], tickformat='.0%', title='Attention leader win rate')
p_labs = ['p < 0.001' if pv<0.001 else f'p = {pv:.3f}' for pv in pvals]
style(fig, 'Figure 5. Attention leader win rate at every level of government',
      f'Presidential {p_labs[0]} · Senate {p_labs[1]} · House {p_labs[2]} — <i>synthetic data</i>',
      w=700, h=520)
save(fig, 'figures/fig5_win_rate_comparison.png')

# ── Figure 6: Senate scatter ──────────────────────────────────────────────────

r6, p6 = stats.pearsonr(df_senate['mention_share'], df_senate['popular_vote_share'])
m6, b6 = np.polyfit(df_senate['mention_share'], df_senate['popular_vote_share'], 1)
xr6 = np.linspace(0.22, 0.80, 200)

fig = go.Figure()
for party, color, label in [('D',D_COLOR,'Democrat'),('R',R_COLOR,'Republican')]:
    sub = df_senate[df_senate['party']==party]
    fig.add_trace(go.Scatter(
        x=sub['mention_share'], y=sub['popular_vote_share'], mode='markers',
        marker=dict(color=color, size=5, opacity=0.40, line=dict(width=0)), name=label,
    ))
fig.add_trace(go.Scatter(x=xr6, y=m6*xr6+b6, mode='lines',
    line=dict(color='#222', dash='dash', width=2), name=f'OLS (r = {r6:.2f})'))
p6s = 'p < 0.001' if p6<0.001 else f'p = {p6:.3f}'
fig.update_xaxes(tickformat='.0%', title='Mention share (12 months pre-election day)')
fig.update_yaxes(tickformat='.0%', title='Vote share')
style(fig, f'Figure 6. Senate: mention share vs. vote share (r = {r6:.2f}, {p6s})',
      f'n = {len(df_senate):,} candidate-races, 1960–2024 — <i>synthetic data</i>', w=820, h=560)
save(fig, 'figures/fig6_senate_scatter.png')

# ── Figure 7: House seat share ────────────────────────────────────────────────

r7, p7 = stats.pearsonr(df_house['d_attention_share'], df_house['d_seat_share'])
m7, b7 = np.polyfit(df_house['d_attention_share'], df_house['d_seat_share'], 1)
xr7 = np.linspace(df_house['d_attention_share'].min()-0.01, df_house['d_attention_share'].max()+0.01, 200)

fig = go.Figure()
fig.add_shape(type='rect', x0=0.5, x1=0.68, y0=0.5, y1=0.75,
    fillcolor='rgba(16,96,200,0.05)', line_width=0)
fig.add_shape(type='rect', x0=0.32, x1=0.5, y0=0.25, y1=0.5,
    fillcolor='rgba(200,32,16,0.05)', line_width=0)
for majority, color, label in [('D',D_COLOR,'Democrat majority'),('R',R_COLOR,'Republican majority')]:
    sub = df_house[df_house['majority']==majority]
    fig.add_trace(go.Scatter(
        x=sub['d_attention_share'], y=sub['d_seat_share'], mode='markers+text',
        text=sub['year'].astype(str), textposition='top center', textfont=dict(size=9),
        marker=dict(color=color, size=10, line=dict(color='white', width=1.5)), name=label,
        customdata=sub[['year','d_seats','majority']].values,
        hovertemplate='%{customdata[0]}<br>D attention: %{x:.1%}<br>D seats: %{customdata[1]}<extra></extra>',
    ))
fig.add_trace(go.Scatter(x=xr7, y=m7*xr7+b7, mode='lines',
    line=dict(color='#666', dash='dash', width=1.5), name=f'OLS (r = {r7:.2f})'))
fig.add_vline(x=0.5, line_dash='dot', line_color='#aaa', line_width=1.2)
fig.add_hline(y=0.5, line_dash='dot', line_color='#aaa', line_width=1.2)
p7s = 'p < 0.001' if p7<0.001 else f'p = {p7:.3f}'
fig.update_xaxes(tickformat='.0%', title='Democratic party attention share (that cycle)', range=[0.33,0.67])
fig.update_yaxes(tickformat='.0%', title='Democratic seat share (out of 435)', range=[0.38,0.72])
style(fig, f'Figure 7. House: party attention share vs. seat share (r = {r7:.2f}, {p7s})',
      'Each point = one election cycle, 1960–2024 — <i>synthetic data</i>', w=820, h=600)
save(fig, 'figures/fig7_house_seat_share.png')

# ── Figure 8: Attention gap distribution — histogram + KDE ───────────────────

def compute_gaps(df_, level_):
    rows = []
    df_ = df_.copy()
    df_['race_id'] = df_.groupby('year').cumcount() // 2
    for (yr, rid), g in df_.groupby(['year','race_id']):
        w = g[g['popular_vote_winner']==1]
        l = g[g['popular_vote_winner']==0]
        if len(w)==0 or len(l)==0: continue
        rows.append(dict(level=level_, year=yr,
                         gap=float(w['mention_share'].values[0])/float(l['mention_share'].values[0])))
    return pd.DataFrame(rows)

gaps_p = compute_gaps(df_pres, 'Presidential')
gaps_s = compute_gaps(df_senate, 'Senate')

# KDE for smooth density curves
from scipy.stats import gaussian_kde

# Senate: histogram + KDE. Presidential: KDE only + rug (n=17 too small for bars).
xk = np.linspace(0.1, 4.5, 400)

fig = go.Figure()

# Senate histogram (behind everything)
bins = np.linspace(0.1, 4.5, 45)
bin_c = (bins[:-1] + bins[1:]) / 2
counts_s, _ = np.histogram(gaps_s['gap'], bins=bins, density=True)
fig.add_trace(go.Bar(
    x=bin_c, y=counts_s, name='Senate (histogram)',
    marker_color='#5a8ec4', opacity=0.45, marker_line_width=0,
    width=(bins[1]-bins[0])*0.92,
    hovertemplate='Senate<br>Gap: %{x:.2f}×<br>Density: %{y:.3f}<extra></extra>',
))

# KDE curves for both
for gaps, color, label, bw in [(gaps_s,'#1a5fa8','Senate (KDE)',0.18),(gaps_p,NAVY,'Presidential (KDE)',0.30)]:
    kde = gaussian_kde(gaps['gap'], bw_method=bw)
    fig.add_trace(go.Scatter(
        x=xk, y=kde(xk), mode='lines', name=label,
        line=dict(color=color, width=2.5), hoverinfo='skip',
    ))

# Presidential rug — individual election tick marks along y=0
rug_y = np.zeros(len(gaps_p))
fig.add_trace(go.Scatter(
    x=gaps_p['gap'], y=rug_y, mode='markers', name='Presidential (elections)',
    marker=dict(symbol='line-ns', size=12, color=NAVY,
                line=dict(color=NAVY, width=2)),
    hovertemplate='%{customdata}<br>Gap: %{x:.2f}×<extra></extra>',
    customdata=gaps_p['year'].astype(str),
))

fig.add_vline(x=1, line_dash='dot', line_color='#555', line_width=2,
    annotation_text='Equal attention (1×)', annotation_position='top right',
    annotation_font=dict(size=11))
fig.update_layout(barmode='overlay')
fig.update_xaxes(title='Mention ratio (winner ÷ loser)', range=[0.1, 4.5])
fig.update_yaxes(title='Density', showticklabels=False, range=[-0.05, None])
style(fig, 'Figure 8. Distribution of attention gaps by race type',
      'Senate: histogram + KDE · Presidential: KDE + individual elections (tick marks) — <i>synthetic data</i>',
      w=800, h=500)
save(fig, 'figures/fig8_gap_distribution.png')


# ══════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTAL FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ── Figure S1: Win rate by media era ─────────────────────────────────────────
# Tests whether the attention→outcome link strengthened as media evolved.

ERAS = [
    ('Print / Radio\n1960–1972',  [1960,1964,1968,1972]),
    ('Network TV\n1976–1988',     [1976,1980,1984,1988]),
    ('Cable / Web\n1992–2008',    [1992,1996,2000,2004,2008]),
    ('Social Media\n2012–2024',   [2012,2016,2020,2024]),
]

era_rates, era_ns, era_hi, era_lo, era_cols = [], [], [], [], ERA_COLORS
for era_label, years in ERAS:
    sub = per_pres[per_pres['year'].isin(years)]
    k_e, n_e = sub['won'].sum(), len(sub)
    if n_e == 0: continue
    bi_e = stats.binomtest(k_e, n=n_e, p=0.5, alternative='two-sided')
    ci_e = bi_e.proportion_ci()
    era_rates.append(k_e/n_e)
    era_ns.append(n_e)
    era_hi.append(ci_e.high - k_e/n_e)
    era_lo.append(k_e/n_e - ci_e.low)

era_labels = [e[0] for e in ERAS]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=era_labels, y=era_rates,
    mode='markers+lines+text',
    text=[f'{r:.0%}' for r in era_rates],
    textposition='top center', textfont=dict(size=13),
    error_y=dict(type='data', array=era_hi, arrayminus=era_lo, thickness=2),
    marker=dict(color=ERA_COLORS, size=14, line=dict(color='white', width=2)),
    line=dict(color='#888', dash='dot', width=1.5),
    showlegend=False,
))
fig.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5,
    annotation_text='Chance (50%)', annotation_position='right')
fig.update_yaxes(range=[0, 1.15], tickformat='.0%', title='Attention leader win rate')
fig.update_xaxes(title='')
style(fig, 'Figure S1. Presidential attention–outcome link by media era',
      'Does the effect strengthen as media moves from print to social? — <i>synthetic data</i>', w=720, h=500)
save(fig, 'figures/figS1_by_era.png')

# ── Figure S2: Sensitivity — measurement window ──────────────────────────────
# Shorter windows = noisier mention data. We simulate by adding noise to
# presidential mention shares and recomputing the win rate.

window_labels = ['3-month window', '6-month window', '12-month window\n(primary)']
window_noise  = [0.35, 0.18, 0.0]  # additional noise added to mention shares
w_rates, w_hi, w_lo = [], [], []

for noise in window_noise:
    correct = 0
    for year, grp in df_pres.groupby('year'):
        grp = grp.copy()
        grp['ms_noisy'] = (grp['mention_share'] + rng.normal(0, noise, len(grp))).clip(0.01, 0.99)
        grp['ms_noisy'] /= grp['ms_noisy'].sum()
        attn_winner_party = grp.loc[grp['ms_noisy'].idxmax(), 'party']
        vote_winner_party = grp.loc[grp['popular_vote_winner']==1, 'party'].values[0]
        correct += int(attn_winner_party == vote_winner_party)
    n_ = df_pres['year'].nunique()
    bi_ = stats.binomtest(correct, n=n_, p=0.5, alternative='two-sided')
    ci_ = bi_.proportion_ci()
    w_rates.append(correct/n_)
    w_hi.append(ci_.high - correct/n_)
    w_lo.append(correct/n_ - ci_.low)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=window_labels, y=w_rates,
    error_y=dict(type='data', array=w_hi, arrayminus=w_lo, thickness=2),
    marker_color=[NAVY, '#2d4a7a', '#3d6b9e'], marker_line_width=0, width=0.45,
    text=[f'<b>{r:.0%}</b>' for r in w_rates],
    textposition='outside', textfont=dict(size=14),
))
fig.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5,
    annotation_text='Chance (50%)', annotation_position='right')
fig.update_yaxes(range=[0, 1.12], tickformat='.0%', title='Attention leader win rate')
style(fig, 'Figure S2. Sensitivity: does the measurement window matter?',
      'Shorter windows modeled by adding noise to mention shares — <i>synthetic data</i>', w=640, h=480)
save(fig, 'figures/figS2_window_sensitivity.png')

# ── Table S1: Presidential elections summary ──────────────────────────────────

tbl = df_pres.copy()
tbl['label'] = tbl['candidate'] + ' (' + tbl['party'] + ')'
tbl['Mention share'] = tbl['mention_share'].map('{:.1%}'.format)
tbl['Vote share'] = tbl['popular_vote_share'].map('{:.1%}'.format)
tbl['Won popular vote'] = tbl['popular_vote_winner'].map({1:'✓', 0:''})
tbl['Attention leader'] = tbl['attention_leader'].map({1:'✓', 0:''})
tbl_out = (tbl[['year','label','Mention share','Vote share','Won popular vote','Attention leader']]
    .rename(columns={'year':'Year','label':'Candidate'})
    .sort_values('Year'))
tbl_out.to_csv('data/processed/table_s1_presidential_FAKE.csv', index=False)
print('  data/processed/table_s1_presidential_FAKE.csv')

print('\nAll done. ✓')
