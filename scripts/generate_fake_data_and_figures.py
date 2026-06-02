"""
Generates synthetic mention data for visual prototyping.
All candidate names are fictional Zorblaxian placeholders.
D/R party colors are preserved for when real data is substituted.

Usage: python scripts/generate_fake_data_and_figures.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

np.random.seed(42)
Path('data/processed').mkdir(exist_ok=True)
Path('figures').mkdir(exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────

FONT     = 'Georgia, serif'
BG       = '#FAFAFA'
GRID     = '#E8E8E8'
D_COLOR  = '#1060C8'
R_COLOR  = '#C82010'
NAVY     = '#1a1a2e'
UPSET    = '#E05050'
GRAY     = '#CCCCCC'

def apply_style(fig, title, subtitle=None, w=820, h=520):
    full_title = title
    if subtitle:
        full_title += f'<br><sup>{subtitle}</sup>'
    fig.update_layout(
        title=dict(text=full_title, font=dict(family=FONT, size=17, color='#111'), x=0),
        font=dict(family=FONT, size=13),
        plot_bgcolor=BG,
        paper_bgcolor='white',
        width=w, height=h,
        margin=dict(l=72, r=44, t=80, b=64),
        legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PRESIDENTIAL DATA (Zorblaxian names)
# ══════════════════════════════════════════════════════════════════════════════

PRES_ELECTIONS = [
    # year  winner_name           loser_name            wp   lp  inc  winner_vote  loser_vote
    (1960, 'Zorbax Krenvel',     'Thrumble Quist',     'D', 'R', 0,  0.4972, 0.4955),
    (1964, 'Glornak Venturis',   'Frixel Stombard',    'D', 'R', 1,  0.6131, 0.3858),
    (1968, 'Quist II',           'Yarvok Drensch',     'R', 'D', 0,  0.4342, 0.4298),
    (1972, 'Quist II',           'Blexor Marvine',     'R', 'D', 1,  0.6077, 0.3780),
    (1976, 'Wumple Crondix',     'Garfax Trellen',     'D', 'R', 0,  0.5008, 0.4796),
    (1980, 'Splorg Vanthorn',    'Wumple Crondix',     'R', 'D', 0,  0.5075, 0.4107),
    (1984, 'Splorg Vanthorn',    'Zeldrick Munvale',   'R', 'D', 1,  0.5849, 0.4083),
    (1988, 'Borrax Grundle',     'Quelvin Droskip',    'R', 'D', 0,  0.5337, 0.4585),
    (1992, 'Mirval Prondex',     'Borrax Grundle',     'D', 'R', 0,  0.4300, 0.3739),
    (1996, 'Mirval Prondex',     'Flubb Ranstorm',     'D', 'R', 1,  0.4927, 0.4072),
    (2000, 'Xenthox Quivel',     'Drembix Faltor',     'R', 'D', 0,  0.4763, 0.4838),
    (2004, 'Xenthox Quivel',     'Clorvis Stemple',    'R', 'D', 1,  0.5073, 0.4823),
    (2008, 'Zumbrix Halvore',    'Skrendle Mactorvish','D', 'R', 0,  0.5259, 0.4573),
    (2012, 'Zumbrix Halvore',    'Prumdex Vorthalack', 'D', 'R', 1,  0.5106, 0.4706),
    (2016, 'Grumvox Spralton',   'Clindivar Norquess', 'R', 'D', 0,  0.4609, 0.4818),
    (2020, 'Blengle Arvondex',   'Grumvox Spralton',   'D', 'R', 0,  0.5135, 0.4685),
    (2024, 'Grumvox Spralton',   'Harrinda Vex',       'R', 'D', 0,  0.4977, 0.4823),
]

# Attention upsets: these years the loser got more coverage
PRES_UPSETS = {1968, 2000, 2016}

pres_rows = []
for year, wn, ln, wp, lp, inc, wv, lv in PRES_ELECTIONS:
    w_share = np.random.uniform(0.52, 0.72)
    l_share = 1.0 - w_share
    if year in PRES_UPSETS:
        w_share, l_share = l_share, w_share
    for name, party, vote_share, pv_winner, attn_leader in [
        (wn, wp, wv, 1, int(year not in PRES_UPSETS)),
        (ln, lp, lv, 0, int(year in PRES_UPSETS)),
    ]:
        pres_rows.append(dict(
            level='Presidential', year=year, candidate=name, party=party,
            mention_share=w_share if name == wn else l_share,
            popular_vote_share=vote_share, popular_vote_winner=pv_winner,
            attention_leader=attn_leader, incumbent_running=inc, source='FAKE',
        ))

df_pres = pd.DataFrame(pres_rows)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SENATE DATA
# ══════════════════════════════════════════════════════════════════════════════

ZORB_FIRST = [
    'Zorbax','Thrumble','Glornak','Frixel','Yarvok','Blexor','Wumple','Garfax',
    'Splorg','Zeldrick','Borrax','Quelvin','Mirval','Flubb','Xenthox','Drembix',
    'Clorvis','Zumbrix','Skrendle','Prumdex','Grumvox','Clindivar','Blengle',
    'Harrinda','Vrenlik','Quorvast','Drensch','Marvine','Stombard','Arvondex',
    'Thumlek','Vorthalack','Mactorvish','Prondex','Krenvel','Venturis','Crondix',
    'Vanthorn','Grundle','Droskip','Ranstorm','Quivel','Faltor','Stemple','Halvore',
    'Spralton','Norquess','Vex','Blendrix','Gorvalak','Threxis','Smundrel',
    'Quarvox','Plorbex','Stelzik','Drovnik','Frundax','Glorbik','Wrendle',
]
ZORB_LAST = [
    'Vrenlik','Quorvast','Drensch','Marvine','Stombard','Krenvel','Venturis',
    'Crondix','Grundle','Droskip','Ranstorm','Stemple','Halvore','Norquess',
    'Blendrix','Gorvalak','Threxis','Smundrel','Quarvox','Plorbex','Stelzik',
    'Drovnik','Frundax','Glorbik','Wrendle','Thumlek','Vex','Prondex','Faltor',
    'Spralton','Arvondex','Vanthorn','Mactorvish','Venturis','Grumvox','Zorblax',
]

rng = np.random.default_rng(99)

def zorb_name(seed_offset=0):
    i = rng.integers(0, len(ZORB_FIRST))
    j = rng.integers(0, len(ZORB_LAST))
    return f'{ZORB_FIRST[i]} {ZORB_LAST[j]}'

# Senate election cycles: every 2 years 1960-2024 = 33 cycles
# ~33 races per cycle, competitive races only (won't model uncontested)
SENATE_CYCLES = list(range(1960, 2025, 2))
RACES_PER_CYCLE = 33

senate_rows = []
for cycle in SENATE_CYCLES:
    for _ in range(RACES_PER_CYCLE):
        party_w = rng.choice(['D', 'R'])
        party_l = 'R' if party_w == 'D' else 'D'

        # Winner gets 53–72% of mentions ~78% of the time; upset ~22%
        upset = rng.random() < 0.22
        w_mention = rng.uniform(0.53, 0.72)
        l_mention = 1.0 - w_mention
        if upset:
            w_mention, l_mention = l_mention, w_mention

        # Vote share: winner typically 51-63%
        w_vote = rng.uniform(0.51, 0.63)
        # Slight correlation with mention share
        w_vote = np.clip(w_vote + 0.08 * (w_mention - 0.5), 0.50, 0.68)
        l_vote = 1.0 - w_vote

        for suffix, party, ms, vs, pvw, al in [
            ('_w', party_w, w_mention, w_vote, 1, int(not upset)),
            ('_l', party_l, l_mention, l_vote, 0, int(upset)),
        ]:
            senate_rows.append(dict(
                level='Senate', year=cycle,
                candidate=zorb_name(), party=party,
                mention_share=ms, popular_vote_share=vs,
                popular_vote_winner=pvw, attention_leader=al,
                incumbent_running=0, source='FAKE',
            ))

df_senate = pd.DataFrame(senate_rows)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HOUSE DATA (party-aggregate per cycle)
# ══════════════════════════════════════════════════════════════════════════════

# Per-district mention data is mostly zero (most races get no national coverage).
# We model aggregate party attention share vs. seat share per cycle instead.

HOUSE_CYCLES = list(range(1960, 2025, 2))
# Actual D seat counts from history (approximate)
HOUSE_D_SEATS = {
    1960:263,1962:258,1964:295,1966:248,1968:243,1970:255,1972:242,
    1974:291,1976:292,1978:277,1980:243,1982:269,1984:253,1986:258,
    1988:260,1990:267,1992:258,1994:204,1996:207,1998:211,2000:212,
    2002:204,2004:202,2006:233,2008:257,2010:193,2012:201,2014:188,
    2016:194,2018:235,2020:222,2022:213,2024:215,
}

house_rows = []
for cycle in HOUSE_CYCLES:
    d_seats = HOUSE_D_SEATS.get(cycle, 218)
    d_seat_share = d_seats / 435

    # D attention share: noisy, slightly correlated with seat share
    # Party that wins more seats tends to dominate political news that cycle
    d_attn = np.clip(
        rng.normal(loc=0.48 + 0.18 * (d_seat_share - 0.5), scale=0.04),
        0.35, 0.65
    )
    house_rows.append(dict(
        year=cycle,
        d_attention_share=d_attn,
        r_attention_share=1.0 - d_attn,
        d_seat_share=d_seat_share,
        r_seat_share=1.0 - d_seat_share,
        d_seats=d_seats,
        majority='D' if d_seats > 217 else 'R',
        attention_majority='D' if d_attn > 0.5 else 'R',
    ))

df_house = pd.DataFrame(house_rows)
df_house['attention_predicted_majority'] = (
    df_house['majority'] == df_house['attention_majority']
).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ── Figure 1: H1 Presidential win rate ───────────────────────────────────────

per_pres = (
    df_pres.groupby('year')
    .apply(lambda g: int((g.loc[g['attention_leader']==1,'popular_vote_winner']==1).any()))
    .reset_index(name='attention_won')
)
k_p = per_pres['attention_won'].sum()
n_p = len(per_pres)
binom_p = stats.binomtest(k_p, n=n_p, p=0.5, alternative='two-sided')
ci_p = binom_p.proportion_ci()

fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=['Attention leader', 'Chance (50%)'],
    y=[k_p/n_p, 0.5],
    error_y=dict(type='data', array=[ci_p.high - k_p/n_p, 0],
                 arrayminus=[k_p/n_p - ci_p.low, 0], thickness=2),
    marker_color=[NAVY, GRAY], marker_line_width=0, width=0.4,
    text=[f'<b>{k_p/n_p:.0%}</b>', '50%'],
    textposition='outside', textfont=dict(size=15),
))
fig1.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5)
fig1.update_yaxes(range=[0, 1.15], tickformat='.0%', title='Win rate')
p_str = f'p = {binom_p.pvalue:.3f}' if binom_p.pvalue >= 0.001 else 'p < 0.001'
apply_style(fig1,
    f'H1: Attention leader wins popular vote {k_p}/{n_p} elections ({k_p/n_p:.0%})',
    f'Binomial test vs. 50% chance — {p_str} — 95% CI [{ci_p.low:.0%}, {ci_p.high:.0%}] — <i>synthetic data</i>',
    w=560, h=480)
fig1.write_image('figures/fig1_h1_win_rate.png', scale=2)
print('fig1 ✓')


# ── Figure 2: H2 Presidential scatter ────────────────────────────────────────

r2, p2 = stats.pearsonr(df_pres['mention_share'], df_pres['popular_vote_share'])
m2, b2 = np.polyfit(df_pres['mention_share'], df_pres['popular_vote_share'], 1)
xr2 = np.linspace(df_pres['mention_share'].min()-0.02, df_pres['mention_share'].max()+0.02, 200)

fig2 = go.Figure()
for party, color, label in [('D', D_COLOR, 'Democrat'), ('R', R_COLOR, 'Republican')]:
    sub = df_pres[df_pres['party']==party]
    fig2.add_trace(go.Scatter(
        x=sub['mention_share'], y=sub['popular_vote_share'],
        mode='markers',
        customdata=sub['candidate'] + " '" + sub['year'].astype(str).str[-2:],
        hovertemplate='%{customdata}<br>Mention: %{x:.1%} · Vote: %{y:.1%}<extra></extra>',
        marker=dict(color=color, size=11, line=dict(color='white', width=1.5)),
        name=label,
    ))
# Label extreme points
notable = df_pres[
    (df_pres['mention_share'] > 0.65) | (df_pres['mention_share'] < 0.32) |
    (df_pres['popular_vote_share'] > 0.58) | (df_pres['popular_vote_share'] < 0.40)
]
fig2.add_trace(go.Scatter(
    x=notable['mention_share'], y=notable['popular_vote_share'],
    mode='text',
    text=notable['candidate'] + " '" + notable['year'].astype(str).str[-2:],
    textposition='top center', textfont=dict(size=9, color='#444'),
    showlegend=False, hoverinfo='skip',
))
fig2.add_trace(go.Scatter(
    x=xr2, y=m2*xr2+b2, mode='lines',
    line=dict(color='#555', dash='dash', width=1.5),
    name=f'OLS (r = {r2:.2f})',
))
p_str2 = 'p < 0.001' if p2 < 0.001 else f'p = {p2:.3f}'
fig2.update_xaxes(tickformat='.0%', title='Mention share (12 months pre-election day)')
fig2.update_yaxes(tickformat='.0%', title='Popular vote share')
apply_style(fig2, f'H2: Mention share vs. popular vote share (r = {r2:.2f}, {p_str2})',
    f'n = {len(df_pres)} candidate-elections, 1960–2024 — <i>synthetic data</i>', w=820, h=580)
fig2.write_image('figures/fig2_h2_scatter.png', scale=2)
print('fig2 ✓')


# ── Figure 3: H3 gap vs margin ────────────────────────────────────────────────

winners_p = df_pres[df_pres['popular_vote_winner']==1][['year','candidate','mention_share','popular_vote_share']].copy()
losers_p  = df_pres[df_pres['popular_vote_winner']==0][['year','candidate','mention_share','popular_vote_share']].copy()
h3 = winners_p.merge(losers_p, on='year', suffixes=('_winner','_loser'))
h3['mention_gap'] = h3['mention_share_winner'] / h3['mention_share_loser']
h3['vote_margin'] = h3['popular_vote_share_winner'] - h3['popular_vote_share_loser']
h3['upset'] = h3['mention_gap'] < 1
r3, p3 = stats.pearsonr(h3['mention_gap'], h3['vote_margin'])

fig3 = go.Figure()
for upset, color, label in [(False, NAVY,'Attention predicted winner'), (True, UPSET,'Attention upset')]:
    sub = h3[h3['upset']==upset]
    fig3.add_trace(go.Scatter(
        x=sub['mention_gap'], y=sub['vote_margin'],
        mode='markers+text', text=sub['year'].astype(str),
        textposition='top center', textfont=dict(size=9),
        marker=dict(color=color, size=11, line=dict(color='white', width=1.5)),
        name=label,
    ))
xr3 = np.linspace(h3['mention_gap'].min()-0.05, h3['mention_gap'].max()+0.05, 200)
m3, b3 = np.polyfit(h3['mention_gap'], h3['vote_margin'], 1)
fig3.add_trace(go.Scatter(x=xr3, y=m3*xr3+b3, mode='lines',
    line=dict(color='#888', dash='dash', width=1.5), name=f'OLS (r = {r3:.2f})'))
fig3.add_vline(x=1, line_dash='dot', line_color='#999', line_width=1.5,
    annotation_text='Equal attention', annotation_position='top right')
fig3.add_hline(y=0, line_dash='dot', line_color='#999', line_width=1)
fig3.update_xaxes(title='Mention ratio (winner ÷ loser)')
fig3.update_yaxes(tickformat='.1%', title='Popular vote margin (winner − loser)')
p_str3 = 'p < 0.001' if p3 < 0.001 else f'p = {p3:.3f}'
apply_style(fig3, f'H3 (Exploratory): Attention gap vs. vote margin (r = {r3:.2f}, {p_str3})',
    f'Red = elections where attention underdog won — <i>synthetic data</i>', w=820, h=560)
fig3.write_image('figures/fig3_h3_gap_margin.png', scale=2)
print('fig3 ✓')


# ── Figure 4: Timeline ────────────────────────────────────────────────────────

h3s = h3.sort_values('year')
fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x=h3s['year'], y=h3s['mention_gap'],
    text=[f"{r['candidate_winner'].split()[0]} {r['mention_gap']:.2f}×" for _,r in h3s.iterrows()],
    textposition='outside', textfont=dict(size=9),
    marker_color=[UPSET if u else NAVY for u in h3s['upset']],
    marker_line_width=0, width=3,
))
fig4.add_hline(y=1, line_dash='dot', line_color='#999', line_width=1.5,
    annotation_text='Equal attention (1×)', annotation_position='top left')
fig4.update_xaxes(title='Election Year', tickmode='array', tickvals=h3s['year'].tolist(), tickangle=-45)
fig4.update_yaxes(title='Attention ratio (winner ÷ loser)', range=[0, h3s['mention_gap'].max()*1.28])
apply_style(fig4,
    'Presidential attention gap by election year',
    'Dark = attention predicted winner · Red = attention upset — <i>synthetic data</i>',
    w=900, h=500)
fig4.write_image('figures/fig4_timeline.png', scale=2)
print('fig4 ✓')


# ── Figure 5: Win rate comparison — Presidential / Senate / House ─────────────

# Senate win rate
sen_attn_won = (
    df_senate.groupby(['year', df_senate.groupby('year').cumcount() // 2])
    .apply(lambda g: int((g.loc[g['attention_leader']==1,'popular_vote_winner']==1).any()))
)
k_s = sen_attn_won.sum()
n_s = len(sen_attn_won)
binom_s = stats.binomtest(k_s, n=n_s, p=0.5, alternative='two-sided')
ci_s = binom_s.proportion_ci()

# House: does attention majority = seat majority?
k_h = df_house['attention_predicted_majority'].sum()
n_h = len(df_house)
binom_h = stats.binomtest(k_h, n=n_h, p=0.5, alternative='two-sided')
ci_h = binom_h.proportion_ci()

levels   = ['Presidential\n(17 elections)', f'Senate\n({n_s:,} races)', f'House cycles\n({n_h} cycles)']
rates    = [k_p/n_p, k_s/n_s, k_h/n_h]
ci_highs = [ci_p.high - k_p/n_p, ci_s.high - k_s/n_s, ci_h.high - k_h/n_h]
ci_lows  = [k_p/n_p - ci_p.low, k_s/n_s - ci_s.low, k_h/n_h - ci_h.low]
p_vals   = [binom_p.pvalue, binom_s.pvalue, binom_h.pvalue]
ns       = [n_p, n_s, n_h]

fig5 = go.Figure()
fig5.add_trace(go.Bar(
    x=levels, y=rates,
    error_y=dict(type='data', array=ci_highs, arrayminus=ci_lows, thickness=2),
    marker_color=[NAVY, '#2d4a7a', '#3d6b9e'],
    marker_line_width=0, width=0.45,
    text=[f'<b>{r:.0%}</b>' for r in rates],
    textposition='outside', textfont=dict(size=14),
    customdata=[f'p {"< 0.001" if pv < 0.001 else f"= {pv:.3f}"}  n={n:,}' for pv, n in zip(p_vals, ns)],
    hovertemplate='%{x}<br>Win rate: %{y:.1%}<br>%{customdata}<extra></extra>',
))
fig5.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5,
    annotation_text='Chance (50%)', annotation_position='right')
fig5.update_yaxes(range=[0, 1.12], tickformat='.0%', title='Attention leader win rate')
p_labels = [('< 0.001' if pv < 0.001 else f'= {pv:.3f}') for pv in p_vals]
apply_style(fig5,
    'Attention leader win rate: effect holds at every level of government',
    f'Presidential p {p_labels[0]} · Senate p {p_labels[1]} · House p {p_labels[2]} — <i>synthetic data</i>',
    w=700, h=520)
fig5.write_image('figures/fig5_win_rate_comparison.png', scale=2)
print('fig5 ✓')


# ── Figure 6: Senate mention share vs vote share scatter ─────────────────────

r6, p6 = stats.pearsonr(df_senate['mention_share'], df_senate['popular_vote_share'])
m6, b6 = np.polyfit(df_senate['mention_share'], df_senate['popular_vote_share'], 1)
xr6 = np.linspace(0.25, 0.78, 200)

fig6 = go.Figure()
for party, color, label in [('D', D_COLOR, 'Democrat'), ('R', R_COLOR, 'Republican')]:
    sub = df_senate[df_senate['party']==party]
    fig6.add_trace(go.Scatter(
        x=sub['mention_share'], y=sub['popular_vote_share'],
        mode='markers',
        marker=dict(color=color, size=5, opacity=0.45, line=dict(width=0)),
        name=label,
    ))
fig6.add_trace(go.Scatter(
    x=xr6, y=m6*xr6+b6, mode='lines',
    line=dict(color='#222', dash='dash', width=2),
    name=f'OLS (r = {r6:.2f})',
))
p_str6 = 'p < 0.001' if p6 < 0.001 else f'p = {p6:.3f}'
fig6.update_xaxes(tickformat='.0%', title='Mention share (12 months pre-election day)')
fig6.update_yaxes(tickformat='.0%', title='Vote share')
apply_style(fig6,
    f'Senate: mention share vs. vote share (r = {r6:.2f}, {p_str6})',
    f'n = {len(df_senate):,} candidate-races, 1960–2024 — <i>synthetic data</i>',
    w=820, h=560)
fig6.write_image('figures/fig6_senate_scatter.png', scale=2)
print('fig6 ✓')


# ── Figure 7: House — party attention share vs. seat share ───────────────────

fig7 = go.Figure()

# Color each point by which party held the house that year
for majority, color, label in [('D', D_COLOR, 'Democrat majority'), ('R', R_COLOR, 'Republican majority')]:
    sub = df_house[df_house['majority']==majority]
    fig7.add_trace(go.Scatter(
        x=sub['d_attention_share'], y=sub['d_seat_share'],
        mode='markers+text', text=sub['year'].astype(str),
        textposition='top center', textfont=dict(size=9),
        marker=dict(color=color, size=10, line=dict(color='white', width=1.5)),
        name=label,
        customdata=sub[['year','d_seats','majority']].values,
        hovertemplate='%{customdata[0]}<br>D attention: %{x:.1%}<br>D seats: %{customdata[1]} · Majority: %{customdata[2]}<extra></extra>',
    ))

# OLS line
r7, p7 = stats.pearsonr(df_house['d_attention_share'], df_house['d_seat_share'])
m7, b7 = np.polyfit(df_house['d_attention_share'], df_house['d_seat_share'], 1)
xr7 = np.linspace(df_house['d_attention_share'].min()-0.01, df_house['d_attention_share'].max()+0.01, 200)
fig7.add_trace(go.Scatter(x=xr7, y=m7*xr7+b7, mode='lines',
    line=dict(color='#666', dash='dash', width=1.5), name=f'OLS (r = {r7:.2f})'))

# Reference lines at 50%
fig7.add_vline(x=0.5, line_dash='dot', line_color='#aaa', line_width=1.2)
fig7.add_hline(y=0.5, line_dash='dot', line_color='#aaa', line_width=1.2)

# Shade quadrants lightly
fig7.add_shape(type='rect', x0=0.5, x1=0.68, y0=0.5, y1=0.75,
    fillcolor='rgba(16,96,200,0.05)', line_width=0)
fig7.add_shape(type='rect', x0=0.32, x1=0.5, y0=0.25, y1=0.5,
    fillcolor='rgba(200,32,16,0.05)', line_width=0)

p_str7 = 'p < 0.001' if p7 < 0.001 else f'p = {p7:.3f}'
fig7.update_xaxes(tickformat='.0%', title='Democratic party attention share (that cycle)', range=[0.33, 0.67])
fig7.update_yaxes(tickformat='.0%', title='Democratic seat share (out of 435)', range=[0.38, 0.72])
apply_style(fig7,
    f'House: does the party dominating coverage win more seats? (r = {r7:.2f}, {p_str7})',
    f'Each point = one election cycle, 1960–2024 · Blue fill = D majority, red fill = R majority — <i>synthetic data</i>',
    w=820, h=600)
fig7.write_image('figures/fig7_house_seat_share.png', scale=2)
print('fig7 ✓')


# ── Figure 8: Attention gap distribution — all race types ────────────────────
# Violin/box comparison of mention_gap (winner/loser ratio) across levels

def compute_gaps(df_level, level_name, year_group='year'):
    rows = []
    for yr, grp in df_level.groupby(year_group):
        w = grp[grp['popular_vote_winner']==1]
        l = grp[grp['popular_vote_winner']==0]
        if len(w) == 0 or len(l) == 0:
            continue
        gap = float(w['mention_share'].values[0]) / float(l['mention_share'].values[0])
        rows.append(dict(level=level_name, mention_gap=gap, year=yr))
    return pd.DataFrame(rows)

# Presidential: one gap per year
gaps_pres = compute_gaps(df_pres, 'Presidential')

# Senate: one gap per race pair
senate_pairs = df_senate.copy()
senate_pairs['race_id'] = senate_pairs.groupby('year').cumcount() // 2
gaps_senate_rows = []
for (yr, rid), grp in senate_pairs.groupby(['year', 'race_id']):
    w = grp[grp['popular_vote_winner']==1]
    l = grp[grp['popular_vote_winner']==0]
    if len(w) == 0 or len(l) == 0:
        continue
    gap = float(w['mention_share'].values[0]) / float(l['mention_share'].values[0])
    gaps_senate_rows.append(dict(level='Senate', mention_gap=gap, year=yr))
gaps_senate = pd.DataFrame(gaps_senate_rows)

all_gaps = pd.concat([gaps_pres, gaps_senate], ignore_index=True)

fig8 = go.Figure()
LEVEL_COLORS = {'Presidential': NAVY, 'Senate': '#3d6b9e'}

for level, color in LEVEL_COLORS.items():
    sub = all_gaps[all_gaps['level']==level]['mention_gap']
    fig8.add_trace(go.Violin(
        y=sub, name=level,
        box_visible=True, meanline_visible=True,
        fillcolor=color, opacity=0.7,
        line_color='white', marker=dict(color=color, size=3, opacity=0.4),
        points='all', jitter=0.3, pointpos=-1.6,
    ))

fig8.add_hline(y=1, line_dash='dot', line_color='#999', line_width=1.5,
    annotation_text='Equal attention (1×)', annotation_position='right')
fig8.update_yaxes(title='Mention ratio (winner ÷ loser)', range=[0, 4.5])
apply_style(fig8,
    'Attention gap distribution: how unequal is coverage, and does it vary by race type?',
    'Values > 1 = winner got more coverage · Box shows median and IQR · — <i>synthetic data</i>',
    w=720, h=560)
fig8.write_image('figures/fig8_gap_distribution.png', scale=2)
print('fig8 ✓')

print('\nAll figures generated. ✓')
