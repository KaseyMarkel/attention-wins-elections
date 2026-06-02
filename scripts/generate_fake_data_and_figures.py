"""
Generates synthetic mention data for visual prototyping.
Run BEFORE real data collection to validate figure layouts.
Delete or archive once real data is in place.

Usage: python scripts/generate_fake_data_and_figures.py
"""

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

np.random.seed(42)

# ── Fake data generation ──────────────────────────────────────────────────────

ELECTIONS = [
    (1960, 'Kennedy', 'Nixon',    'D', 'R', 0),
    (1964, 'Johnson', 'Goldwater','D', 'R', 1),
    (1968, 'Nixon',   'Humphrey', 'R', 'D', 0),
    (1972, 'Nixon',   'McGovern', 'R', 'D', 1),
    (1976, 'Carter',  'Ford',     'D', 'R', 0),
    (1980, 'Reagan',  'Carter',   'R', 'D', 0),
    (1984, 'Reagan',  'Mondale',  'R', 'D', 1),
    (1988, 'Bush',    'Dukakis',  'R', 'D', 0),
    (1992, 'Clinton', 'Bush',     'D', 'R', 0),
    (1996, 'Clinton', 'Dole',     'D', 'R', 1),
    (2000, 'Bush',    'Gore',     'R', 'D', 0),
    (2004, 'Bush',    'Kerry',    'R', 'D', 1),
    (2008, 'Obama',   'McCain',   'D', 'R', 0),
    (2012, 'Obama',   'Romney',   'D', 'R', 1),
    (2016, 'Trump',   'Clinton',  'R', 'D', 0),
    (2020, 'Biden',   'Trump',    'D', 'R', 0),
    (2024, 'Trump',   'Harris',   'R', 'D', 0),
]

vote = pd.read_csv('data/processed/election_results.csv')

rows = []
for year, winner_name, loser_name, winner_party, loser_party, incumbent in ELECTIONS:
    # Winner gets more attention ~75% of the time, with realistic variance
    winner_mention_raw = np.random.uniform(0.52, 0.72)
    loser_mention_raw  = 1.0 - winner_mention_raw

    winner_vote = vote.loc[(vote.year==year) & (vote.candidate==winner_name), 'popular_vote_share'].values[0]
    loser_vote  = vote.loc[(vote.year==year) & (vote.candidate==loser_name),  'popular_vote_share'].values[0]

    rows += [
        dict(year=year, candidate=winner_name, party=winner_party,
             mention_share=winner_mention_raw, popular_vote_share=winner_vote,
             popular_vote_winner=1, attention_leader=1, incumbent_running=incumbent, source='FAKE'),
        dict(year=year, candidate=loser_name, party=loser_party,
             mention_share=loser_mention_raw, popular_vote_share=loser_vote,
             popular_vote_winner=0, attention_leader=0, incumbent_running=0, source='FAKE'),
    ]

# Inject a few upsets where loser had MORE attention (makes it more interesting)
upset_years = [1968, 2000, 2016]
for row in rows:
    if row['year'] in upset_years:
        row['attention_leader'] = 1 - row['attention_leader']
        row['mention_share']    = 1.0 - row['mention_share']

df = pd.DataFrame(rows)
Path('data/processed').mkdir(exist_ok=True)
df.to_csv('data/processed/mention_share_FAKE.csv', index=False)
print(f"Generated {len(df)} rows of fake data ({len(df)//2} elections, 3 injected upsets)")


# ── Shared style ──────────────────────────────────────────────────────────────

FONT = 'Georgia, serif'
BG   = '#FAFAFA'
GRID = '#E8E8E8'

def apply_style(fig, title, w=800, h=500):
    fig.update_layout(
        title=dict(text=title, font=dict(family=FONT, size=18, color='#111')),
        font=dict(family=FONT, size=13),
        plot_bgcolor=BG,
        paper_bgcolor='white',
        width=w, height=h,
        margin=dict(l=70, r=40, t=70, b=60),
        xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False),
    )
    return fig


# ── Figure 1: H1 — Win rate bar ───────────────────────────────────────────────

per_election = (
    df.groupby('year')
    .apply(lambda g: int((g.loc[g['attention_leader']==1, 'popular_vote_winner']==1).any()))
    .reset_index(name='attention_won')
)
k = per_election['attention_won'].sum()
n = len(per_election)
binom = stats.binomtest(k, n=n, p=0.5, alternative='two-sided')
ci_lo = binom.proportion_ci().low
ci_hi = binom.proportion_ci().high

fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=['Attention leader', 'Chance (50%)'],
    y=[k/n, 0.5],
    error_y=dict(type='data', array=[ci_hi - k/n, 0], arrayminus=[k/n - ci_lo, 0], thickness=2),
    marker_color=['#1a1a2e', '#CCCCCC'],
    marker_line_width=0,
    width=0.4,
    text=[f'<b>{k/n:.0%}</b>', '50%'],
    textposition='outside',
    textfont=dict(size=15),
))
fig1.add_hline(y=0.5, line_dash='dot', line_color='#999', line_width=1.5)
fig1.update_yaxes(range=[0, 1.15], tickformat='.0%', title='Win rate')
fig1.update_xaxes(title='')
p_str = f'p = {binom.pvalue:.3f}' if binom.pvalue >= 0.001 else 'p < 0.001'
apply_style(fig1,
    f'H1: Attention leader wins popular vote {k}/{n} elections ({k/n:.0%})<br>'
    f'<sup>Binomial test vs. 50% chance — {p_str} — 95% CI [{ci_lo:.0%}, {ci_hi:.0%}]</sup>',
    w=560, h=480)
fig1.write_image('figures/fig1_h1_win_rate.png', scale=2)
print('Saved figures/fig1_h1_win_rate.png')


# ── Figure 2: H2 — Mention share vs. vote share scatter ──────────────────────

r, p = stats.pearsonr(df['mention_share'], df['popular_vote_share'])
m, b = np.polyfit(df['mention_share'], df['popular_vote_share'], 1)
x_range = np.linspace(df['mention_share'].min() - 0.02, df['mention_share'].max() + 0.02, 200)

PARTY_COLOR = {'D': '#1060C8', 'R': '#C82010'}
PARTY_LABEL = {'D': 'Democrat', 'R': 'Republican'}

fig2 = go.Figure()

for party in ['D', 'R']:
    sub = df[df['party'] == party]
    fig2.add_trace(go.Scatter(
        x=sub['mention_share'], y=sub['popular_vote_share'],
        mode='markers',
        customdata=sub['candidate'] + " '" + sub['year'].astype(str).str[-2:],
        hovertemplate='%{customdata}<br>Mention share: %{x:.1%}<br>Vote share: %{y:.1%}<extra></extra>',
        marker=dict(color=PARTY_COLOR[party], size=11, line=dict(color='white', width=1.5)),
        name=PARTY_LABEL[party],
    ))

# Add selective labels only for notable/extreme points
notable = df[
    (df['mention_share'] > 0.65) |
    (df['mention_share'] < 0.32) |
    (df['popular_vote_share'] > 0.58) |
    (df['popular_vote_share'] < 0.395)
].copy()
fig2.add_trace(go.Scatter(
    x=notable['mention_share'], y=notable['popular_vote_share'],
    mode='text',
    text=notable['candidate'] + " '" + notable['year'].astype(str).str[-2:],
    textposition='top center',
    textfont=dict(size=9, color='#444'),
    showlegend=False,
    hoverinfo='skip',
))

fig2.add_trace(go.Scatter(
    x=x_range, y=m * x_range + b,
    mode='lines',
    line=dict(color='#555', dash='dash', width=1.5),
    name=f'OLS (r = {r:.2f})',
    showlegend=True,
))

p_str2 = f'p = {p:.3f}' if p >= 0.001 else 'p < 0.001'
fig2.update_xaxes(tickformat='.0%', title='Mention share (12 months pre-election day)')
fig2.update_yaxes(tickformat='.0%', title='Popular vote share')
apply_style(fig2,
    f'H2: Mention share vs. popular vote share (r = {r:.2f}, {p_str2})<br>'
    f'<sup>n = {len(df)} candidate-elections, 1960–2024 — <i>synthetic data</i></sup>',
    w=820, h=580)
fig2.write_image('figures/fig2_h2_scatter.png', scale=2)
print('Saved figures/fig2_h2_scatter.png')


# ── Figure 3: H3 — Mention gap vs. vote margin ───────────────────────────────

winners = df[df['popular_vote_winner']==1][['year','candidate','mention_share','popular_vote_share']].copy()
losers  = df[df['popular_vote_winner']==0][['year','candidate','mention_share','popular_vote_share']].copy()
h3 = winners.merge(losers, on='year', suffixes=('_winner','_loser'))
h3['mention_gap']  = h3['mention_share_winner'] / h3['mention_share_loser']
h3['vote_margin']  = h3['popular_vote_share_winner'] - h3['popular_vote_share_loser']
h3['upset'] = h3['mention_gap'] < 1

r3, p3 = stats.pearsonr(h3['mention_gap'], h3['vote_margin'])

fig3 = go.Figure()
for is_upset, color, label in [(False,'#1a1a2e','Attention predicted winner'), (True,'#E05050','Attention upset')]:
    sub = h3[h3['upset']==is_upset]
    fig3.add_trace(go.Scatter(
        x=sub['mention_gap'], y=sub['vote_margin'],
        mode='markers+text',
        text=sub['year'].astype(str),
        textposition='top center',
        textfont=dict(size=9),
        marker=dict(color=color, size=11, line=dict(color='white', width=1.5)),
        name=label,
    ))

xr = np.linspace(h3['mention_gap'].min()-0.05, h3['mention_gap'].max()+0.05, 200)
m3, b3 = np.polyfit(h3['mention_gap'], h3['vote_margin'], 1)
fig3.add_trace(go.Scatter(
    x=xr, y=m3*xr+b3, mode='lines',
    line=dict(color='#888', dash='dash', width=1.5),
    name=f'OLS (r = {r3:.2f})',
))
fig3.add_vline(x=1, line_dash='dot', line_color='#999', line_width=1.5,
               annotation_text='Equal attention', annotation_position='top right')
fig3.add_hline(y=0, line_dash='dot', line_color='#999', line_width=1.5)
fig3.update_xaxes(title='Mention ratio (winner mentions ÷ loser mentions)')
fig3.update_yaxes(tickformat='.1%', title='Popular vote margin (winner − loser)')
p_str3 = f'p = {p3:.3f}' if p3 >= 0.001 else 'p < 0.001'
apply_style(fig3,
    f'H3 (Exploratory): Attention gap vs. vote margin (r = {r3:.2f}, {p_str3})<br>'
    f'<sup>Red = elections where underdog in attention won — <i>synthetic data</i></sup>',
    w=820, h=560)
fig3.write_image('figures/fig3_h3_gap_margin.png', scale=2)
print('Saved figures/fig3_h3_gap_margin.png')


# ── Figure 4: Timeline of attention gaps ─────────────────────────────────────

h3s = h3.sort_values('year')
colors = ['#E05050' if u else '#1a1a2e' for u in h3s['upset']]
labels = [
    f"{row['candidate_winner']} {row['mention_gap']:.2f}×"
    for _, row in h3s.iterrows()
]

fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x=h3s['year'], y=h3s['mention_gap'],
    text=labels,
    textposition='outside',
    textfont=dict(size=9),
    marker_color=colors,
    marker_line_width=0,
    width=3,
))
fig4.add_hline(y=1, line_dash='dot', line_color='#999', line_width=1.5,
               annotation_text='Equal attention (1×)', annotation_position='top left')
fig4.update_xaxes(
    title='Election Year',
    tickmode='array', tickvals=h3s['year'].tolist(),
    tickangle=-45,
)
fig4.update_yaxes(title='Attention ratio (winner ÷ loser)', range=[0, h3s['mention_gap'].max()*1.25])
apply_style(fig4,
    'Attention gap by election: how dominant was the winner in media mentions?<br>'
    '<sup>Dark = attention predicted winner · Red = attention upset — <i>synthetic data</i></sup>',
    w=900, h=500)
fig4.write_image('figures/fig4_timeline.png', scale=2)
print('Saved figures/fig4_timeline.png')

print('\nAll figures generated. ✓')
