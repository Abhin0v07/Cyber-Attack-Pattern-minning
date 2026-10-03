"""Shared look & feel: an editorial 'threat report' style: paper background, ink type, one signal colour."""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

INK, MUTED, RULE, PAPER, CARD, ACCENT = "#1d1c1a", "#6b6760", "#d9d3c7", "#f5f2ec", "#fffdf9", "#d9480f"
PALETTE = ["#d9480f", "#1c7ed6", "#2f9e44", "#f59f00", "#ae3ec9", "#495057", "#0ca678", "#e64980", "#74b816",
           "#868e96"]
ATTACK_COLORS = {"Normal": "#2f9e44", "DoS": "#d9480f", "Probe": "#f59f00", "R2L": "#1c7ed6", "U2R": "#ae3ec9",
                 "Botnet": "#495057",
                 "Low": "#2f9e44", "Medium": "#f59f00", "High": "#e8590c", "Critical": "#c92a2a"}
SEQ = [[0, "#fbf8f3"], [0.45, "#f8c9a0"], [1, "#e8590c"]]

_template = go.layout.Template(pio.templates["simple_white"])
_template.layout.update(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, system-ui, sans-serif", color=INK, size=13),
    title=dict(font=dict(family="Space Grotesk, sans-serif", size=15, color=INK), x=0, xanchor="left"),
    colorway=PALETTE, margin=dict(l=10, r=10, t=44, b=10),
    xaxis=dict(gridcolor="#e7e1d6", linecolor=RULE, tickcolor=RULE, showgrid=False),
    yaxis=dict(gridcolor="#e7e1d6", linecolor=RULE, tickcolor=RULE, showgrid=True),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
    hoverlabel=dict(bgcolor=CARD, font_color=INK, bordercolor=RULE),
)
pio.templates["report"] = _template
pio.templates.default = "report"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
.stApp {{ background: {PAPER}; color: {INK}; }}
header[data-testid="stHeader"] {{ background: {PAPER}; border-bottom: 1px solid {RULE}; }}
.block-container {{ max-width: 1180px; padding-top: 4.6rem; padding-left: 16px; padding-right: 16px; }}
h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif !important; color: {INK}; letter-spacing: -.02em; }}
code {{ color: {ACCENT}; background: #f1ebe1; }}

.mast {{ border-bottom: 2px solid {INK}; padding: 4px 0 16px; margin-bottom: 20px; }}
.mast .tag {{ font-family: 'IBM Plex Mono', monospace; font-size: .74rem; color: {ACCENT}; text-transform: uppercase;
             letter-spacing: .1em; }}
.mast h1 {{ font-size: clamp(1.7rem, 3.4vw, 2.5rem); margin: .25rem 0 .45rem; line-height: 1.08; padding: 0; }}
.mast p {{ color: {MUTED}; max-width: 780px; margin: 0; font-size: 1rem; }}

.lead {{ font-family: 'Space Grotesk', sans-serif; font-size: clamp(1.25rem, 2.4vw, 1.7rem); line-height: 1.3;
         max-width: 900px; margin: 0 0 18px; color: {INK}; }}
.lead b {{ color: {ACCENT}; }}

.strip {{ display: flex; flex-wrap: wrap; border-top: 1px solid {RULE}; border-bottom: 1px solid {RULE}; margin: 0 0 22px; }}
.strip .s {{ flex: 1 1 160px; padding: 12px 18px 12px 0; margin-right: 18px; border-right: 1px solid {RULE}; }}
.strip .s:last-child {{ border-right: none; margin-right: 0; }}
.strip .l {{ font-family: 'IBM Plex Mono', monospace; font-size: .7rem; color: {MUTED}; text-transform: uppercase;
            letter-spacing: .08em; }}
.strip .v {{ font-family: 'IBM Plex Mono', monospace; font-size: clamp(1.15rem, 2vw, 1.6rem); font-weight: 600;
            color: {INK}; margin: 2px 0; line-height: 1.2; }}
.strip .s.hot .v {{ color: {ACCENT}; }}
.strip .d {{ font-size: .8rem; color: {MUTED}; }}

.note {{ display: grid; grid-template-columns: 96px 1fr; gap: 14px; border: 1px dashed #bdb5a6; background: {CARD};
         padding: 12px 14px; margin: 4px 0 18px; font-size: .92rem; line-height: 1.55; }}
.note .k {{ font-family: 'IBM Plex Mono', monospace; font-size: .7rem; color: {ACCENT}; text-transform: uppercase;
           letter-spacing: .1em; padding-top: 3px; }}
.note b {{ color: {INK}; }}
@media (max-width: 640px) {{ .note {{ grid-template-columns: 1fr; gap: 4px; }} }}

.sec {{ display: flex; align-items: baseline; gap: 12px; border-top: 1px solid {INK}; padding-top: 10px; margin: 26px 0 8px; }}
.sec .n {{ font-family: 'IBM Plex Mono', monospace; color: {ACCENT}; font-size: .85rem; }}
.sec h3 {{ margin: 0; padding: 0; font-size: 1.25rem; }}
.sec .sub {{ color: {MUTED}; font-size: .88rem; margin-left: auto; }}

.rule {{ font-family: 'IBM Plex Mono', monospace; font-size: .8rem; border-bottom: 1px solid {RULE}; padding: 8px 2px;
         display: flex; flex-wrap: wrap; gap: 4px 8px; color: {INK}; }}
.rule .if, .rule .then {{ font-weight: 600; color: {ACCENT}; }}
.rule .m {{ color: {MUTED}; margin-left: auto; }}

.find {{ counter-reset: f; list-style: none; padding: 0; margin: 0; }}
.find li {{ counter-increment: f; display: grid; grid-template-columns: 38px 1fr; padding: 10px 0;
            border-bottom: 1px solid {RULE}; }}
.find li::before {{ content: counter(f, decimal-leading-zero); font-family: 'IBM Plex Mono', monospace; color: {ACCENT}; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {RULE}; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'IBM Plex Mono', monospace; font-size: .8rem; padding: 8px 14px; }}
div[data-testid="stMetric"] {{ border-left: 3px solid {ACCENT}; padding: 6px 12px; background: {CARD}; }}
div[data-testid="stExpander"] details {{ background: {CARD}; }}
.foot {{ font-family: 'IBM Plex Mono', monospace; font-size: .72rem; color: {MUTED}; border-top: 1px solid {RULE};
         margin-top: 40px; padding-top: 10px; display: flex; flex-wrap: wrap; gap: 6px 22px; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(tag, title, subtitle):
    st.markdown(f'<div class="mast"><div class="tag">{tag}</div><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def kpis(items):
    """items: list of (label, value, sub, color); color 'red' marks the value as highlighted."""
    cells = "".join(f'<div class="s {"hot" if color == "red" else ""}"><div class="l">{label}</div>'
                    f'<div class="v">{value}</div><div class="d">{sub}</div></div>'
                    for label, value, sub, color in items)
    st.markdown(f'<div class="strip">{cells}</div>', unsafe_allow_html=True)


def theory(html, key="Theory"):
    st.markdown(f'<div class="note"><div class="k">{key}</div><div>{html}</div></div>', unsafe_allow_html=True)


def section(num, title, sub=""):
    st.markdown(f'<div class="sec"><span class="n">{num}</span><h3>{title}</h3><span class="sub">{sub}</span></div>',
                unsafe_allow_html=True)


def rule_card(antecedent, consequent, extra=""):
    st.markdown(f'<div class="rule"><span class="if">IF</span><span>{antecedent}</span><span class="then">THEN</span>'
                f'<span>{consequent}</span><span class="m">{extra}</span></div>', unsafe_allow_html=True)


def footer(text_items):
    st.markdown('<div class="foot">' + "".join(f"<span>{t}</span>" for t in text_items) + "</div>",
                unsafe_allow_html=True)


def show(fig, height=None):
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch")
