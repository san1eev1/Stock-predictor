"""Look and feel: the night desk. Deep green-graphite, one marigold accent, jade gains and
coral losses on numbers only; Bricolage Grotesque for headings and figures, Geist for text,
tabular digits everywhere. Dark only (no theme switching, no reload on first visit)."""

from __future__ import annotations

import html

import streamlit as st

# Muted on purpose: nothing glows. Sand accent, sage gains, clay losses.
UP, DOWN, NEUTRAL = "#8DBA9E", "#D08C80", "#C4AB7C"

_NIGHT = {"bg": "#111413", "surface": "#171B1A", "raised": "#1D2220", "rule": "#283029",
          "ink": "#E3E6E2", "muted": "#8E9792", "up": UP, "down": DOWN, "signal": NEUTRAL}
PALETTE = {True: _NIGHT, False: _NIGHT}      # one theme; kept as a mapping for callers
# Categories in charts: muted, and never red / green (those mean loss / gain here).
CATEGORIES = ["#C4AB7C", "#7FA3B5", "#A59CC6", "#7FAFA5", "#D6C8A4", "#8E9BB8", "#C39CAE",
              "#A9B27F", "#B8957A", "#86BFC0", "#9C93AA", "#CDB98A", "#7898A6", "#B6A6D2",
              "#93B49F", "#C9BE8C"]

FONTS = ("https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..700"
         "&family=Geist:wght@400;500;600&display=swap")


def dark() -> bool:
    return True


def colors() -> dict:
    return _NIGHT


def setup() -> None:
    """Call once per run, before the page."""
    c = _NIGHT
    st.markdown(f"""<style>
@import url('{FONTS}');
html, body, p, li, label, input, textarea, button, td, th, span, div,
[data-testid="stMarkdownContainer"], [data-testid="stSidebar"] {{
  font-family: 'Geist', system-ui, sans-serif; font-feature-settings: 'tnum' 1; }}
[data-testid="stIconMaterial"], .material-symbols-rounded {{
  font-family: 'Material Symbols Rounded' !important; }}
.stApp {{ background: {c['bg']}; }}
h1, h2, h3, h1 *, h2 *, h3 *, .sp-figure, .sp-value, [data-testid="stMetricValue"],
[data-testid="stHeading"] * {{
  font-family: 'Bricolage Grotesque', 'Geist', sans-serif !important; color: {c['ink']};
  font-variant-numeric: tabular-nums; }}
h1 {{ font-size: 2.6rem; font-weight: 650; letter-spacing: -0.02em; line-height: 1.05;
  padding-bottom: 0.2rem; }}
h2, h3 {{ font-weight: 600; letter-spacing: -0.01em; }}
h3 {{ font-size: 1.35rem; margin-top: 1.6rem; }}
.block-container {{ padding-top: 3.4rem; padding-bottom: 4rem; max-width: 1200px; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, .sp-note {{
  font-size: 0.92rem; line-height: 1.6; color: {c['muted']}; max-width: 75ch; }}
a {{ color: {c['signal']} !important; text-decoration-thickness: 1px; text-underline-offset: 3px; }}

/* sidebar: quieter than the page, marigold marks where you are */
[data-testid="stSidebar"] {{ background: #0D100F; border-right: 1px solid {c['rule']}; }}
[data-testid="stSidebarNav"] a {{ border-radius: 8px; }}
[data-testid="stSidebarNav"] a:hover {{ background: {c['surface']} !important; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: {c['raised']} !important;
  box-shadow: inset 3px 0 0 {c['signal']}; }}
[data-testid="stSidebarNav"] a[aria-current="page"] span {{ color: {c['ink']} !important; font-weight: 600; }}
[data-testid="stNavSectionHeader"] {{ color: {c['muted']} !important; font-size: 0.78rem;
  letter-spacing: 0.02em; }}

/* fixed top-right bar: alerts, portfolio, settings (stays while scrolling) */
.st-key-topbar {{ position: fixed; top: 0.6rem; right: 3.6rem; z-index: 999990; width: auto !important;
  background: rgba(23, 27, 26, 0.92); border: 1px solid {c['rule']}; border-radius: 999px;
  padding: 3px 6px; backdrop-filter: blur(6px); flex-wrap: nowrap; }}
.st-key-topbar [data-testid="stPageLink"] a, .st-key-topbar button {{ border-radius: 999px;
  padding: 2px 10px; min-height: 0; border: 0; background: transparent; }}
.st-key-topbar [data-testid="stPageLink"] a:hover, .st-key-topbar button:hover {{
  background: {c['raised']} !important; }}
.st-key-topbar p {{ font-size: 0.86rem; white-space: nowrap; }}
@media (max-width: 760px) {{ .st-key-topbar {{ right: 0.6rem; top: 3.2rem; }} }}

/* surfaces */
[data-testid="stMetric"], [data-testid="stExpander"] details {{ background: {c['surface']};
  border: 1px solid {c['rule']} !important; border-radius: 10px; }}
[data-testid="stMetric"] {{ padding: 14px 18px; }}
[data-testid="stExpander"] summary:hover {{ color: {c['signal']}; }}
[data-testid="stDataFrame"] {{ border: 1px solid {c['rule']}; border-radius: 10px; overflow: hidden; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 12px; }}
[data-baseweb="tab-list"] {{ gap: 1.2rem; border-bottom: 1px solid {c['rule']}; }}
[data-baseweb="tab"] p {{ font-size: 0.95rem; }}
[data-baseweb="tab-highlight"] {{ background-color: {c['signal']}; }}
[data-testid="stAlert"] {{ border-radius: 10px; }}
.stButton button {{ border-radius: 999px; border: 1px solid {c['rule']}; background: {c['surface']}; }}
.stButton button:hover {{ border-color: {c['signal']}; color: {c['signal']}; }}
:focus-visible {{ outline: 2px solid {c['signal']} !important; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}

/* tickets: one figure each, its note, and a thin meter */
.sp-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 12px; margin: 6px 0 22px; }}
.sp-ticket {{ background: {c['surface']}; border: 1px solid {c['rule']}; border-radius: 12px;
  padding: 16px 18px 14px; display: flex; flex-direction: column; min-height: 118px; }}
.sp-label {{ font-size: 0.86rem; font-weight: 500; color: {c['muted']}; }}
.sp-value {{ font-size: 1.85rem; font-weight: 600; line-height: 1.15; margin-top: 4px;
  letter-spacing: -0.01em; }}
.sp-value.up {{ color: {c['up']}; }} .sp-value.down {{ color: {c['down']}; }}
.sp-sub {{ font-size: 0.84rem; color: {c['muted']}; margin: 3px 0 12px; }}
.sp-meter {{ height: 3px; margin-top: auto; background: {c['rule']}; border-radius: 3px; overflow: hidden; }}
.sp-meter > span {{ display: block; height: 100%; border-radius: 3px; }}
.sp-meter.up > span {{ background: {c['up']}; }} .sp-meter.down > span {{ background: {c['down']}; }}
.sp-meter.neutral > span {{ background: {c['signal']}; }}

/* the overview's one loud element */
.sp-hero {{ display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(240px, 1fr); gap: 32px;
  padding: 4px 0 26px; border-bottom: 1px solid {c['rule']}; margin-bottom: 24px; }}
.sp-when {{ color: {c['signal']}; font-size: 0.95rem; font-weight: 500; }}
.sp-figure {{ font-size: clamp(3.6rem, 9vw, 6.6rem); font-weight: 650; line-height: 1;
  letter-spacing: -0.035em; margin: 10px 0 12px; }}
.sp-figure.up {{ color: {c['up']}; }} .sp-figure.down {{ color: {c['down']}; }}
.sp-next {{ background: {c['surface']}; border: 1px solid {c['rule']}; border-radius: 14px;
  padding: 16px 18px; }}
.sp-next h4 {{ margin: 0 0 6px; font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.05rem;
  font-weight: 600; color: {c['ink']}; }}
.sp-next ol {{ list-style: none; margin: 0; padding: 0; }}
.sp-next li {{ display: grid; grid-template-columns: 3.3rem 1fr; gap: 10px; padding: 7px 0;
  border-bottom: 1px solid {c['rule']}; font-size: 0.9rem; color: {c['ink']}; }}
.sp-next li:last-child {{ border-bottom: 0; }}
.sp-next li.done {{ opacity: 0.4; }}
.sp-next time {{ font-weight: 600; font-variant-numeric: tabular-nums; color: {c['signal']}; }}
@media (max-width: 760px) {{
  .sp-hero {{ grid-template-columns: 1fr; }}
  h1 {{ font-size: 2.1rem; }}
}}
</style>""", unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "", fill: float = 0.0, tone: str = "neutral",
         ring_text: str = "") -> str:
    """One ticket: label, the figure, a short note, and a meter for `fill` (0..1)."""
    tone = tone if tone in ("up", "down") else "neutral"
    f = max(0.0, min(1.0, fill))
    title = f' title="{html.escape(ring_text)}"' if ring_text else ""
    return (f'<div class="sp-ticket {tone}"{title}>'
            f'<div class="sp-label">{html.escape(label)}</div>'
            f'<div class="sp-value {tone}">{html.escape(value)}</div>'
            f'<div class="sp-sub">{html.escape(sub)}</div>'
            f'<div class="sp-meter {tone}"><span style="width:{f * 100:.1f}%"></span></div></div>')


def cards(items: list[str]) -> None:
    st.markdown(f'<div class="sp-grid">{"".join(items)}</div>', unsafe_allow_html=True)


def money(v: float, cur: str = "₹") -> str:
    sign = "+" if v > 0 else "−" if v < 0 else ""
    return f"{sign}{cur}{abs(v):,.2f}" if cur == "$" else f"{sign}{cur}{abs(v):,.0f}"


def gain_card(label: str, gain: float, base: float, scale: float = 0.03,
              label_base: str = "", cur: str = "₹") -> str:
    """Gain/loss with its meter showing it as % of `base` (full meter = ±scale)."""
    pct = gain / base if base else 0.0
    tone = "up" if gain > 0 else "down" if gain < 0 else "neutral"
    sub = (f"{pct:+.2%} on {cur}{base:,.0f} {label_base}".rstrip() if label_base
           else f"{pct:+.2%} of {cur}{base:,.0f}")
    return card(label, money(gain, cur), sub, abs(pct) / scale, tone, f"{pct:+.1%}")


def share_card(label: str, part: float, whole: float, value: str, sub: str = "") -> str:
    """A part of a whole (e.g. money invested of the capital)."""
    f = part / whole if whole else 0.0
    return card(label, value, sub, f, "neutral", f"{f:.0%}")


def hero(when: str, figure: float, caption: str, upcoming: list[tuple[str, str, bool]],
         cur: str = "₹") -> None:
    """Today's P&L as one big figure, and what the system does next.
    upcoming: [(HH:MM, what, already done)]."""
    tone = "up" if figure > 0 else "down" if figure < 0 else ""
    items = "".join(f'<li class="{"done" if done else ""}"><time>{html.escape(t)}</time>'
                    f'<span>{html.escape(w)}</span></li>' for t, w, done in upcoming)
    st.markdown(
        f'<section class="sp-hero"><div><div class="sp-when">{html.escape(when)}</div>'
        f'<div class="sp-figure {tone}">{html.escape(money(figure, cur))}</div>'
        f'<div class="sp-note">{html.escape(caption)}</div></div>'
        f'<div class="sp-next"><h4>Today on the desk</h4><ol>{items}</ol></div></section>',
        unsafe_allow_html=True)


def pnl_colors(df, cols):
    """Jade / coral text for gain-loss columns; a Styler for st.dataframe."""
    def tint(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return ""
        return (f"color: {UP}; font-weight: 600" if v > 0 else
                f"color: {DOWN}; font-weight: 600" if v < 0 else "")
    return df.style.map(tint, subset=[col for col in cols if col in df.columns])
