"""Look and feel: the trading-floor ticket. Every book reads like an order slip - the rupee
figure first, then what changed - in condensed tabular figures, with short serif captions.
Dark by default (switchable in ⋮ → Settings); colours follow the active theme."""

from __future__ import annotations

import html

import streamlit as st
import streamlit.components.v1 as components

# Mid tones that read on both themes (charts, tables); CSS uses per-theme shades below.
UP, DOWN, NEUTRAL = "#3E9E6B", "#C25E45", "#4F7FB8"

PALETTE = {
    True: {"bg": "#1B2430", "surface": "#222D3B", "rule": "#33404F", "ink": "#E8ECE6",
           "muted": "#9AA6B2", "up": "#5BB884", "down": "#D9775C", "signal": "#7FA6D6"},
    False: {"bg": "#EEF1EC", "surface": "#FFFFFF", "rule": "#D3D9D0", "ink": "#1B2430",
            "muted": "#5D6874", "up": "#2E8B57", "down": "#B5523B", "signal": "#3A6EA5"},
}

# Streamlit keeps the light/dark choice per page in the browser and starts from the system
# setting. This makes Dark the default and carries the viewer's last choice to every page.
_THEME_JS = """
<script>
(function () {
  try {
    const w = window.parent, ls = w.localStorage;
    const key = "stActiveTheme-" + w.location.pathname + "-v2";
    const mine = ls.getItem(key);
    if (mine) { ls.setItem("sp-theme", mine); return; }
    if (w.sessionStorage.getItem("sp-theme-set-" + w.location.pathname)) return;
    w.sessionStorage.setItem("sp-theme-set-" + w.location.pathname, "1");
    ls.setItem(key, ls.getItem("sp-theme") || '"Dark"');
    w.location.reload();
  } catch (e) {}
})();
</script>
"""

FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600;700"
         "&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500"
         "&display=swap")


def dark() -> bool:
    try:
        return (st.context.theme.type or "dark") == "dark"
    except Exception:
        return True


def colors() -> dict:
    return PALETTE[dark()]


def setup() -> None:
    """Call once per run, before the page."""
    components.html(_THEME_JS, height=0)
    c = colors()
    st.markdown(f"""<style>
@import url('{FONTS}');
html, body, p, li, label, input, textarea, button, [data-testid="stMarkdownContainer"] {{
  font-family: 'IBM Plex Sans', system-ui, sans-serif; }}
[data-testid="stIconMaterial"], .material-symbols-rounded {{
  font-family: 'Material Symbols Rounded' !important; }}
h1, h2, h3, h4, [data-testid="stMetricValue"], .sp-figure, .sp-value {{
  font-family: 'IBM Plex Sans Condensed', 'IBM Plex Sans', sans-serif;
  font-variant-numeric: tabular-nums; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1240px; }}
h1 {{ font-weight: 700; font-size: 2.15rem; letter-spacing: -0.01em; line-height: 1.15; }}
h2, h3 {{ font-weight: 600; letter-spacing: 0; }}
h3 {{ font-size: 1.3rem; margin-top: 1.4rem; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, .sp-note {{
  font-family: 'Source Serif 4', Georgia, serif; font-size: 0.98rem; line-height: 1.6;
  color: {c['muted']}; max-width: 72ch; }}
[data-testid="stMetric"] {{ background: {c['surface']}; border: 1px solid {c['rule']};
  border-left: 3px solid {c['signal']}; border-radius: 6px; padding: 12px 16px; }}
[data-testid="stMetricValue"] {{ font-size: 1.6rem; font-weight: 600; }}
[data-testid="stDataFrame"] {{ border: 1px solid {c['rule']}; border-radius: 6px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 8px; }}
[data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {c['rule']}; }}
[data-baseweb="tab-highlight"] {{ background-color: {c['signal']}; }}
:focus-visible {{ outline: 2px solid {c['signal']} !important; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}

.sp-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px; margin: 8px 0 22px; }}
.sp-ticket {{ background: {c['surface']}; border: 1px solid {c['rule']}; border-left-width: 3px;
  border-radius: 6px; padding: 13px 16px 0; display: flex; flex-direction: column; }}
.sp-ticket.up {{ border-left-color: {c['up']}; }}
.sp-ticket.down {{ border-left-color: {c['down']}; }}
.sp-ticket.neutral {{ border-left-color: {c['signal']}; }}
.sp-label {{ font-size: 0.9rem; font-weight: 500; color: {c['muted']}; }}
.sp-value {{ font-size: 1.75rem; font-weight: 600; line-height: 1.2; margin-top: 2px; color: {c['ink']}; }}
.sp-value.up {{ color: {c['up']}; }} .sp-value.down {{ color: {c['down']}; }}
.sp-sub {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 0.88rem; color: {c['muted']};
  margin: 2px 0 11px; }}
.sp-meter {{ height: 4px; margin: auto -16px 0 -13px; background: {c['rule']}; border-radius: 0 0 5px 0; }}
.sp-meter > span {{ display: block; height: 100%; border-radius: 0 0 0 0; }}
.sp-meter.up > span {{ background: {c['up']}; }} .sp-meter.down > span {{ background: {c['down']}; }}
.sp-meter.neutral > span {{ background: {c['signal']}; }}

.sp-hero {{ display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(220px, 1fr); gap: 28px;
  padding: 6px 0 24px; border-bottom: 1px solid {c['rule']}; margin-bottom: 22px; }}
.sp-when {{ font-family: 'Source Serif 4', Georgia, serif; color: {c['muted']}; font-size: 1rem; }}
.sp-figure {{ font-size: clamp(3.2rem, 8vw, 5.6rem); font-weight: 700; line-height: 1;
  letter-spacing: -0.02em; margin: 6px 0 8px; color: {c['ink']}; }}
.sp-figure.up {{ color: {c['up']}; }} .sp-figure.down {{ color: {c['down']}; }}
.sp-next {{ border-left: 1px solid {c['rule']}; padding-left: 22px; }}
.sp-next h4 {{ margin: 0 0 8px; font-size: 1.05rem; font-weight: 600; }}
.sp-next ol {{ list-style: none; margin: 0; padding: 0; }}
.sp-next li {{ display: grid; grid-template-columns: 3.4rem 1fr; gap: 10px; padding: 6px 0;
  border-bottom: 1px solid {c['rule']}; font-size: 0.95rem; }}
.sp-next li.done {{ opacity: 0.45; }}
.sp-next time {{ font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 600;
  font-variant-numeric: tabular-nums; color: {c['signal']}; }}
@media (max-width: 760px) {{
  .sp-hero {{ grid-template-columns: 1fr; }}
  .sp-next {{ border-left: 0; padding-left: 0; }}
}}
</style>""", unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "", fill: float = 0.0, tone: str = "neutral",
         ring_text: str = "") -> str:
    """One order-slip ticket: label, the figure, a short note, and a meter for `fill` (0..1)."""
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
    """The overview's one loud element: today's P&L as a big tabular figure, and what the
    system does next. upcoming: [(HH:MM, what, already done)]."""
    tone = "up" if figure > 0 else "down" if figure < 0 else ""
    items = "".join(f'<li class="{"done" if done else ""}"><time>{html.escape(t)}</time>'
                    f'<span>{html.escape(w)}</span></li>' for t, w, done in upcoming)
    st.markdown(
        f'<section class="sp-hero"><div><div class="sp-when">{html.escape(when)}</div>'
        f'<div class="sp-figure {tone}">{html.escape(money(figure, cur))}</div>'
        f'<div class="sp-note">{html.escape(caption)}</div></div>'
        f'<div class="sp-next"><h4>Next</h4><ol>{items}</ol></div></section>',
        unsafe_allow_html=True)


def pnl_colors(df, cols):
    """Green / brick text for gain-loss columns; a Styler for st.dataframe."""
    c = colors()

    def tint(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return ""
        return (f"color: {c['up']}; font-weight: 600" if v > 0 else
                f"color: {c['down']}; font-weight: 600" if v < 0 else "")
    return df.style.map(tint, subset=[col for col in cols if col in df.columns])
