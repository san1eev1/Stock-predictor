"""Look and feel: theme-aware cards with ring gauges, global styling, dark mode by default
(still switchable in ⋮ → Settings)."""

from __future__ import annotations

import html
import math

import streamlit as st
import streamlit.components.v1 as components

UP, DOWN, NEUTRAL = "#22c55e", "#ef4444", "#60a5fa"

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


def dark() -> bool:
    try:
        return (st.context.theme.type or "dark") == "dark"
    except Exception:
        return True


def setup() -> None:
    """Call once per run, before the page."""
    components.html(_THEME_JS, height=0)
    d = dark()
    card = "rgba(30,41,59,.72)" if d else "rgba(255,255,255,.9)"
    edge = "rgba(148,163,184,.18)" if d else "rgba(15,23,42,.08)"
    shadow = "0 8px 24px rgba(0,0,0,.35)" if d else "0 6px 18px rgba(15,23,42,.08)"
    st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, p, li, label, h1, h2, h3, h4, input, textarea, button, [data-testid="stMetric"] *,
[data-testid="stMarkdownContainer"] {{ font-family: 'Inter', sans-serif; }}
[data-testid="stIconMaterial"], .material-symbols-rounded {{ font-family: 'Material Symbols Rounded' !important; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px; }}
h1 {{ font-weight: 700; letter-spacing: -.02em; }}
h2, h3 {{ font-weight: 600; letter-spacing: -.01em; }}
[data-testid="stMetric"] {{ background: {card}; border: 1px solid {edge}; border-radius: 16px;
  padding: 14px 18px; box-shadow: {shadow}; }}
[data-testid="stMetricValue"] {{ font-size: 1.55rem; font-weight: 650; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 18px; }}
[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}
.sp-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px;
  margin: 6px 0 18px; }}
.sp-card {{ background: {card}; border: 1px solid {edge}; border-radius: 18px; padding: 16px;
  box-shadow: {shadow}; display: flex; align-items: center; gap: 14px; backdrop-filter: blur(6px); }}
.sp-ring {{ flex: 0 0 auto; }}
.sp-label {{ font-size: .78rem; opacity: .7; text-transform: uppercase; letter-spacing: .06em; }}
.sp-value {{ font-size: 1.35rem; font-weight: 700; line-height: 1.25; }}
.sp-sub {{ font-size: .8rem; opacity: .75; }}
.sp-up {{ color: {UP}; }} .sp-down {{ color: {DOWN}; }}
.sp-hero {{ border-radius: 22px; padding: 22px 26px; margin-bottom: 14px; border: 1px solid {edge};
  background: linear-gradient(135deg, rgba(37,99,235,.22), rgba(34,197,94,.12)); }}
</style>""", unsafe_allow_html=True)


def ring(fill: float, color: str, size: int = 76, text: str = "") -> str:
    """Circular gauge: `fill` 0..1 of the circle in `color`, `text` in the middle."""
    r, sw = size / 2 - 6, 7
    c = 2 * math.pi * r
    f = max(0.0, min(1.0, fill))
    track = "rgba(148,163,184,.22)"
    return (f'<svg class="sp-ring" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{track}" stroke-width="{sw}"/>'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{color}" stroke-width="{sw}"'
            f' stroke-linecap="round" stroke-dasharray="{c * f:.1f} {c:.1f}"'
            f' transform="rotate(-90 {size/2} {size/2})"/>'
            f'<text x="50%" y="52%" text-anchor="middle" dominant-baseline="middle" '
            f'font-size="{size/6:.0f}" font-weight="700" fill="{color}">{html.escape(text)}</text></svg>')


def card(label: str, value: str, sub: str = "", fill: float = 0.0, tone: str = "neutral",
         ring_text: str = "") -> str:
    color = {"up": UP, "down": DOWN}.get(tone, NEUTRAL)
    cls = {"up": "sp-up", "down": "sp-down"}.get(tone, "")
    return (f'<div class="sp-card">{ring(fill, color, text=ring_text)}<div>'
            f'<div class="sp-label">{html.escape(label)}</div>'
            f'<div class="sp-value {cls}">{html.escape(value)}</div>'
            f'<div class="sp-sub">{html.escape(sub)}</div></div></div>')


def cards(items: list[str]) -> None:
    st.markdown(f'<div class="sp-grid">{"".join(items)}</div>', unsafe_allow_html=True)


def gain_card(label: str, gain: float, base: float, scale: float = 0.03,
              label_base: str = "") -> str:
    """Rupee gain/loss with a ring showing it as % of `base` (full ring = ±scale)."""
    pct = gain / base if base else 0.0
    tone = "up" if gain > 0 else "down" if gain < 0 else "neutral"
    sub = f"{pct:+.2%} on ₹{base:,.0f} {label_base}".rstrip() if label_base else f"{pct:+.2%} of ₹{base:,.0f}"
    return card(label, f"{'+' if gain > 0 else '−' if gain < 0 else ''}₹{abs(gain):,.0f}",
                sub, abs(pct) / scale, tone, f"{pct:+.1%}")


def share_card(label: str, part: float, whole: float, value: str, sub: str = "") -> str:
    """A part of a whole (e.g. money invested of the capital) as a ring."""
    f = part / whole if whole else 0.0
    return card(label, value, sub, f, "neutral", f"{f:.0%}")


def pnl_colors(df, cols):
    """Green / red text for gain-loss columns (like broker apps); a Styler for st.dataframe."""
    def tint(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return ""
        return f"color: {UP}; font-weight: 600" if v > 0 else \
            f"color: {DOWN}; font-weight: 600" if v < 0 else ""
    return df.style.map(tint, subset=[c for c in cols if c in df.columns])
