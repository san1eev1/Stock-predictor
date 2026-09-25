"""Chart helpers (Altair). Colours follow the entity, never its rank, and were
checked for colour-blind separation in light and dark themes."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

# Fixed entity -> categorical slot (light, dark).
SERIES = {
    "Model": ("#2a78d6", "#3987e5"),
    "Nifty 50": ("#eb6834", "#d95926"),
    "Equal-weight Nifty 250": ("#1baf7a", "#199e70"),
    "Momentum only": ("#eda100", "#c98500"),
    "Long-term": ("#2a78d6", "#3987e5"),
    "Intraday": ("#eb6834", "#d95926"),
    "Buy book": ("#2a78d6", "#3987e5"),
    "Sell book": ("#1baf7a", "#199e70"),
}
SINGLE = ("#2a78d6", "#3987e5")
GOOD, CRITICAL = "#0ca30c", "#d03b3b"


def _dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def _color(pair: tuple[str, str]) -> str:
    return pair[1] if _dark() else pair[0]


def _axis():
    return dict(gridOpacity=0.35, domainOpacity=0.4, tickOpacity=0.4, labelFontSize=11)


def _field(name: str) -> str:
    return str(name).replace(":", "\\:").replace(".", "\\.")


def lines(df: pd.DataFrame, x: str, y: str, series: str, y_format: str = ",.0f",
          y_title: str = "") -> alt.Chart:
    """Multi-series line chart with a hover crosshair and a legend."""
    names = [n for n in SERIES if n in set(df[series])] + \
        [n for n in df[series].unique() if n not in SERIES]
    colors = [_color(SERIES.get(n, SINGLE)) for n in names]
    color = alt.Color(f"{series}:N", scale=alt.Scale(domain=names, range=colors),
                      legend=alt.Legend(orient="top", title=None))
    base = alt.Chart(df).encode(x=alt.X(f"{x}:T", title=None, axis=alt.Axis(**_axis())))
    line = base.mark_line(strokeWidth=2).encode(
        y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=False),
                axis=alt.Axis(format=y_format, **_axis())), color=color)
    hover = alt.selection_point(fields=[x], nearest=True, on="pointerover", empty=False)
    rule = base.mark_rule(opacity=0.4).encode(
        opacity=alt.condition(hover, alt.value(0.4), alt.value(0)),
        tooltip=[alt.Tooltip(f"{x}:T", title="Date")]
        # series names may contain ':' ("Intraday 12:30"), which Altair reads as a type
        + [alt.Tooltip(f"{_field(n)}:Q", title=n, format=y_format) for n in names],
    ).transform_pivot(series, value=y, groupby=[x]).add_params(hover)
    points = line.mark_point(size=60, filled=True).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    return (line + points + rule).properties(height=320)


def bars(df: pd.DataFrame, x: str, y: str, y_format: str = ".0%", y_title: str = "",
         ref: float | None = None, sort=None) -> alt.Chart:
    """Single-series bar chart with tooltip and optional reference line."""
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4,
                                   color=_color(SINGLE)).encode(
        x=alt.X(f"{x}:N", title=None, sort=sort, axis=alt.Axis(labelAngle=0, **_axis())),
        y=alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(format=y_format, **_axis())),
        tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=y_format)])
    if ref is not None:
        rule = alt.Chart(pd.DataFrame({"v": [ref]})).mark_rule(strokeDash=[4, 4], opacity=0.6) \
            .encode(y="v:Q")
        chart = chart + rule
    return chart.properties(height=260)


def hbars(df: pd.DataFrame, cat: str, val: str, fmt: str = ".0%", title: str = "") -> alt.Chart:
    return alt.Chart(df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
                                  color=_color(SINGLE)).encode(
        y=alt.Y(f"{cat}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260, **_axis())),
        x=alt.X(f"{val}:Q", title=title, axis=alt.Axis(format=fmt, tickCount=5, **_axis())),
        tooltip=[alt.Tooltip(f"{cat}:N"), alt.Tooltip(f"{val}:Q", format=fmt)],
    ).properties(height=max(120, 26 * len(df)))


def money(v: float) -> str:
    """Signed rupee text: '+₹1,234' / '-₹1,234'. Streamlit metrics add the ▲/▼ arrow and
    colour from the sign, so meaning never rests on colour alone."""
    return f"{'-' if v < 0 else '+'}₹{abs(v):,.0f}"
