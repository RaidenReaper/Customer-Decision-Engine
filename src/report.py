"""Scenario report as a PDF, for download from the Pricing Simulator.

Deliberately builds a structured report rather than a screenshot of the page.
Capturing the rendered page would need a headless browser in the image; the
report carries the same content — context, levers, per-product outcome and the
drivers behind it — in a form that prints and reads better than a screenshot.

No Streamlit import here, so the whole thing is testable without a running app.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any

from fpdf import FPDF

try:
    from src.utils import (
        APP_VERSION,
        LEVER_FEATURES,
        PRICE_BASE_COLUMN,
        lever_label_for_value,
        shap_chart_contributions,
    )
except ModuleNotFoundError:  # running as a script from inside src/
    from utils import (
        APP_VERSION,
        LEVER_FEATURES,
        PRICE_BASE_COLUMN,
        lever_label_for_value,
        shap_chart_contributions,
    )

# Same purple as the app header, converted to RGB for fpdf.
PURPLE = (82, 36, 1)
PURPLE_LIGHT = (251, 232, 221)
INK = (31, 31, 41)
MUTED = (110, 102, 98)
RULE = (225, 216, 211)
# The app's SHAP pair, so the printed chart reads like the on-screen waterfall.
SHAP_POSITIVE = (30, 136, 229)
SHAP_NEGATIVE = (255, 13, 87)

PAGE_WIDTH = 210.0
MARGIN = 14.0
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN

# Same set, labels and order as app.py's LEVER_CONFIGS: Dead Net Price plus
# the five v8 model inputs exposed as levers.
LEVER_LABELS = {
    "dead_net_price": "Dead Net Price",
    "bob_Prior Season Shr of Wallet_enc": "Share of Wallet",
    "bob_Prior Season HS Partnership Advocacy_enc": "Partnership Advocacy",
    "cde_engagement_length_log": "Engagement Score",
    "negotiation_receptivity_score": "Negotiation Stance",
    "apt_clinical_bucket_enc": "Clinical Focus",
}


def _latin1(text: Any) -> str:
    """fpdf's core fonts are latin-1 only.

    Account names and driver labels come from the source data and do contain
    characters outside it, which would raise mid-render. Substituting the few
    typographic ones we emit and dropping the rest is better than shipping a
    Unicode font file for the sake of an em dash.
    """
    swaps = {"—": "-", "–": "-", "·": "-", "’": "'", "‘": "'"}
    out = "".join(swaps.get(character, character) for character in str(text))
    return out.encode("latin-1", "replace").decode("latin-1")


def _lever_display(key: str, value: Any) -> str:
    """Render a lever the way the screen does: options as their dropdown
    label (a missing input as its No Data / Unknown option), price as money."""
    if key == "dead_net_price":
        return "n/a" if value is None or value != value else f"${float(value):,.2f}"
    if key in LEVER_FEATURES:
        return lever_label_for_value(key, value)
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _lever_value(item: dict[str, Any], key: str, product_rows: dict[str, Any]) -> Any:
    """The scenario's lever value, falling back to the account's own row.

    A baseline/unmodified scenario scores with an empty ``overrides`` dict, so
    every lever would otherwise print "n/a" instead of the default the row
    actually carries.
    """
    value = item["overrides"].get(key)
    if value is not None:
        return value
    row = product_rows.get(item["product"])
    if row is None:
        return None
    # dead_net_price is scored from PRICE_BASE_COLUMN ("dead_net_price_per_dose"),
    # not from a same-named row column — see app.py's collect_overrides().
    column = PRICE_BASE_COLUMN if key == "dead_net_price" else key
    return row.get(column)


def report_filename(customer: str, on: date | None = None, suffix: str | None = None) -> str:
    """<account name>_<YYYY-MM-DD>.pdf, safe for Windows and POSIX. A
    ``suffix`` goes before the date (<account name>_<suffix>_<date>.pdf), so
    the drivers export doesn't share the scenario report's name."""
    stamp = (on or date.today()).isoformat()
    safe = "".join(
        character if (character.isalnum() or character in " -_") else "_"
        for character in str(customer)
    ).strip()
    safe = "_".join(safe.split()) or "scenario"
    if suffix:
        safe = f"{safe}_{suffix}"
    return f"{safe}_{stamp}.pdf"


class _Report(FPDF):
    # The line under the title in the purple header band.
    subtitle = "Scenario report - internal use only"

    def header(self) -> None:  # noqa: D102 - fpdf hook
        self.set_fill_color(*PURPLE)
        self.rect(0, 0, PAGE_WIDTH, 22, style="F")
        self.set_xy(MARGIN, 6)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.cell(0, 6, "Customer Response Prediction Engine", new_x="LMARGIN", new_y="NEXT")
        self.set_x(MARGIN)
        self.set_font("Helvetica", "", 8.5)
        self.cell(0, 4, _latin1(self.subtitle), new_x="LMARGIN", new_y="NEXT")
        self.set_y(30)
        self.set_text_color(*INK)

    def footer(self) -> None:  # noqa: D102 - fpdf hook
        self.set_y(-13)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*MUTED)
        self.cell(
            0,
            4,
            _latin1(
                f"Version {APP_VERSION}  |  Generated {date.today().isoformat()}  |  "
                f"Page {self.page_no()}"
            ),
            align="C",
        )


def _section(pdf: _Report, title: str) -> None:
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(*PURPLE)
    pdf.cell(0, 6, _latin1(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*RULE)
    pdf.line(MARGIN, pdf.get_y(), PAGE_WIDTH - MARGIN, pdf.get_y())
    pdf.ln(2)
    pdf.set_text_color(*INK)


def _table(pdf: _Report, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_fill_color(*PURPLE_LIGHT)
    pdf.set_text_color(*PURPLE)
    for header, width in zip(headers, widths):
        pdf.cell(width, 7, _latin1(header), border=0, fill=True, align="L")
    pdf.ln(7)

    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*INK)
    for index, row in enumerate(rows):
        if index % 2:
            pdf.set_fill_color(250, 249, 252)
            fill = True
        else:
            fill = False
        for cell, width in zip(row, widths):
            pdf.cell(width, 6.5, _latin1(cell), border=0, fill=fill, align="L")
        pdf.ln(6.5)


def _grouped_table(
    pdf: _Report, headers: list[str], groups: list[tuple[str, list[list[str]]]], widths: list[float]
) -> None:
    """_table with a full-width category line above each group of rows."""
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_fill_color(*PURPLE_LIGHT)
    pdf.set_text_color(*PURPLE)
    for header, width in zip(headers, widths):
        pdf.cell(width, 7, _latin1(header), border=0, fill=True, align="L")
    pdf.ln(7)

    for title, rows in groups:
        pdf.ln(1.5)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(*PURPLE)
        pdf.cell(sum(widths), 6, _latin1(title), border="B", align="L")
        pdf.ln(6.5)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(*INK)
        for index, row in enumerate(rows):
            fill = bool(index % 2)
            if fill:
                pdf.set_fill_color(250, 249, 252)
            for cell, width in zip(row, widths):
                pdf.cell(width, 6.5, _latin1(cell), border=0, fill=fill, align="L")
            pdf.ln(6.5)


def build_drivers_pdf(
    customer: str,
    season: str,
    products: list[str],
    groups: list[tuple[str, list[list[str]]]],
    generated_on: date | None = None,
) -> bytes:
    """The Customer Deep Dive's drivers export: every model driver's value for
    this account, one column per product, grouped by driver category.

    ``groups`` is [(category, [[driver, value per product, ...], ...]), ...],
    already formatted for display by the app (as Account signals shows them),
    so this module stays free of the app's formatting rules.
    """
    stamp = (generated_on or date.today()).isoformat()
    pdf = _Report(orientation="P", unit="mm", format="A4")
    pdf.subtitle = "Account drivers - internal use only"
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 8, _latin1(customer), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*MUTED)
    pdf.cell(
        0,
        5,
        _latin1(f"Season {season}   |   {', '.join(products)}   |   Generated {stamp}"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_text_color(*INK)

    driver_count = sum(len(rows) for _, rows in groups)
    _section(pdf, f"Model drivers ({driver_count})")
    width_first = 60.0
    width_rest = (CONTENT_WIDTH - width_first) / max(len(products), 1)
    _grouped_table(pdf, ["Driver"] + list(products), groups, [width_first] + [width_rest] * len(products))

    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(
        CONTENT_WIDTH,
        4.4,
        _latin1(
            f"The account's own values for the {season} season, as the Customer Deep Dive shows them. "
            "Season is also a model input but not a business driver, so it is not listed. "
            "No data = not recorded for this account."
        ),
    )
    return bytes(pdf.output())


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _probability_deltas(shap: Any, base_value: float) -> list[float]:
    """Each driver's contribution re-expressed in probability space.

    SHAP additivity holds in log-odds, not probability — sigmoid is
    non-linear, so this is NOT sigmoid(each driver's shap value). It is the
    same link="logit" technique the app's waterfall and SHAP's own waterfall
    plot use: chain the drivers into a running log-odds total (order matters,
    largest |shap| first, same as the on-screen chart), then take the change
    in sigmoid(running total) that each step causes.
    """
    running = base_value
    deltas = []
    for _, driver in shap.iloc[::-1].iterrows():
        start = running
        running += float(driver["shap"])
        deltas.append(_sigmoid(running) - _sigmoid(start))
    return list(reversed(deltas))


def _shap_block(pdf: _Report, item: dict[str, Any], mode: str = "Log-odds") -> None:
    """A diverging bar chart of every feature's SHAP contribution, drawn with
    fpdf primitives.

    Drawn rather than rendered to an image on purpose. Turning the app's Altair
    waterfall into a PNG would need vl-convert or matplotlib, and
    requirements.txt records that matplotlib and shap were deliberately dropped
    to keep roughly 200 MB out of the image. Rectangles cost nothing.
    """
    shap = shap_chart_contributions(item.get("shap"))
    product = str(item["product"])
    is_probability = mode == "Probability"
    heading = f"Model drivers - {product} ({'probability' if is_probability else 'log-odds'})"

    if shap is None or getattr(shap, "empty", True):
        _section(pdf, heading)
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(*MUTED)
        reason = item.get("shap_error") or "not available for this scenario"
        pdf.cell(0, 5, _latin1(f"Contributions {reason}."), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*INK)
        return

    # base_value and final needed in both modes now, for the Account
    # Prediction / Average Win Probability reference line below — previously only computed
    # inside the probability branch, since only _probability_deltas() used it.
    base_value = float(getattr(shap, "attrs", {}).get("base_value", 0.0))
    final = base_value + float(shap["shap"].sum())

    if is_probability:
        values = _probability_deltas(shap, base_value)
        value_format = "{:+.1%}"
    # Log-odds branch disabled per request — _shap_block() is only ever
    # called with mode="Probability" (see the caller below), so this is
    # unreachable. Left commented (not deleted) so it can be restored.
    # else:
    #     values = [float(v) for v in shap["shap"]]
    #     value_format = "{:+.3f}"

    label_width, value_width = 52.0, 24.0
    plot_width = CONTENT_WIDTH - label_width - value_width
    centre = MARGIN + label_width + plot_width / 2
    largest = max((abs(v) for v in values), default=0.0) or 1.0
    row_height = 4.8

    # Break before the heading, not after it. Measuring once the title is on the
    # page leaves it stranded at the foot of the previous one, with the bars
    # starting overleaf under a second, duplicate title. +6 for the Account
    # Prediction / Average Win Probability line added between the heading and the bars.
    needed = row_height * len(shap) + 18 + 6
    if pdf.get_y() + needed > 297 - 18:
        pdf.add_page()

    _section(pdf, heading)

    # Account Prediction / Average Win Probability reference line — same two figures as the
    # on-screen waterfall's caption (app.py's fx_text) and the same two
    # values the vertical zero-line/reference marks below sit at, so the PDF
    # states the same reference point the chart is read against.
    if is_probability:
        prediction_text = f"{_sigmoid(final):.1%}"
        baseline_text = f"{_sigmoid(base_value):.1%}"
    # else:
    #     prediction_text = f"{final:+.3f}"
    #     baseline_text = f"{base_value:+.3f}"
    y = pdf.get_y()
    pdf.set_xy(MARGIN, y)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*MUTED)
    pdf.cell(pdf.get_string_width("Account Prediction = "), 5.5, _latin1("Account Prediction = "))
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(*INK)
    pdf.cell(pdf.get_string_width(prediction_text) + 4, 5.5, _latin1(prediction_text))
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*MUTED)
    pdf.cell(
        pdf.get_string_width("  ·  Average Win Probability = "),
        5.5,
        _latin1("  ·  Average Win Probability = "),
    )
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(*INK)
    pdf.cell(0, 5.5, _latin1(baseline_text), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*INK)

    top = pdf.get_y()
    for (_, driver), value in zip(shap.iterrows(), values):
        width = abs(value) / largest * (plot_width / 2 - 2)
        y = pdf.get_y()

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*INK)
        pdf.set_xy(MARGIN, y)
        pdf.cell(label_width, row_height, _latin1(str(driver["label"])), align="L")

        if value >= 0:
            pdf.set_fill_color(*SHAP_POSITIVE)
            pdf.rect(centre, y + 1.0, width, row_height - 2.0, style="F")
        else:
            pdf.set_fill_color(*SHAP_NEGATIVE)
            pdf.rect(centre - width, y + 1.0, width, row_height - 2.0, style="F")

        pdf.set_xy(MARGIN + label_width + plot_width, y)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*(SHAP_POSITIVE if value >= 0 else SHAP_NEGATIVE))
        pdf.cell(value_width, row_height, value_format.format(value), align="R")
        pdf.ln(row_height)

    # Zero line last, so it sits over the bars.
    pdf.set_draw_color(*MUTED)
    pdf.line(centre, top, centre, pdf.get_y())

    pdf.set_text_color(*MUTED)
    pdf.set_font("Helvetica", "", 7.5)
    if is_probability:
        footnote = (
            "Blue raises the win probability, red lowers it. Contributions are the "
            "change in predicted probability each driver causes, at its position in "
            "the ranked order — not independent of the others."
        )
    # else:
    #     footnote = (
    #         "Blue raises the win probability, red lowers it. Contributions are in "
    #         "log-odds and sum to the model's margin."
    #     )
    pdf.multi_cell(CONTENT_WIDTH, 4.5, _latin1(footnote))
    pdf.set_text_color(*INK)


def build_scenario_pdf(
    customer: str,
    season: str,
    product_rows: dict[str, Any],
    result: dict[str, Any],
    generated_on: date | None = None,
) -> bytes:
    """The scenario the user has just run, as a one-page PDF.

    ``result`` is the dict returned by score_selected_products: it carries the
    portfolio-level call plus a product_results entry per brand, each with its
    own overrides, threshold and SHAP contributions.
    """
    stamp = (generated_on or date.today()).isoformat()
    items = result["product_results"]

    pdf = _Report(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.add_page()

    # --- context -------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 8, _latin1(customer), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*MUTED)
    products = ", ".join(str(product) for product in product_rows)
    pdf.cell(
        0,
        5,
        _latin1(f"Season {season}   |   {products}   |   Generated {stamp}"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_text_color(*INK)

    # --- per product ---------------------------------------------------------
    #
    # No portfolio headline. Each product carries its own decision threshold
    # (HD 0.65 / STD 0.60 / FLK 0.65), so averaging the probabilities and
    # comparing them to an averaged threshold produces a number that belongs to
    # no product and can call the portfolio a win while a brand inside it loses.
    _section(pdf, "Predicted outcome by product")
    _table(
        pdf,
        ["Product", "Win probability", "Threshold", "Predicted", "Confidence", "Imputed"],
        [
            [
                str(item["product"]),
                f"{item['win_probability']:.1%}",
                f"{item['threshold']:.2f}",
                str(item["predicted_outcome"]),
                # The same win/loss model confidence the KPI card shows, not
                # the margin-based confidence_band() this used to print — two
                # different measures under one "Confidence" heading is exactly
                # how print and screen drift apart.
                str(item.get("winloss_confidence_label") or "-"),
                str(len(item["imputed_features"])),
            ]
            for item in items
        ],
        [44, 30, 22, 24, 26, 20],
    )

    # --- levers --------------------------------------------------------------
    _section(pdf, "Contracting levers")
    lever_keys = list(LEVER_LABELS)
    width_first = 52.0
    width_rest = (CONTENT_WIDTH - width_first) / max(len(items), 1)
    _table(
        pdf,
        ["Lever"] + [str(item["product"]) for item in items],
        [
            [LEVER_LABELS[key]]
            + [_lever_display(key, _lever_value(item, key, product_rows)) for item in items]
            for key in lever_keys
        ],
        [width_first] + [width_rest] * len(items),
    )

    # --- SHAP probability chart per selected product -------------------------
    # Log-odds dropped from the PDF per request — Probability only, matching
    # the app's own default view.
    for item in items:
        _shap_block(pdf, item, "Probability")

    if result.get("imputed_features"):
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(
            CONTENT_WIDTH,
            4.4,
            _latin1(
                "Imputed to the training-product median (not recorded for this account): "
                + ", ".join(result["imputed_features"])
            ),
        )

    return bytes(pdf.output())
