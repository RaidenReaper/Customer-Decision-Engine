"""Landing-page variants. LANDING_VARIANT=1..5 (or ?landing=N) picks one; unset shows none."""
from __future__ import annotations

import importlib
import os
from typing import Any

import streamlit as st

from landing.blocks import BLOCKS_CSS
from landing.common import html, inject_base_css

VARIANTS = ("1", "2", "3", "4", "5")


def active_variant() -> str | None:
    chosen = st.query_params.get("landing") or os.environ.get("LANDING_VARIANT", "")
    return chosen if chosen in VARIANTS else None


_SWITCH_KEY = "_page_switch"


def go_to(show_landing: bool) -> None:
    """Button callback for every landing <-> tool switch.

    A callback, not st.rerun(): state flips before the next run starts. It
    also flags that run as a page switch for page_switch_effects().
    """
    st.session_state["show_landing"] = show_landing
    st.session_state[_SWITCH_KEY] = True


# Runs once, on the first run after a switch. Streamlit keeps the previous
# page's elements on screen, faded, until each slot is redrawn, and the old
# page's CSS goes first, so it broke apart as the new page arrived. Hiding the
# stale elements makes the old page disappear at once; the new page starts at
# the top (the bottom buttons left you scrolled down) and fades in.
_SWITCH_EFFECTS = """
<style>
[data-stale="true"] {display: none !important;}
[data-testid="stMainBlockContainer"] {animation: lp-page-in .2s ease-out;}
@keyframes lp-page-in {from {opacity: 0;} to {opacity: 1;}}
[data-testid="stElementContainer"]:has(> .stHtml .lp-switch) {display: none;}
</style>
<span class="lp-switch"></span>
<script>
(() => {
  const doc = window.parent?.document || document;
  doc.querySelectorAll('[data-testid="stMain"], [data-testid="stAppViewContainer"], section.main')
     .forEach((el) => el.scrollTo({top: 0, behavior: "instant"}));
  window.scrollTo(0, 0);
})();
</script>
"""


def page_switch_effects() -> None:
    if not st.session_state.pop(_SWITCH_KEY, False):
        return
    st.html(_SWITCH_EFFECTS, unsafe_allow_javascript=True)
    if not st.session_state.get("show_landing", True):
        # Into the tool: take the landing container's slot with a hidden
        # element. Otherwise, after a tool -> landing -> tool round trip, the
        # landing container and the tool's first container share a slot,
        # Streamlit reuses the block, and the landing's quick guide lingers
        # inside the account card until the run ends.
        st.html("<span class='lp-switch'></span>")


def render_landing(variant: str, metadata: dict[str, Any] | None) -> None:
    module = importlib.import_module(f"landing.v{variant}")
    # One container for the whole landing, styles included: the tool's first
    # element then replaces it in a single step, instead of the landing being
    # overwritten piece by piece (expanders and columns are never marked stale,
    # so the CSS in _SWITCH_EFFECTS cannot hide them).
    with st.container(key="lp-page"):
        inject_base_css()
        html(BLOCKS_CSS)
        module.render(metadata)
