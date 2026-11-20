"""Role-based access to the three Simulation views and their accounts (11/17).

The signed-in user's Entra app roles arrive in the ID token's ``roles`` claim
(st.user["roles"]). Each role opens one Simulation view; Customer Deep Dive is
open to every role. Values shared with the Cloud team (11/17):

    FullSimulationView       -> Full Simulation        (display: "Full Simulation view")
    DirectionalView          -> Directional Prediction ("Directional Prediction view")
    CustomerInteligenceView  -> Customer Intelligence  ("Customer Intelligence view")

Matching is tolerant — case, spaces, dots/underscores/hyphens, a "CDE" prefix
and a "View" suffix are ignored, and Intelligence/Inteligence both match — so
the final values the Cloud team provisions work without a code change as long
as they name the same view.

TESTING PHASE (user, 11/17): a user with no recognised role sees ALL views, so
nobody is locked out while roles are being assigned (and sessions signed in
before the roles existed carry no roles claim until the user signs in again).
Once every user is assigned, flip NO_ROLE_SEES_ALL to False to deny instead.

Pure functions, no Streamlit — app.py reads st.user and calls these.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

# View ids as used by app.py's SIMULATION_VIEWS, in tab order.
VIEW_ORDER = ("full", "directional", "intel")

# Normalised role value -> view id (see normalise_role()).
ROLE_TO_VIEW = {
    "fullsimulation": "full",
    "directional": "directional",
    "directionalprediction": "directional",
    "customerinteligence": "intel",   # the value as shared, single "l"
    "customerintelligence": "intel",
}

NO_ROLE_SEES_ALL = True  # testing phase; False = a user with no role is denied

# Health systems each Simulation view shows, by IQVIA LOC ID (the app data's
# ACCOUNT_ID_COLUMN); a view not listed here shows every account. IDs, not
# names: names are not unique (two IDNs share "Example Health").
# Two effects (user, 11/17):
#   - the tab itself: a listed view's tab only ever shows its accounts, for
#     every user who can open it — a user with no role or several roles
#     included (view_accounts());
#   - the user: someone whose roles are all listed views is limited to those
#     accounts everywhere, Customer Deep Dive included (allowed_accounts()).
# Empty for now (user, 11/17): every view shows every account until the lists
# are confirmed with the client. A 7-account Full Simulation list was tested
# and parked; to restore it:
#     "full": ["IDN00000000", "IDN00000000", "IDN00000000", "IDN00000000",
#              "IDN00000000", "IDN00000000", "IDN00000000"],
# (Example Health, Example Health, Example Health,
#  Upmc, Example Health, Example Health, Example Health)
ROLE_ACCOUNTS: dict[str, list[str]] = {}


def view_accounts(view: str) -> list[str] | None:
    """IQVIA LOC IDs a Simulation tab shows, whoever opens it; None = every account."""
    return ROLE_ACCOUNTS.get(view)


def allowed_accounts(views: Iterable[str], fallback: bool) -> list[str] | None:
    """IQVIA LOC IDs the user may open, or None for every account.

    A user with no role (fallback) sees every account while testing. Several
    roles combine: any role whose view is unrestricted opens every account,
    otherwise the restricted lists are merged.
    """
    if fallback:
        return None
    ids: list[str] = []
    for view in views:
        if view not in ROLE_ACCOUNTS:
            return None
        ids += [i for i in ROLE_ACCOUNTS[view] if i not in ids]
    return ids


def normalise_role(value: Any) -> str:
    """'FullSimulationView', 'CDE.Full Simulation view', 'full_simulation' -> 'fullsimulation'."""
    text = re.sub(r"[^a-z]", "", str(value).lower())
    if text.startswith("cde"):
        text = text[3:]
    if text.endswith("view"):
        text = text[:-4]
    return text


def roles_from_claim(claim: Any) -> list[str]:
    """The ``roles`` claim as a list of strings (Entra sends a list; tolerate one string)."""
    if claim is None:
        return []
    if isinstance(claim, str):
        return [claim] if claim.strip() else []
    if isinstance(claim, Iterable):
        return [str(item) for item in claim if str(item).strip()]
    return []


def resolve_views(roles: Iterable[str]) -> dict[str, Any]:
    """Which Simulation views a user may open.

    Returns {"views": [...view ids in tab order], "matched": [...roles that map
    to a view], "unrecognised": [...roles that don't], "fallback": bool}.
    Several recognised roles open all of their views. No recognised role ->
    every view while NO_ROLE_SEES_ALL (fallback=True), else none.
    """
    matched, unrecognised, views = [], [], set()
    for role in roles:
        view = ROLE_TO_VIEW.get(normalise_role(role))
        if view:
            matched.append(role)
            views.add(view)
        else:
            unrecognised.append(role)
    fallback = not views
    if fallback:
        views = set(VIEW_ORDER) if NO_ROLE_SEES_ALL else set()
    ordered = [v for v in VIEW_ORDER if v in views]
    return {
        "views": ordered,
        "accounts": allowed_accounts(ordered, fallback),  # None = every account
        "matched": matched,
        "unrecognised": unrecognised,
        "fallback": fallback,
    }
