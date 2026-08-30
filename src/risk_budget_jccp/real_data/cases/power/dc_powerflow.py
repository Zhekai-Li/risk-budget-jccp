from __future__ import annotations

import pandas as pd

from risk_budget_jccp.data.rts_gmlc import build_dc_ptdf


def compute_ptdf(bus: pd.DataFrame, branch: pd.DataFrame, slack_bus: str | int):
    return build_dc_ptdf(bus, branch, slack_bus=slack_bus)
