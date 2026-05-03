#!/usr/bin/env python3
"""
Two-cell conflict analysis schema for legacy TXP/CIO/HYS/TTT runs.

This keeps the main RET-based analyzer unchanged for current experiments and
injects the older parameter map before reusing the same analysis pipeline.
"""

import analyze_twocell_conflicts as base


base.PARAMS = [
    "c0_Param_TxPower", "c1_Param_TxPower",
    "c0_Param_CIO", "c1_Param_CIO",
    "c0_Param_Hys", "c1_Param_Hys",
    "c0_Param_TTT", "c1_Param_TTT",
]

base.KPIS = [
    "c0_SINR", "c1_SINR",
    "c0_dlThroughput", "c1_dlThroughput",
    "c0_trafficLoad", "c1_trafficLoad",
    "c0_rbUtil", "c1_rbUtil",
]

base.FEATURES = base.PARAMS + base.KPIS

base.SHORT_NAME.update({
    "c0_Param_Hys": "c0.Hys", "c1_Param_Hys": "c1.Hys",
    "c0_Param_TTT": "c0.TTT", "c1_Param_TTT": "c1.TTT",
})

base.PARAM_KPI_GT = {
    "c0_Param_TxPower": ["c0_SINR", "c0_dlThroughput", "c0_trafficLoad", "c0_rbUtil"],
    "c1_Param_TxPower": ["c1_SINR", "c1_dlThroughput", "c1_trafficLoad", "c1_rbUtil"],
    "c0_Param_CIO": ["c0_trafficLoad", "c1_trafficLoad", "c0_rbUtil", "c1_rbUtil"],
    "c1_Param_CIO": ["c0_trafficLoad", "c1_trafficLoad", "c0_rbUtil", "c1_rbUtil"],
    "c0_Param_Hys": ["c0_trafficLoad", "c1_trafficLoad", "c0_rbUtil", "c1_rbUtil"],
    "c1_Param_Hys": ["c0_trafficLoad", "c1_trafficLoad", "c0_rbUtil", "c1_rbUtil"],
    "c0_Param_TTT": ["c0_trafficLoad", "c1_trafficLoad", "c0_rbUtil", "c1_rbUtil"],
    "c1_Param_TTT": ["c0_trafficLoad", "c1_trafficLoad", "c0_rbUtil", "c1_rbUtil"],
}

base.XAPP_PARAMS_GT = {
    "CCO": ["c0_Param_TxPower", "c1_Param_TxPower"],
    "ES": ["c0_Param_TxPower", "c1_Param_TxPower"],
    "MLB": [
        "c0_Param_CIO", "c1_Param_CIO",
        "c0_Param_Hys", "c1_Param_Hys",
        "c0_Param_TTT", "c1_Param_TTT",
    ],
}


def map_action_param_to_icp(adjusted_param: str) -> str | None:
    if not isinstance(adjusted_param, str):
        return None
    cell = adjusted_param.split("_")[0]
    if "_TxPower_delta" in adjusted_param:
        return f"{cell}_Param_TxPower"
    if "_CIO_delta" in adjusted_param:
        return f"{cell}_Param_CIO"
    if "_Hys_delta" in adjusted_param:
        return f"{cell}_Param_Hys"
    if "_TTT_delta" in adjusted_param:
        return f"{cell}_Param_TTT"
    return None


def derive_xapp_param(df):
    xapp_param = {}
    for xapp, group in df.groupby("xapp"):
        icps = []
        for raw_param in group["adjusted_param"].dropna().unique().tolist():
            icp = map_action_param_to_icp(raw_param)
            if icp is not None and icp not in icps:
                icps.append(icp)
        xapp_param[xapp] = sorted(icps)
    return xapp_param


base.map_action_param_to_icp = map_action_param_to_icp
base.derive_xapp_param = derive_xapp_param


if __name__ == "__main__":
    base.main()
