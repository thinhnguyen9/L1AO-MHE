import os
import json
import inspect
import numpy as np
from simulate_quadrotor import main


def load_scenario_json(fname):
    with open(fname, "r") as f:
        cfg = json.load(f)
    array_vars = ["v_means", "v_stds", "w_means", "w_stds", "x0_stds"]
    diag_vars = ["Q", "R", "P0"]
    for k in array_vars:
        if k in cfg:
            cfg[k] = np.array(cfg[k])
    for k in diag_vars:
        if k in cfg:
            cfg[k] = np.diag(cfg[k])
    return cfg


if __name__ == "__main__":

    # ====================================================== #
    scenario_json_name = "scenarios/scenario2.json"
    # ====================================================== #
    
    scen_file = os.path.join(os.path.dirname(__file__), scenario_json_name)
    if not os.path.exists(scen_file):
        raise FileNotFoundError(f"Scenario file not found: {scen_file}")
    cfg = load_scenario_json(scen_file)

    # Filter to only parameters accepted by main() to avoid unexpected kwargs
    sig = inspect.signature(main)
    allowed = set(sig.parameters.keys())
    filtered_cfg = {k: v for k, v in cfg.items() if k in allowed}

    # Call main with only available/allowed variables from JSON
    main(**filtered_cfg)
