"""Example real-time client using OAuth; no embedded tokens."""

import argparse
import json
import sys
import uuid
from pathlib import Path
from databricks.sdk import WorkspaceClient

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from modeling import validate_features
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--profile", default="used-car-azure")
p.add_argument("--endpoint", default="used-car-prod")
p.add_argument("--input", type=Path)
args = p.parse_args()
records = (
    json.loads(args.input.read_text())
    if args.input
    else [
        {
            "age_years": 5.0,
            "mileage_km": 60000.0,
            "engine_cc": 1500.0,
            "owners": 1.0,
            "brand": "Honda",
            "fuel": "Petrol",
            "transmission": "Manual",
        }
    ]
)
frame = validate_features(pd.DataFrame(records))
w = WorkspaceClient(profile=args.profile)
result = w.serving_endpoints.query(
    args.endpoint,
    dataframe_records=frame.to_dict(orient="records"),
    client_request_id=str(uuid.uuid4()),
)
print(json.dumps({"currency": "INR", "predictions": result.predictions}, indent=2))
