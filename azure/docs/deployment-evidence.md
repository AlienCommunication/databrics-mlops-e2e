# Deployment evidence — 4 October 2026

This records observed Azure results, rather than treating configuration files as proof of a run.
The workspace is `db-azure-workspace`, ID `7405610390336981`, East US, Premium Serverless.
The repository change is [PR #3](https://github.com/AlienCommunication/databrics-mlops-e2e/pull/3).

## Verified environment runs

| Environment | Training and serving | Daily pipeline |
|---|---|---|
| dev | [SUCCESS](https://adb-7405610390336981.1.azuredatabricks.net/jobs/1079735519494240/runs/955397897519745) after repairing the SDK endpoint configuration | [SUCCESS](https://adb-7405610390336981.1.azuredatabricks.net/jobs/879108158572237/runs/763696430038612); [same-date repeat SUCCESS](https://adb-7405610390336981.1.azuredatabricks.net/jobs/879108158572237/runs/70451135256019) |
| staging | [SUCCESS](https://adb-7405610390336981.1.azuredatabricks.net/jobs/334953625339881/runs/551185837209640) | [SUCCESS](https://adb-7405610390336981.1.azuredatabricks.net/jobs/133198000492598/runs/934654017703799) after repairing monitoring for newly initialized log-table schemas |
| prod | [SUCCESS](https://adb-7405610390336981.1.azuredatabricks.net/jobs/605367397119690/runs/955242223883722) | [SUCCESS](https://adb-7405610390336981.1.azuredatabricks.net/jobs/152437854449222/runs/1070333942155352) |

All three use registered model `db_azure_workspace.used_car_<env>.used_car_price`, version 1,
with Champion pointing to version 1. All three endpoints returned the same prediction for the
same sample, independently of the batch path.

## Real-time result

```json
{
  "age_years": 5.0,
  "mileage_km": 60000.0,
  "engine_cc": 1500.0,
  "owners": 1.0,
  "brand": "Honda",
  "fuel": "Petrol",
  "transmission": "Manual"
}
```

Production endpoint `used-car-prod` returned **641156.3946873571 INR**. Reproduce from `azure/`:

```bash
python scripts/predict.py --endpoint used-car-prod
```

The deployment task also compares online predictions with the exact registered model's local
pyfunc predictions before moving Champion. The client sends a UUID `client_request_id`.

## Batch and quality result

Production scored **700 rows**, model version **1**, date **2026-10-04**, with **0 duplicate keys**.
The seven-day replay yields **500 available labels** because outcomes arrive two days later.

| Metric | Observed value |
|---|---:|
| Available-label MAE | INR 41,238.92 |
| Available-label RMSE | INR 66,023.44 |
| Available-label MAPE | 6.2551% |
| Available-label R² | 0.98859 |
| Largest numeric-feature PSI | 0.02275 (age) |
| Batch monitoring alerts | none |
| Held-out training test MAE | INR 44,248.95 |
| Held-out training test MAPE | 6.4672% |

These measure synthetic generator behavior, not real used-car market accuracy.
The same-date development rerun preserved 700 rows and zero duplicate keys.

## Active schedules

Verified by reading the Jobs API after deployment:

- Production daily job: **UNPAUSED**, `0 0 8 * * ?`, **Asia/Kolkata** (08:00 daily).
- Production training job: **UNPAUSED**, `0 0 7 ? * SUN`, **Asia/Kolkata** (07:00 Sunday).
- Development and staging schedules stay paused.

The first scheduled future firing has not yet been observed. The same job graph was manually
executed successfully before activation. These schedules incur usage charges and use the
interactive owner's identity. Pause them with the command in the release runbook.

## Online logging

AI Gateway inference-table logging is enabled for all three endpoints. During verification,
the dev table expanded from its placeholder request-ID-only schema to the full request/response
schema. New staging/prod tables were still initializing. Monitoring explicitly records pending
telemetry through `log_status`; zero counts while pending are not an availability measurement.
Use the inspection notebook and latest `online_monitoring` row to see current delivery state.
Operational logging is not yet an end-to-end online sale-label accuracy join.

## Code and test evidence

Eight local tests passed, including model quality, serialization parity, validation rejection,
concurrent promotion checks, version pinning and deployment compensation. Ruff lint and formatting
passed. GitHub Azure CI passed on commit `72a58e1`:
[CI run](https://github.com/AlienCommunication/databrics-mlops-e2e/actions/runs/37218307617).

Production training initially received the earlier release label `bbb4632` while deployed source
had advanced to `fed23a7f0ce9e1ae4cf567f0ea357883229d1a6f`. This was not silently rewritten:
we compared deployed `workflow.py` and `modeling.py` byte-for-byte with that source, saved both
files and SHA256 hashes to the model's MLflow run under `verified_source`, and added a
`verified_source_commit` tag and a `release_label_note`. Monitoring then received the `72a58e1`
fix. Future releases must use the exact checked-out commit SHA throughout.

## Remaining work before unattended business operation

1. **CI/CD identity is not configured.** GitHub tests run, but the OIDC release workflow has not
   authenticated or deployed. Account-console sign-in hit a personal-account/tenant problem and
   then an unresolved Microsoft password prompt for the external directory identifier. Workspace
   login remains valid; no separate Databricks password was created. Use an authorized account
   administrator to complete federation and the migration steps in the release runbook.
2. **GitHub release approval:** configure the new Azure environments, reviewers, exact-subject
   federation and environment variables. The PR is unmerged; manual workflow dispatch normally
   becomes available after its workflow is in the default branch.
3. **Personal demo boundaries:** same workspace and owner across environments; synthetic data;
   no external email/Slack alert delivery; no real transaction ingestion; small pandas scoring.
   These are explicitly documented design limits, not completed enterprise controls.

For inspection in Databricks, open `/Shared/used-car-mlops/inspect-deployment`, select `prod`, and
run it on serverless compute. The notebook displays inputs, predictions, duplicate checks,
monitoring and request logs, then returns a compact JSON evidence record.
