# Release, recovery, and identity setup

## Code release

1. Work from `azure/` in `AlienCommunication/databrics-mlops-e2e`.
2. Create a branch, make changes, and run `pytest tests -q`.
3. Review `git diff` for model/input changes and generated job configuration.
4. Deploy the same commit to staging, run training, daily pipeline, and real-time smoke query.
5. Inspect MLflow `validation.json`, endpoint version, and monitoring output.
6. Release that commit to prod. Save the commit SHA in bundle variable `release`.
7. Run production training and daily pipeline once before enabling schedules.

## Unattended CI/CD identity

The Azure workflow uses GitHub OIDC, not a copied personal token. It is intentionally separate
from the repository's existing AWS workflows. The YAML is not proof that authentication is configured.

Follow [Databricks' GitHub federation setup](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/auth/provider-github):

1. Create dedicated Databricks deployment service principals for staging and production and assign
   each only to this Azure workspace. Grant workspace access and permission to use serverless compute.
2. Create GitHub environments `azure-staging` and `azure-prod`. Restrict allowed release branches;
   configure required reviewers on production before enabling unattended production deployment.
3. Create a federation policy per principal with issuer `https://token.actions.githubusercontent.com`,
   audience equal to the Azure Databricks **account ID** (not workspace/subscription ID), and exact subjects:
   `repo:AlienCommunication/databrics-mlops-e2e:environment:azure-staging` and
   `repo:AlienCommunication/databrics-mlops-e2e:environment:azure-prod`.
4. Put the corresponding application's client ID in GitHub environment variable
   `DATABRICKS_CLIENT_ID`. No client secret is needed.
5. Grant `USE CATALOG` on `db_azure_workspace`, and the required schema/model/table privileges only
   on that principal's environment schema. A practical small-project configuration assigns schema
   ownership to its deployment identity; do not grant metastore admin or access to unrelated schemas.
6. Grant edit/manage access to that environment's experiment and bundle folder. If adopting jobs
   initially created by the interactive user, grant management and bind their IDs to the new bundle
   deployment before deploying. The current default root is per-user, so changing the identity alone
   creates a different deployment; do not silently leave duplicate active schedules.
7. Serving records its creator identity. An endpoint initially created by a person does not change
   creator when a service principal updates it. Plan a reviewed endpoint migration to the durable
   identity, test it, then switch clients; do not remove the original creator's access prematurely.
8. Run the manual Azure release workflow with production unchecked. Confirm the full staging run.
   Only then run it with production selected and approve the environment gate.

## Incident recovery

Training failure: current Champion and existing serving remain available. Inspect the failed task;
repair the failed task when its inputs are still valid, otherwise start a new full run.

Validation rejection: inspect metrics and dataset version; do not change Champion manually to bypass
the gate. If thresholds are wrong, update them in code with a reviewed rationale.

Deployment failure: compare endpoint's numeric version with Champion. Endpoint readiness and alias
updates are separate operations. The deployment task attempts online compensation if there is an
older approved version, but a compensation call itself can fail. Inspect both states after failure.

Rollback a passed version:

```bash
python scripts/rollback.py --environment prod --version 1
python scripts/predict.py --endpoint used-car-prod
databricks bundle run daily_pipeline -t prod -p used-car-azure
```

Pause recurring compute:

```bash
databricks bundle deploy -t prod -p used-car-azure --var pause_status=PAUSED
```

This pauses job schedules; it does not delete endpoint or stored data. Endpoints use scale to zero,
but requests can wake them. Inspect Serving to manage endpoint capacity.

Backfill a date (same car/date/version key remains unique):

```bash
databricks bundle run daily_pipeline -t prod -p used-car-azure --params as_of=2026-10-04
```

Monitoring failure: investigate missing predictions, invalid inputs, PSI, and available-label MAPE.
The job's failure signal appears in the Jobs UI. Alert delivery to email/Slack requires an explicitly
configured notification destination; none is implied by writing an alert row.

Do not run `bundle destroy` as a routine rollback. It removes bundle-managed resources, while this
project's runtime-created tables, model versions, and endpoints have a separate lifecycle.
