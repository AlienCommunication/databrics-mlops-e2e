# Reproduce the Azure workspace setup

Verified on 4 October 2026. This is the Azure deployment; the older `cloud.databricks.com`
workspace is on AWS. A numeric workspace ID alone does not identify its cloud or hostname.

## Portal path and the route that actually succeeded

1. Sign in to https://portal.azure.com with your existing Microsoft identity.
2. Search **Azure Databricks**, then **Create**. Choose the active subscription by ID.
3. Create/select resource group `db-azure`, choose East US, and name the workspace
   `db-azure-workspace` (use a different name for a second workspace).
4. Choose **Premium** and **Serverless**. These incur usage charges; this is not a free trial.
5. Review networking, encryption and security settings. This learning deployment uses public
   authenticated access, service-managed encryption, and no compliance add-ons.
6. Optionally add `project=used-car-mlops` and `environment=dev` tags.
7. Select **Review + create**. Only submit after validation succeeds and you accept the terms.
8. Open the resource and **Launch Workspace** after provisioning succeeds.

During our session both portal forms returned a generic validation error. We did not keep
resubmitting the form. The equivalent ARM template passed validation and deployed through
Azure Cloud Shell. That is the verified procedure below.

## Verified Cloud Shell procedure

Open Cloud Shell in Azure portal and choose Bash. The session may be ephemeral; retain the
[ARM template](workspace-serverless.json) in Git. These commands create paid resources.
For a second deployment, edit the template workspace name and resource-group name first.

```bash
az account list --all --query '[].{name:name,id:id,state:state}' -o table
az account set --subscription <your-active-subscription-id>
az group create --name db-azure --location eastus --tags project=used-car-mlops environment=dev
```

Upload `docs/workspace-serverless.json` to Cloud Shell, or paste its contents into a file named
`workspace-serverless.json`. The template uses API version `2026-01-01`, Premium SKU,
`computeMode: Serverless`, and `publicNetworkAccess: Enabled`.

```bash
az deployment group validate --resource-group db-azure \
  --template-file workspace-serverless.json --query properties.provisioningState -o tsv
az deployment group create --name used-car-workspace-setup --resource-group db-azure \
  --template-file workspace-serverless.json --query properties.provisioningState -o tsv
az databricks workspace show --name db-azure-workspace --resource-group db-azure \
  --query '{state:provisioningState,url:workspaceUrl,id:workspaceId,region:location,tier:sku.name,compute:computeMode}'
```

The official Azure CLI Databricks extension may be requested by the last command.
Our verified result was `Succeeded`, workspace ID `7405610390336981`, East US, Premium,
Serverless, and hostname `adb-7405610390336981.1.azuredatabricks.net`.
Open the returned HTTPS URL and select **Continue with Microsoft Entra ID**.
Confirm the workspace name, Catalog, Jobs & Pipelines, Experiments, Models, and Serving.
Then follow [the project walkthrough](../README.md) from the repository's `azure/` directory.

## Microsoft sign-in and account administration

Workspace sign-in succeeded using the existing Azure identity. No separate Databricks password
was created. The Azure directory's `#EXT#` username represents the external identity; it is not
an instruction to invent a new password. Account-console sign-in later routed the personal
Microsoft account to the wrong tenant; entering its directory UPN reached an unresolved password
prompt. Do not reset that identity solely to complete this demo.

Workspace admin and Databricks account admin are different roles. If the account already has
an admin, have that admin provision the dedicated CI/CD identity. If this is the first account
admin, follow the official bootstrap procedure with an authorized Entra Global Administrator.
Do not grant Global Administrator to the CI/CD service principal.

Sources: [serverless workspaces](https://learn.microsoft.com/en-us/azure/databricks/admin/workspace/serverless-workspaces),
[ARM deployment](https://learn.microsoft.com/en-us/azure/databricks/admin/workspace/arm-template),
[account admin and tenant access](https://learn.microsoft.com/en-us/azure/databricks/admin/admin-concepts).
