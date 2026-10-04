"""Explicit rollback to a passed model; update serving first, then batch alias."""
import argparse
from datetime import timedelta
import mlflow
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput

p=argparse.ArgumentParser()
p.add_argument('--profile',default='used-car-azure')
p.add_argument('--environment',choices=['dev','staging','prod'],required=True)
p.add_argument('--version',required=True)
args=p.parse_args()
mlflow.set_tracking_uri(f'databricks://{args.profile}')
mlflow.set_registry_uri(f'databricks-uc://{args.profile}')
c=MlflowClient(); name=f'db_azure_workspace.used_car_{args.environment}.used_car_price'
if c.get_model_version(name,args.version).tags.get('validation')!='passed':
    raise ValueError('Rollback target must have passed validation')
w=WorkspaceClient(profile=args.profile)
w.serving_endpoints.update_config_and_wait(f'used-car-{args.environment}',served_entities=[
    ServedEntityInput(entity_name=name,entity_version=args.version,workload_size='Small',scale_to_zero_enabled=True)],
    timeout=timedelta(minutes=40))
c.set_registered_model_alias(name,'Champion',args.version)
print(f'Online and batch now use {name} version {args.version}')
