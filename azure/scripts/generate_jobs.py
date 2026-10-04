"""Generate the reviewable bundle jobs from one shared configuration."""

from pathlib import Path
import yaml

base = {
    "catalog": "${var.catalog}",
    "schema": "${var.schema}",
    "endpoint": "used-car-${bundle.target}",
    "experiment": "/Shared/used-car-mlops/${bundle.target}",
    "release": "${var.release}",
    "as_of": "{{job.parameters.as_of}}",
}


def task(name, deps=()):
    return dict(
        task_key=name,
        depends_on=[{"task_key": d} for d in deps],
        timeout_seconds=3600 if name == "deploy" else 1800,
        max_retries=0,
        notebook_task={"notebook_path": "../src/task.py", "base_parameters": dict(base, task=name)},
    )


def job(name, tasks, cron=None):
    return dict(
        name="used-car-${bundle.target}-" + name,
        max_concurrent_runs=1,
        tags={"project": "used-car-mlops", "environment": "${bundle.target}"},
        parameters=[{"name": "as_of", "default": ""}],
        tasks=tasks,
        **(
            {
                "schedule": {
                    "quartz_cron_expression": cron,
                    "timezone_id": "Asia/Kolkata",
                    "pause_status": "${var.pause_status}",
                }
            }
            if cron
            else {}
        ),
    )


jobs = {
    "training": job(
        "training",
        [
            task("prepare"),
            task("train", ["prepare"]),
            task("validate", ["train"]),
            task("deploy", ["validate"]),
        ],
        "0 0 7 ? * SUN",
    ),
    "batch_inference": job("batch", [task("inputs"), task("batch", ["inputs"])]),
    "monitoring": job("monitoring", [task("monitor")]),
}
jobs["daily_pipeline"] = job(
    "daily",
    [
        {
            "task_key": "score",
            "run_job_task": {
                "job_id": "${resources.jobs.batch_inference.id}",
                "job_parameters": {"as_of": "{{job.parameters.as_of}}"},
            },
        },
        {
            "task_key": "monitor",
            "depends_on": [{"task_key": "score"}],
            "run_job_task": {
                "job_id": "${resources.jobs.monitoring.id}",
                "job_parameters": {"as_of": "{{job.parameters.as_of}}"},
            },
        },
    ],
    "0 0 8 * * ?",
)
Path("resources/jobs.yml").write_text(
    yaml.safe_dump({"resources": {"jobs": jobs}}, sort_keys=False)
)
