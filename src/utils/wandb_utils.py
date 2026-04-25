"""Weights & Biases utility wrappers."""

from __future__ import annotations

import os
from typing import Any

import wandb
from dotenv import load_dotenv


def init_wandb_run(
    *,
    config: dict[str, Any],
    enabled: bool,
    project: str,
    run_name: str | None = None,
    tags: list[str] | None = None,
) -> wandb.sdk.wandb_run.Run | None:
    """Initialize a W&B run safely using environment variables."""
    # Force loading project .env even if a stale WANDB_API_KEY
    # was already exported in the current shell/session.
    load_dotenv(override=True)

    if not enabled:
        return None

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        print("WANDB_API_KEY no encontrado. Se ejecutara sin tracking remoto.")
        return wandb.init(project=project, config=config, name=run_name, tags=tags, mode="disabled")
    api_key = api_key.strip().strip('"').strip("'")
    os.environ["WANDB_API_KEY"] = api_key
    # Improve reliability in Windows/Jupyter sessions where the W&B service
    # can take longer to boot and fail with "Failed to read port info".
    os.environ.setdefault("WANDB__SERVICE_WAIT", "60")

    return wandb.init(
        project=os.getenv("WANDB_PROJECT", project),
        entity=os.getenv("WANDB_ENTITY"),
        config=config,
        name=run_name,
        tags=tags,
    )


def finish_wandb_run(run: wandb.sdk.wandb_run.Run | None) -> None:
    """Close an existing W&B run if initialized."""
    if run is not None:
        run.finish()
