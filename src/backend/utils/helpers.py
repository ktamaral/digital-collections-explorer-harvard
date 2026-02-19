import json
import os
from pathlib import Path

import torch


def load_config(config_path=None):
    """Load configuration from JSON file"""
    if config_path is None:
        # Get the project root directory
        root_dir = Path(__file__).parent.parent.parent.parent
        config_path = root_dir / "config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    # Convert relative paths to absolute paths
    root_dir = Path(os.path.dirname(os.path.abspath(config_path)))
    for key in [
        "raw_data_dir",
        "processed_data_dir",
        "embeddings_dir",
        "thumbnails_dir",
    ]:
        if key in config:
            config[key] = str(root_dir / config[key])

    return config


def extract_embeddings(features) -> torch.Tensor:
    """Extract embeddings from CLIP model output, handling both v4.x and v5.x transformers

    Args:
        features: Output from CLIP model (can be Tensor or ModelOutput)

    Returns:
        torch.Tensor: Extracted embeddings tensor
    """
    # Handle both tensor and ModelOutput types
    if isinstance(features, torch.Tensor):
        # transformers v4.x returns tensor directly
        embeddings = features
    elif hasattr(features, 'pooler_output'):
        # transformers v5.x prefers pooler_output (2D: [batch, embedding_dim])
        embeddings = features.pooler_output
    elif hasattr(features, 'last_hidden_state'):
        # If we get last_hidden_state (3D), take the pooled token (first token)
        embeddings = features.last_hidden_state[:, 0, :]
    else:
        embeddings = torch.tensor(features)

    return embeddings
