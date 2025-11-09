"""This module contains the configuration entity for data ingestion,"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    transformed_data_file: Path
    source_data_file: Path
    selected_columns: list 

@dataclass(frozen=True)
class ModelTrainerConfig:
    best_model_path: Path
    data_file_path: Path
    target_column: str
    test_size: float
    random_state: int