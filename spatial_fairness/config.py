"""
Configuration for the Spatial Fairness term.

This module defines all configurable parameters for the spatial fairness
computation, including grid dimensions, temporal aggregation settings,
and computation options.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import TermConfig


@dataclass
class SpatialFairnessConfig(TermConfig):
    """
    Configuration for the Spatial Fairness term.
    
    Attributes:
        period_type: Temporal aggregation level
            - "time_bucket": Each 5-min bucket (finest, 288 periods/day)
            - "hourly": Each hour (24 periods/day)  
            - "daily": Each day (6 periods total)
            - "all": Aggregate all data into single period
        grid_dims: Spatial grid dimensions (x, y)
        num_taxis: Number of active taxis in the dataset
        num_days: Number of days in the dataset
        include_zero_cells: Whether to include cells with zero activity
        data_is_one_indexed: Whether the data uses 1-based indexing
        min_activity_threshold: Minimum total activity to include a cell
    """
    
    # Temporal aggregation
    period_type: str = "hourly"  # "time_bucket", "hourly", "daily", "all"
    
    # Spatial configuration
    grid_dims: Tuple[int, int] = (48, 90)  # (x_cells, y_cells)
    
    # Fleet parameters
    num_taxis: int = 50                # Number of active taxis
    num_days: float = 21.0             # Days in dataset (July 2016 weekdays)
    
    # Computation options
    include_zero_cells: bool = True    # Include cells with zero activity
    data_is_one_indexed: bool = True   # Whether source data uses 1-based indexing
    min_activity_threshold: int = 0    # Minimum events to include cell
    
    # Days to include (None = all)
    days_filter: Optional[List[int]] = None  # e.g., [1, 2, 3] for Mon-Wed
    
    # Time buckets to include (None = all)
    time_filter: Optional[Tuple[int, int]] = None  # e.g., (1, 144) for first half of day
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        super().validate()
        
        valid_periods = ["time_bucket", "hourly", "daily", "all"]
        if self.period_type not in valid_periods:
            raise ValueError(
                f"Invalid period_type '{self.period_type}'. "
                f"Must be one of: {valid_periods}"
            )
        
        if self.grid_dims[0] <= 0 or self.grid_dims[1] <= 0:
            raise ValueError("Grid dimensions must be positive")
        
        if self.num_taxis <= 0:
            raise ValueError("num_taxis must be positive")
        
        if self.num_days <= 0:
            raise ValueError("num_days must be positive")
        
        if self.min_activity_threshold < 0:
            raise ValueError("min_activity_threshold must be non-negative")


# Predefined configurations for common use cases
FINE_GRAINED_CONFIG = SpatialFairnessConfig(
    period_type="time_bucket",
    include_zero_cells=True,
    verbose=False,
)

HOURLY_CONFIG = SpatialFairnessConfig(
    period_type="hourly",
    include_zero_cells=True,
    verbose=False,
)

DAILY_CONFIG = SpatialFairnessConfig(
    period_type="daily",
    include_zero_cells=True,
    verbose=False,
)

AGGREGATE_CONFIG = SpatialFairnessConfig(
    period_type="all",
    include_zero_cells=True,
    verbose=False,
)

# Peak hours analysis (morning and evening rush)
PEAK_HOURS_CONFIG = SpatialFairnessConfig(
    period_type="hourly",
    time_filter=(7*12+1, 10*12),  # 7am-10am bucket range
    include_zero_cells=True,
    verbose=False,
)

# Active cells only (exclude cells with no activity)
ACTIVE_CELLS_CONFIG = SpatialFairnessConfig(
    period_type="hourly",
    include_zero_cells=False,
    min_activity_threshold=1,
    verbose=False,
)
