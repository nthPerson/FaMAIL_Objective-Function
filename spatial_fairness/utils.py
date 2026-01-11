"""
Utility functions for the Spatial Fairness term.

This module contains:
- Gini coefficient computation
- Data loading and preprocessing
- Service rate calculations
- Validation functions
"""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict


# =============================================================================
# GINI COEFFICIENT COMPUTATION
# =============================================================================

def compute_gini(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient for a distribution of values.
    
    The Gini coefficient measures inequality in a distribution:
    - G = 0: Perfect equality (all values identical)
    - G = 1: Maximum inequality (one entity has everything)
    
    Formula:
        G = 1 + (1/n) - (2 / (n² * μ)) * Σ(n-i+1) * x_(i)
    
    where x_(i) is the i-th smallest value.
    
    Args:
        values: Array of non-negative values
        
    Returns:
        Gini coefficient in [0, 1]
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    
    # Handle edge cases
    if n == 0:
        return 0.0
    
    if n == 1:
        return 0.0
    
    total = np.sum(values)
    if total == 0:
        return 0.0  # No inequality among zeros
    
    mean_value = total / n
    
    # Sort values in ascending order
    sorted_values = np.sort(values)
    
    # Compute weighted sum: Σ(n-i+1) * x_(i)
    # Weights are [n, n-1, n-2, ..., 2, 1]
    weights = np.arange(n, 0, -1)
    weighted_sum = np.sum(weights * sorted_values)
    
    # Gini formula
    gini = 1.0 + (1.0 / n) - (2.0 / (n * n * mean_value)) * weighted_sum
    
    # Ensure result is in valid range
    return float(np.clip(gini, 0.0, 1.0))


def compute_gini_alternative(values: np.ndarray) -> float:
    """
    Alternative Gini computation using mean absolute difference.
    
    Formula:
        G = Σ|xi - xj| / (2 * n² * μ)
    
    This is mathematically equivalent but useful for verification.
    
    Args:
        values: Array of non-negative values
        
    Returns:
        Gini coefficient in [0, 1]
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    
    if n == 0 or n == 1:
        return 0.0
    
    total = np.sum(values)
    if total == 0:
        return 0.0
    
    mean_value = total / n
    
    # Compute mean absolute difference directly
    # Using pairwise differences
    sorted_values = np.sort(values)
    
    # For sorted values, sum of |xi - xj| can be computed efficiently
    # |xi - xj| for sorted = xj - xi for j > i
    # Sum = 2 * sum_i (2*i - n - 1) * x_i for i from 0 to n-1
    indices = np.arange(n)
    weights = 2 * indices - n + 1
    mad = np.sum(weights * sorted_values)
    
    gini = mad / (n * n * mean_value)
    
    return float(np.clip(gini, 0.0, 1.0))


def compute_lorenz_curve(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Lorenz curve for a distribution.
    
    The Lorenz curve plots cumulative share of values against
    cumulative share of population (sorted from lowest to highest).
    
    Args:
        values: Array of non-negative values
        
    Returns:
        Tuple of (x_values, y_values) for the Lorenz curve
        where x = cumulative population share [0, 1]
        and y = cumulative value share [0, 1]
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    
    total = np.sum(values)
    if total == 0:
        # Perfect equality line when all zeros
        return np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1)
    
    sorted_values = np.sort(values)
    cumsum = np.cumsum(sorted_values)
    
    # Prepend 0 for starting point
    x = np.concatenate([[0], np.arange(1, n + 1) / n])
    y = np.concatenate([[0], cumsum / total])
    
    return x, y


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_pickup_dropoff_counts(filepath: str) -> Dict[Tuple[int, int, int, int], Tuple[int, int]]:
    """
    Load pickup/dropoff counts from pickle file.
    
    Args:
        filepath: Path to pickup_dropoff_counts.pkl
        
    Returns:
        Dictionary mapping (x, y, time, day) -> (pickup_count, dropoff_count)
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def aggregate_counts_by_period(
    data: Dict[Tuple[int, int, int, int], Tuple[int, int]],
    period_type: str = "hourly",
    days_filter: Optional[List[int]] = None,
    time_filter: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[Tuple[Tuple[int, int], Any], int], Dict[Tuple[Tuple[int, int], Any], int]]:
    """
    Aggregate pickup/dropoff counts by spatial cell and time period.
    
    Args:
        data: Raw pickup_dropoff_counts data
        period_type: One of "time_bucket", "hourly", "daily", "all"
        days_filter: Optional list of days to include (1-6)
        time_filter: Optional tuple of (start_bucket, end_bucket) to include
        
    Returns:
        Tuple of (pickups, dropoffs) where each is a dict:
        {((x, y), period): count}
    """
    pickups: Dict[Tuple[Tuple[int, int], Any], int] = defaultdict(int)
    dropoffs: Dict[Tuple[Tuple[int, int], Any], int] = defaultdict(int)
    
    for key, counts in data.items():
        x, y, time_bucket, day = key
        pickup_count, dropoff_count = counts
        
        # Apply filters
        if days_filter is not None and day not in days_filter:
            continue
        
        if time_filter is not None:
            if time_bucket < time_filter[0] or time_bucket > time_filter[1]:
                continue
        
        # Determine period based on period_type
        if period_type == "time_bucket":
            period = (time_bucket, day)
        elif period_type == "hourly":
            hour = (time_bucket - 1) // 12  # Convert to 0-23
            period = (hour, day)
        elif period_type == "daily":
            period = day
        elif period_type == "all":
            period = "all"
        else:
            raise ValueError(f"Unknown period_type: {period_type}")
        
        cell = (x, y)
        pickups[(cell, period)] += pickup_count
        dropoffs[(cell, period)] += dropoff_count
    
    return dict(pickups), dict(dropoffs)


def get_unique_periods(
    pickups: Dict[Tuple[Tuple[int, int], Any], int],
    dropoffs: Dict[Tuple[Tuple[int, int], Any], int],
) -> List[Any]:
    """
    Get list of unique time periods from aggregated data.
    
    Args:
        pickups: Aggregated pickup counts
        dropoffs: Aggregated dropoff counts
        
    Returns:
        Sorted list of unique periods
    """
    periods = set()
    for key in pickups.keys():
        periods.add(key[1])  # key is ((x, y), period)
    for key in dropoffs.keys():
        periods.add(key[1])
    
    return sorted(periods, key=lambda p: (p,) if not isinstance(p, tuple) else p)


def compute_period_duration_days(
    period: Any,
    period_type: str,
    num_days: float = 21.0,
) -> float:
    """
    Compute the duration of a period in days.
    
    Args:
        period: The period identifier
        period_type: Type of period ("time_bucket", "hourly", "daily", "all")
        num_days: Total number of days in dataset
        
    Returns:
        Duration in days
    """
    if period_type == "time_bucket":
        # Each time bucket is 5 minutes = 1/288 of a day
        return 1.0 / 288.0
    elif period_type == "hourly":
        # Each hour is 1/24 of a day
        return 1.0 / 24.0
    elif period_type == "daily":
        # Each day is 1 day
        return 1.0
    elif period_type == "all":
        # All data = total days
        return num_days
    else:
        raise ValueError(f"Unknown period_type: {period_type}")


# =============================================================================
# SERVICE RATE COMPUTATION
# =============================================================================

def compute_service_rates_for_period(
    pickups: Dict[Tuple[Tuple[int, int], Any], int],
    dropoffs: Dict[Tuple[Tuple[int, int], Any], int],
    period: Any,
    grid_dims: Tuple[int, int],
    num_taxis: int,
    period_duration_days: float,
    include_zero_cells: bool = True,
    data_is_one_indexed: bool = True,
    min_activity_threshold: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Departure Service Rate (DSR) and Arrival Service Rate (ASR)
    for all cells in a given period.
    
    DSR = pickups / (num_taxis * period_duration)
    ASR = dropoffs / (num_taxis * period_duration)
    
    Args:
        pickups: Aggregated pickup counts {((x, y), period): count}
        dropoffs: Aggregated dropoff counts
        period: The period to compute rates for
        grid_dims: Grid dimensions (x_cells, y_cells)
        num_taxis: Number of active taxis
        period_duration_days: Duration of period in days
        include_zero_cells: Whether to include cells with zero activity
        data_is_one_indexed: Whether data uses 1-based indexing
        min_activity_threshold: Minimum total activity to include cell
        
    Returns:
        Tuple of (dsr_values, asr_values) as numpy arrays
    """
    x_dim, y_dim = grid_dims
    normalization = num_taxis * period_duration_days
    
    dsr_list = []
    asr_list = []
    
    # Determine index range based on data indexing
    x_start = 1 if data_is_one_indexed else 0
    y_start = 1 if data_is_one_indexed else 0
    x_end = x_dim + x_start
    y_end = y_dim + y_start
    
    for x in range(x_start, x_end):
        for y in range(y_start, y_end):
            cell = (x, y)
            key = (cell, period)
            
            pickup_count = pickups.get(key, 0)
            dropoff_count = dropoffs.get(key, 0)
            
            total_activity = pickup_count + dropoff_count
            
            # Apply activity threshold
            if not include_zero_cells and total_activity == 0:
                continue
            
            if total_activity < min_activity_threshold:
                continue
            
            dsr = pickup_count / normalization if normalization > 0 else 0
            asr = dropoff_count / normalization if normalization > 0 else 0
            
            dsr_list.append(dsr)
            asr_list.append(asr)
    
    return np.array(dsr_list), np.array(asr_list)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_pickup_dropoff_data(
    data: Dict[Tuple[int, int, int, int], Tuple[int, int]]
) -> Tuple[bool, List[str]]:
    """
    Validate pickup/dropoff counts data.
    
    Args:
        data: Raw pickup_dropoff_counts data
        
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    
    if len(data) == 0:
        errors.append("Data is empty")
        return False, errors
    
    # Check structure
    sample_key = list(data.keys())[0]
    sample_value = list(data.values())[0]
    
    if not isinstance(sample_key, tuple) or len(sample_key) != 4:
        errors.append(f"Keys must be 4-tuples, got: {type(sample_key)}")
    
    if not isinstance(sample_value, (tuple, list)) or len(sample_value) != 2:
        errors.append(f"Values must be 2-element sequences, got: {type(sample_value)}")
    
    if errors:
        return False, errors
    
    # Check ranges
    x_vals = [k[0] for k in data.keys()]
    y_vals = [k[1] for k in data.keys()]
    time_vals = [k[2] for k in data.keys()]
    day_vals = [k[3] for k in data.keys()]
    
    # Check for negative counts
    neg_pickups = sum(1 for v in data.values() if v[0] < 0)
    neg_dropoffs = sum(1 for v in data.values() if v[1] < 0)
    
    if neg_pickups > 0:
        errors.append(f"{neg_pickups} entries have negative pickup counts")
    if neg_dropoffs > 0:
        errors.append(f"{neg_dropoffs} entries have negative dropoff counts")
    
    # Report statistics (not errors, just info)
    stats = {
        'x_range': (min(x_vals), max(x_vals)),
        'y_range': (min(y_vals), max(y_vals)),
        'time_range': (min(time_vals), max(time_vals)),
        'day_range': (min(day_vals), max(day_vals)),
        'total_keys': len(data),
    }
    
    return len(errors) == 0, errors


def get_data_statistics(
    data: Dict[Tuple[int, int, int, int], Tuple[int, int]]
) -> Dict[str, Any]:
    """
    Compute statistics about the pickup/dropoff data.
    
    Args:
        data: Raw pickup_dropoff_counts data
        
    Returns:
        Dictionary of statistics
    """
    if len(data) == 0:
        return {'error': 'Empty data'}
    
    pickups = [v[0] for v in data.values()]
    dropoffs = [v[1] for v in data.values()]
    
    x_vals = [k[0] for k in data.keys()]
    y_vals = [k[1] for k in data.keys()]
    time_vals = [k[2] for k in data.keys()]
    day_vals = [k[3] for k in data.keys()]
    
    return {
        'total_keys': len(data),
        'total_pickups': sum(pickups),
        'total_dropoffs': sum(dropoffs),
        'pickup_stats': {
            'mean': np.mean(pickups),
            'std': np.std(pickups),
            'min': min(pickups),
            'max': max(pickups),
            'nonzero_count': sum(1 for p in pickups if p > 0),
        },
        'dropoff_stats': {
            'mean': np.mean(dropoffs),
            'std': np.std(dropoffs),
            'min': min(dropoffs),
            'max': max(dropoffs),
            'nonzero_count': sum(1 for d in dropoffs if d > 0),
        },
        'spatial': {
            'x_range': (min(x_vals), max(x_vals)),
            'y_range': (min(y_vals), max(y_vals)),
            'unique_cells': len(set((k[0], k[1]) for k in data.keys())),
        },
        'temporal': {
            'time_range': (min(time_vals), max(time_vals)),
            'day_range': (min(day_vals), max(day_vals)),
            'unique_days': len(set(day_vals)),
        },
    }
