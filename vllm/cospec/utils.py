import numpy as np
from typing import List


def remove_outliers(data: List[float]) -> float:
    """Remove outliers using IQR method and return mean of remaining values."""
    if not data:
        return 0.0

    data = np.array(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

    if len(filtered_data) == 0:
        return np.mean(data)

    return np.mean(filtered_data)
