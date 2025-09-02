import time
from collections import defaultdict

import jax.numpy as jnp
from jax.typing import ArrayLike


class AverageMeter:
    """Stores and computes the weighted average of multiple metrics over time."""

    def __init__(self, *, with_timer: bool = False, timer_key: str = "elapsed_time"):
        """Initializes the AverageMeter.

        Args:
            with_timer: Whether to track elapsed time from the meter reset to the compute call.
            timer_key: The key name for the elapsed time metric.
        """
        self.with_timer = with_timer
        self.timer_key = timer_key
        self.reset()

    def update(self, metric_dict: dict[str, ArrayLike], n: int = 1, *, prefix: str = "") -> None:
        """Updates the meter with new metric values.

        Args:
            metric_dict: A dictionary where keys are metric names and values are
                the corresponding metric values (can be arrays).
            n: The number of samples the metrics correspond to (default is 1).
            prefix: A string prefix to add to each metric name (default is "").

        Returns:
            The updated AverageMeter instance.
        """
        for metric_name, value in metric_dict.items():
            metric_name = f"{prefix}{metric_name}"
            avg_value = float(jnp.mean(value))
            self.total_dict[metric_name] += avg_value * n
            self.count_dict[metric_name] += n

    def compute(self) -> dict[str, float]:
        """Computes the average for each metric.

        Returns:
            A dictionary where keys are metric names and values are their average.
        """
        computed = {
            metric_name: self.total_dict[metric_name] / self.count_dict[metric_name]
            for metric_name in self.total_dict
        }

        if self.with_timer and self.start_time is not None:
            computed[self.timer_key] = time.time() - self.start_time

        return computed

    def reset(self) -> None:
        """Resets the meter to initial state."""
        self.total_dict = defaultdict(float)
        self.count_dict = defaultdict(int)

        self.start_time = None
        if self.with_timer:
            self.start_time = time.time()
