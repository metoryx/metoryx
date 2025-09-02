import jax.numpy as jnp
from jax import tree


class AverageMeter:
    """Stores and computes the average of multiple metrics."""

    def __init__(self):
        """Initialize the AverageMeter."""
        self.sums = {}
        self.counts = {}

    def update(
        self,
        values: dict[str, float],
        weight: float = 1.0,
    ) -> "AverageMeter":
        """Update the meter with new values.

        Args:
            values: Dictionary of metric names and their values.
            weight: Weight for the values.

        Returns:
            Self for method chaining.
        """
        for k, v in values.items():
            scalar = float(jnp.mean(v))
            self.sums[k] = self.sums.get(k, 0.0) + scalar * weight
            self.counts[k] = self.counts.get(k, 0.0) + weight
        return self

    def compute(self) -> dict[str, float]:
        """Compute the average for each metric.

        Returns:
            Dictionary of metric names and their averages.
        """
        return tree.map(lambda s, c: s / c, self.sums, self.counts)
