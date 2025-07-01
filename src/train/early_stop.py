import numpy as np
import logging

logger = logging.getLogger("training")


class EarlyStopper:
    """
    Implements standard early stopping based on validation loss.

    Stops training when the validation loss has not improved by at least a relative threshold for a given number of 'patience' epochs.

    Args:
        patience (int): How many epochs to wait after last time validation loss improved significantly. Default: 5.
        threshold (float): Minimum relative improvement required to consider the loss as having improved. Should be positive. e.g., 0.01 means a 1% decrease is needed. Default: 0.01.
    """

    def __init__(self, *, patience: int = 5, threshold: float = 0.01):
        if patience < 0 or threshold < 0:
            raise ValueError("Patience and threshold must be non-negative.")
        self.patience = patience
        self.threshold = threshold  # Relative improvement threshold
        self.loss_list = []

    def __call__(self, current_loss):
        """Update the loss list and determine if training should stop."""
        self.loss_list.append(current_loss)
        return self.check_stop()

    def check_stop(self):
        """Check whether we should stop training based on the loss list"""
        best_loss = np.inf
        stall_count = 0
        improvement = 0
        for loss in self.loss_list:
            if not np.isfinite(best_loss):
                best_loss = loss
                continue
            improvement = (best_loss - loss) / abs(best_loss)
            if improvement > self.threshold:
                best_loss = loss
                stall_count = 0
            else:
                stall_count += 1
        if stall_count == 0:
            logger.info(
                f"EarlyStopper: Patience reset. New best loss {best_loss:0.04f}. "
                + f"Improvement was {improvement * 100:0.02f}%."
            )
        else:
            logger.info(
                f"EarlyStopper: Patience {stall_count}/{self.patience}. "
                + f"Improvement was {improvement * 100:0.02f}%."
            )
        return stall_count >= self.patience

    # --- Methods for Saving/Loading State ---

    def get_state(self):
        """Returns a dictionary containing the essential state."""
        return {
            "patience": self.patience,
            "threshold": self.threshold,
            "loss_list": self.loss_list,
        }

    @classmethod
    def from_state(cls, state):
        """Creates a new instance from a saved state dictionary."""
        try:
            # Create instance with saved params
            instance = cls(patience=state["patience"], threshold=state["threshold"])
            # Set the dynamic state
            instance.loss_list = state["loss_list"]
            return instance
        except KeyError as e:
            raise ValueError(f"Missing key in state dictionary for creating instance: {e}")
