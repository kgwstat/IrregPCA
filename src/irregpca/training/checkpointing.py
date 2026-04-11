from __future__ import annotations

import torch


class Checkpoint:
    """Early-stopping checkpoint for a single model.

    Tracks the best validation loss and saves a copy of the model state
    at each improvement. The :meth:`maybe_update` method returns ``True``
    when a new best is found.

    Parameters
    ----------
    model : torch.nn.Module
        The model to checkpoint.
    patience : int
        Number of epochs without improvement before :attr:`should_stop`
        becomes ``True``. ``patience=0`` means stop immediately after the
        first epoch without improvement.
    """

    def __init__(self, model: torch.nn.Module, patience: int) -> None:
        self._model = model
        self._patience = patience
        self._best_state: dict = {
            name: v.detach().clone() for name, v in model.state_dict().items()
        }
        self.best_valid_loss: float = float("inf")
        self.best_epoch: int = 0
        self._wait: int = 0
        self.should_stop: bool = False

    def maybe_update(self, valid_loss: float, epoch: int) -> bool:
        """Check if ``valid_loss`` is a new best; update checkpoint if so.

        Parameters
        ----------
        valid_loss : float
        epoch : int

        Returns
        -------
        bool
            ``True`` if a new best was found.
        """
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.best_epoch = epoch
            self._best_state = {
                name: v.detach().clone()
                for name, v in self._model.state_dict().items()
            }
            self._wait = 0
            return True
        self._wait += 1
        if self._wait > self._patience:
            self.should_stop = True
        return False

    def restore(self) -> None:
        """Restore the model to its best-seen state.

        This is always called after the training loop finishes, regardless
        of whether early stopping triggered or the loop ran to completion.
        This fixes the known bug where the last state (not the best) was
        returned when the loop exhausted all epochs.
        """
        self._model.load_state_dict(self._best_state)
