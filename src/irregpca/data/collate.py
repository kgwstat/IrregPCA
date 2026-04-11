from __future__ import annotations

from typing import Any

import torch


def grouped_collate_fn(
    samples: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Collate a list of per-sample dicts into a grouped batch.

    Parameters
    ----------
    samples : list of dict
        Each dict has keys ``sample_id``, ``locations``, ``values``.

    Returns
    -------
    dict with keys:
        - ``sample_ids``: shape ``(batch_size,)``
        - ``locations``: shape ``(N_obs, d)`` — concatenated
        - ``values``: shape ``(N_obs,)`` — concatenated
        - ``group_offsets``: shape ``(batch_size,)`` — start index per sample
        - ``group_lengths``: shape ``(batch_size,)`` — observations per sample
    """
    sample_ids = torch.stack([s["sample_id"] for s in samples])
    lengths = torch.tensor([s["values"].shape[0] for s in samples], dtype=torch.long)
    offsets = torch.zeros(len(samples), dtype=torch.long)
    if len(samples) > 1:
        offsets[1:] = lengths[:-1].cumsum(0)

    all_locs = torch.cat([s["locations"] for s in samples], dim=0)
    all_vals = torch.cat([s["values"] for s in samples], dim=0)

    return {
        "sample_ids": sample_ids,
        "locations": all_locs,
        "values": all_vals,
        "group_offsets": offsets,
        "group_lengths": lengths,
    }
