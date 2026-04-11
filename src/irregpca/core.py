import torch
from torch import optim

from .model import DefaultModel, train_only
from .estimate import mean_fn, covariance_fn, inner_product
from .utils import split_data


def IrregPCA(k, data, device=None,
                 epochs=600, lr=1e-3, patience=300, on_epoch=None):
    """
    Fit IrregPCA to irregularly observed functional data.

    Parameters
    ----------
    k : int
        Number of leading principal directions to estimate.
    data : torch.Tensor
        Tensor of shape (N, d+2), where each row is one observation:
        [sample_id, location_1, ..., location_d, observation].
        - ``data[:, 0]``   — integer sample IDs
        - ``data[:, 1:-1]``— d-dimensional observation locations
        - ``data[:, -1]``  — scalar observation values
    device : str or torch.device, optional
        Device to run on. Defaults to CUDA/MPS if available, else CPU.
    epochs : int, optional
        Maximum number of training epochs per component (default 600).
    lr : float, optional
        Adam learning rate (default 1e-3).
    patience : int, optional
        Early-stopping patience in epochs (default 300).
    on_epoch : callable, optional
        Optional callback invoked at the end of each epoch with signature
        ``on_epoch(j, epoch, joint_losses_train, joint_losses_valid)``.

    Returns
    -------
    models : list of torch.nn.Module
        ``models[0]`` is the estimated mean; ``models[j]`` for ``j >= 1``
        is the j-th scaled principal direction f_j = v_j * e_j.
    joint_losses_train : list of float
        Joint training loss recorded at each epoch across all components.
    joint_losses_valid : list of float
        Joint validation loss recorded at each epoch across all components.
    """
    if data.ndim != 2:
        raise ValueError(
            f"Expected `data` to be a 2-D tensor, but got {data.ndim}-D. "
            "Expected shape is (N, d+2), with rows as observations and "
            "columns [idx, loc..., obs]."
        )
    if data.shape[1] < 3:
        raise ValueError(
            f"Expected `data` to have at least 3 columns (got {data.shape[1]}). "
            "Expected shape is (N, d+2), with columns [idx, loc..., obs]."
        )
    if data.shape[0] < data.shape[1]:
        raise ValueError(
            f"Expected `data` to have shape (N, d+2), but got shape "
            f"{tuple(data.shape)} where the number of rows ({data.shape[0]}) "
            f"is less than the number of columns ({data.shape[1]}). "
            "You may have passed transposed data of shape (d+2, N)."
        )

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    data = data.to(device)

    d = data.shape[1] - 2

    data_train, data_valid = split_data(data, 0.8)
    models = [DefaultModel(d=d).to(device=device) for _ in range(k+1)]
    
    losses_train, losses_valid = [], []
    joint_losses_train, joint_losses_valid = [], []
    
    # construct loss functions
    lossfn = [None for _ in range(k+1)]
    lossfn[0] = lambda models, data: (
        -mean_fn(models[0], data) 
        + 0.5 * inner_product(models[0], models[0], device=device)
    )    
    for j in range(1, k+1):
        lossfn[j] = (lambda j: 
            lambda models, data: (
                -covariance_fn(models[j], data) 
                + 0.5 * inner_product(models[j], models[j], device=device).pow(2) 
                + (torch.stack([inner_product(models[i], models[j], device=device).pow(2) for i in range(1, j)]).sum() if j > 1 else torch.tensor(0.0, device=device))
            )
            )(j)

    loss_joint = lambda models, data: lossfn[0](models, data) + torch.stack([lossfn[j](models, data) for j in range(1, k+1)]).sum()
    
    # train models sequentially
    for j, model in enumerate(models):
        print(f"\n===> Training model {j}")
        
        train_only(j, models)
        optimizer = optim.Adam(model.parameters(), lr=lr, )

        wait = 0
        min_valid_loss = float("inf")
        best_state = {name: v.detach().clone() for name, v in model.state_dict().items()}

        for epoch in range(epochs):
            
            # train
            model.train()
            optimizer.zero_grad()
            loss_train = lossfn[j](models, data_train)
            loss_train.backward()
            optimizer.step()

            # evaluate
            model.eval()
            with torch.no_grad():
                loss_valid = lossfn[j](models, data_valid)

            # log
            joint_losses_train.append(loss_joint(models, data_train).item())
            joint_losses_valid.append(loss_joint(models, data_valid).item())
            losses_train.append(loss_train.item())
            losses_valid.append(loss_valid.item())

            # early stopping
            if loss_valid < min_valid_loss:
                min_valid_loss = loss_valid
                best_state = {name: v.detach().clone() for name, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1

            if wait > patience:
                print("Early stop.")
                model.load_state_dict(best_state)
                break
            
            # epoch end callback
            if on_epoch is not None:
                on_epoch(j, epoch, joint_losses_train, joint_losses_valid)

    train_only(-1, models)
    
    return models, joint_losses_train, joint_losses_valid
