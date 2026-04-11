import torch
from torch import optim

from .model import DefaultModel, train_only
from .estimate import mean_fn, covariance_fn, inner_product
from .utils import split_data


def IrregPCA(k, data, device=None,
                 epochs=600, lr=1e-3, patience=300, on_epoch=None):

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
    if d < 1:
        raise ValueError("Expected data with at least 3 columns: [idx, loc..., obs].")

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
