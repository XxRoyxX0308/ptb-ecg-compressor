import torch
from torch import nn
from torch.utils.data import DataLoader

from pathlib import Path
from tqdm import tqdm


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(
        epochs: int,
        test_eps: int,
        save_eps: int,
        models_dir: Path,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        acc_fn,
        optim: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler = None,
        is_test: bool = True,
        is_optuna: bool = False,
    ):
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)

    model = model.to(device)
    for epoch in range(epochs):
        model.train()

        train_loss = 0
        train_acc = 0
        train_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=not is_optuna)
        for index, (feature, label) in train_loop:
            feature = feature.to(device)
            label = label.to(device)

            pred = model(feature)
            
            loss = loss_fn(pred, label)
            train_loss += loss.item()
            train_acc += acc_fn(label, pred)
            
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            train_loop.set_postfix(acc=f"{train_acc/(index+1)*100:.3f}%", loss=f"{train_loss/(index+1):.3e}")

        if sched:
            sched.step()

        if (epoch+1)%save_eps == 0:
            torch.save(
                model.state_dict(),
                Path.joinpath(models_dir, f"{epoch}_eps.pt")
            )
        

        if is_test and (((epoch+1)%test_eps == 0 and not is_optuna) or (epoch+1) == epochs):
            with torch.inference_mode():
                model.eval()

                test_loss = 0
                test_acc = 0
                test_loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader), leave=not is_optuna)
                for index, (feature, label) in test_loop:
                    feature = feature.to(device)
                    label = label.to(device)

                    pred = model(feature)

                    loss = loss_fn(pred, label)
                    test_loss += loss.item()
                    test_acc += acc_fn(label, pred)

                    test_loop.set_description(f"Test Epoch [{epoch+1}/{epochs}]")
                    test_loop.set_postfix(acc=f"{test_acc/(index+1)*100:.3f}%", loss=f"{test_loss/(index+1):.3e}")

                
                if is_optuna: return test_acc/(index+1)