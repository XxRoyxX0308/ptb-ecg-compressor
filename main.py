## linear model v0 = 0, linear model v1 = 1
model_select = 1
## Use all the data for train or not
train_all_data = False
## turing optimisation
use_optuna = False

if use_optuna:
    ## number of trials
    num_trials = 100
    
    

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataset import PTBDataset
from linear_model_V0 import LinearModelV0
from linear_model_V1 import LinearModelV1
import train
import predict

from pathlib import Path
import optuna
from optuna.visualization import plot_slice
from plotly.io import show


if model_select == 0:
    model_info = LinearModelV0()
elif model_select == 1:
    model_info = LinearModelV1()

torch.set_default_dtype(model_info.data_type_torch)


abs_dir = Path(__file__).parent.absolute()

optuna_result_dir = Path.joinpath(abs_dir, "optuna_result")
optuna_result_dir.mkdir(parents=True, exist_ok=True)


ptb_dataset = PTBDataset(
    data_type=model_info.data_type,
    feature_len=model_info.feature_len,
    is_standard=model_info.is_standard,
    is_paper_zip=model_info.is_paper_zip,
)

train_dataset, test_dataset = random_split(
    dataset=ptb_dataset,
    lengths=[1, 0] if train_all_data else [0.8, 0.2],
    generator=torch.manual_seed(12),
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=model_info.batch,
    shuffle=True,
    pin_memory=True,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=model_info.batch,
    shuffle=True,
    pin_memory=True,
)


torch.manual_seed(12)
torch.cuda.manual_seed(12)

if not use_optuna:
    model = model_info.Model(
        n_input=model_info.feature_len,
        n_hidden=model_info.num_hidden,
        n_output=model_info.num_output,
    )
    model_info.optimizer_load(model.parameters())
    
    train.train(
        epochs=model_info.epochs,
        test_eps=model_info.test_eps,
        save_eps=model_info.save_eps,
        models_dir=model_info.models_dir,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        model=model,
        loss_fn=model_info.loss_function,
        acc_fn=model_info.r2_score,
        optim=model_info.optimizer,
        sched=model_info.scheduler if model_info.is_scheduler else None,
        is_test=not train_all_data,
        is_optuna=False,
    )
    
    predict.predict(model_info)

else:
    def objective(trial):
        model_info.num_hidden = trial.suggest_int('num_hidden', 1, 20)
        model_info.learning_rate = trial.suggest_float('learning_rate', 0.000001, 0.0003)

        model = model_info.Model(
            n_input=model_info.feature_len,
            n_hidden=model_info.num_hidden,
            n_output=model_info.num_output,
        )
        model_info.optimizer_load(model.parameters())

        acc = train.train(
            epochs=model_info.epochs,
            test_eps=model_info.test_eps,
            save_eps=model_info.save_eps,
            models_dir=model_info.models_dir,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            model=model,
            loss_fn=model_info.loss_function,
            acc_fn=model_info.r2_score,
            optim=model_info.optimizer,
            sched=model_info.scheduler if model_info.is_scheduler else None,
            is_test=not train_all_data,
            is_optuna=True,
        )

        return acc
    

    study = optuna.create_study(direction='maximize')
    study.optimize(
        objective,
        n_trials=num_trials,
        n_jobs=-1
    )

    trial = study.best_trial

    print(f"Accuracy: {trial.value}")
    print(f"Best hyperparameters: {trial.params}")

    fig = plot_slice(study)
    show(fig)