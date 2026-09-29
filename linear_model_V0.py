import torch
from torch import nn
from pathlib import Path 

class LinearModelV0():
    def __init__(self):
        self.name = "Linear_Model_V0"

        abs_dir = Path(__file__).parent.absolute()
        models_dir = Path.joinpath(abs_dir, "models", self.name)
        models_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = models_dir

        self.data_type = "float64"
        self.data_type_torch = torch.float64
        self.is_standard = True
        self.is_paper_zip = False

        self.feature_len = 10
        self.num_hidden = 10
        self.num_output = 1

        self.batch = 512
        self.epochs = 10
        self.test_eps = 1
        self.save_eps = 1

        self.learning_rate = 0.001
        self.is_scheduler = True
        if self.is_scheduler:
            self.step_size = 20
            self.gamma = 0.25

        self.loss_function = nn.MSELoss()

    def optimizer_load(self, params):
        self.optimizer = torch.optim.AdamW(
            params=params,
            lr=self.learning_rate,
        )

        if self.is_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.step_size,
                gamma=self.gamma
            )
    
    def r2_score(self, true, pred):
        true_mean = torch.mean(true)
        ss_res = torch.sum((true - pred) ** 2)
        ss_tot = torch.sum((true - true_mean) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2
    

    class Model(nn.Module):
        def __init__(self, n_input, n_hidden, n_output):
            super().__init__()
            self.stack_layer = nn.Sequential(
                nn.Linear(n_input, n_hidden).double(),
                nn.Linear(n_hidden, n_hidden).double(),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden).double(),
                nn.ReLU(),
                nn.Linear(n_hidden, n_output).double(),
            )
        
        def forward(self, x):
            return self.stack_layer(x)