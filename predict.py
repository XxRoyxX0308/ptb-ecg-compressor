## linear model v0 = 0, linear model v1 = 1
model_select = 1
## model epochs
model_eps = 17
## specific mat data
specific_data = [18]



import torch
from torch.utils.data import DataLoader

from dataset import PTBDataset
from linear_model_V0 import LinearModelV0
from linear_model_V1 import LinearModelV1

from pathlib import Path
import numpy as np
from tqdm import tqdm


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def predict(model_info):
    ptb_dataset = PTBDataset(
        data_type=model_info.data_type,
        feature_len=model_info.feature_len,
        is_standard=model_info.is_standard,
        is_paper_zip=model_info.is_paper_zip,
        specific_datas=specific_data,
    )

    dataloader = DataLoader(
        ptb_dataset,
        batch_size=model_info.batch,
        shuffle=False,
        pin_memory=True,
    )


    torch.set_default_dtype(model_info.data_type_torch)

    model = model_info.Model(
        n_input=model_info.feature_len,
        n_hidden=model_info.num_hidden,
        n_output=model_info.num_output,
    ).to(device)

    model.load_state_dict(
        torch.load(
            Path.joinpath(model_info.models_dir, f"{model_info.epochs-1}_eps.pt"),
            map_location=device,
        )
    )
    

    errors = []
    train_loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for index, (feature, label) in train_loop:
        feature = feature.to(device)
        
        pred = model(feature)
        pred = torch.squeeze(pred, 1)
        
        pred = pred.cpu().detach().numpy().reshape(-1, 1)
        if model_info.is_standard:
            pred = ptb_dataset.scaler.inverse_transform(pred)
        pred = np.around(pred)

        label = label.cpu().detach().numpy()
        if model_info.is_standard: label = ptb_dataset.scaler.inverse_transform(label)
        label = np.around(label)
        
        error = np.absolute(pred - label).astype("int")
        errors.append(error)

    errors = np.concatenate(errors).reshape(-1)
    print(np.sum(errors)/len(errors))

    compression_len = 0
    for i in range(len(errors)-3-1):
        mean = np.mean(errors[i:i+3])
        if mean == 0:
            compression_len += 3
            continue

        log2 = np.floor(np.log2(mean))
        compression_len += errors[i+3] % np.pow(2, log2) + 2

    compression_ratio = 16*(len(errors)-3) / compression_len
    print(compression_ratio)


if __name__ == '__main__':
    if model_select == 0:
        model_info = LinearModelV0()
    elif model_select == 1:
        model_info = LinearModelV1()

    model_info.epochs = model_eps+1

    predict(model_info)