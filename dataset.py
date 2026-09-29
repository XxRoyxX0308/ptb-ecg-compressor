from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import numpy as np
import scipy.io
import math

class PTBDataset(Dataset):
    def __init__(
            self,
            datas_dir: Path = Path.joinpath(Path(__file__).parent.absolute(), "datas"),
            data_type: str = np.float32,
            feature_len: int = 10,
            is_standard: bool = True,
            is_paper_zip: bool = False,
            specific_datas: list = None,
        ):
        super().__init__()

        self.data_dirs = datas_dir.iterdir()
        self.feature_len = feature_len
        self.is_paper_zip = is_paper_zip

        datas = []
        for data_dir in self.data_dirs:
            data = scipy.io.loadmat(data_dir)["val"][0]
            datas.append(data)
        all_data = np.concatenate(datas).reshape(-1, 1)

        if specific_datas != None:
            datas = []
            for specific_data in specific_datas:
                data_dir = Path(f"datas/{specific_data}.mat")
                datas.append(scipy.io.loadmat(data_dir)["val"][0])
        
        if is_standard:
            scaler = StandardScaler().fit(all_data)
            self.scaler = scaler
            
            for i in range(len(datas)):
                datas[i] = scaler.transform(datas[i].reshape(-1, 1))
                datas[i] = datas[i].astype(data_type)
        else:
            for i in range(len(datas)):
                datas[i] = datas[i].reshape(-1, 1).astype(data_type)

        self.datas = datas


        self.data_len = []
        
        for i in range(len(self.datas)):
            self.data_len.append((0 if i == 0 else self.data_len[i-1]) + len(self.datas[i]) - self.feature_len)

    def __len__(self):
        return self.data_len[len(self.datas)-1]

    def __getitem__(self, idx):
        for i in range(len(self.data_len)):
            if self.data_len[i]>idx:
                idx -= 0 if i == 0 else self.data_len[i-1]

                if self.is_paper_zip:
                    X = self.datas[i][idx:idx+self.feature_len].reshape(-1)
                    y = self.adaptive_predict(X)
                    if math.isnan(y):
                        return self.__getitem__(idx-12)
                    y = self.datas[i][idx+self.feature_len] - round(y)
                    
                    return X, y
                else:
                    X = self.datas[i][idx:idx+self.feature_len].reshape(-1)
                    y = self.datas[i][idx+self.feature_len]
                    
                    return X, y
            

    def linear_predict(self, x, order):
        if order == 1:
            return x[-1]
        elif order == 2:
            return 2 * x[-1] - x[-2]
        elif order == 3:
            return 3 * x[-1] - 3 * x[-2] + x[-3]
        else:
            raise ValueError("Unsupported order")

    def adaptive_predict(self, x):
        diff12 = abs(x[-1] - x[-2])
        diff34 = abs(x[-3] - x[-4])
        if diff12 < 5 and diff34 < 5:  # Threshold = 5, example
            return self.linear_predict(x, 1)
        else:
            pred2 = self.linear_predict(x, 2)
            pred3 = self.linear_predict(x, 3)
            e2 = abs(x[-1] - pred2)
            e3 = abs(x[-1] - pred3)
            w2 = 2 ** (10 - e2)
            w3 = 2 ** (10 - e3)
            return (w2 * pred2 + w3 * pred3) / (w2 + w3)