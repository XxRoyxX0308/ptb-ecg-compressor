import numpy as np
import math

def linear_predict(x, order):
    if order == 1:
        return x[-1]
    elif order == 2:
        return 2 * x[-1] - x[-2]
    elif order == 3:
        return 3 * x[-1] - 3 * x[-2] + x[-3]
    else:
        raise ValueError("Unsupported order")

def adaptive_predict(x):
    diff12 = abs(x[-1] - x[-2])
    diff34 = abs(x[-3] - x[-4])
    if diff12 < 5 and diff34 < 5:  # Threshold = 5, example
        return linear_predict(x, 1)
    else:
        pred2 = linear_predict(x, 2)
        pred3 = linear_predict(x, 3)
        e2 = abs(x[-1] - pred2)
        e3 = abs(x[-1] - pred3)
        w2 = 2 ** (10 - e2)
        w3 = 2 ** (10 - e3)
        return (w2 * pred2 + w3 * pred3) / (w2 + w3)

def golomb_rice_encode(errors, k):
    encoded = []
    for e in errors:
        m = 2 * e if e >= 0 else 2 * abs(e) - 1
        q = m >> k
        r = m & ((1 << k) - 1)
        unary = '1' * q + '0'
        binary = format(r, f'0{k}b')
        encoded.append(unary + binary)
    return encoded

def compute_k(errors):
    mae = sum(abs(e) for e in errors[-3:]) / 3
    return max(0, int(math.log2(mae)) if mae != 0 else 0)

# --- 測試資料 ---
ecg = [1000, 1003, 1006, 1010, 1013, 1017, 1021, 1024, 1025]
errors = []
history = ecg[:4]

# 模擬壓縮過程
for i in range(4, len(ecg)):
    pred = adaptive_predict(history)
    err = ecg[i] - round(pred)
    errors.append(err)
    history.append(ecg[i])

# GRC 編碼
k = compute_k(errors)
encoded = golomb_rice_encode(errors, k)

# print("Prediction errors:", errors)
# print("GRC parameter k:", k)
# print("Encoded bitstreams:", encoded)

################################################################

from dataset import PTBDataset

ptb_dataset = PTBDataset(
    data_type="float64",
    feature_len=4,
    is_standard=False,
    specific_datas=[4],
)

err = 0
t = 0
for i in range(ptb_dataset.__len__()):
    pred = adaptive_predict(ptb_dataset[i][0])
    if math.isnan(pred):
        print(ptb_dataset[i][0])
        t += 1
        continue
    err += np.absolute(ptb_dataset[i][1] - round(pred))
print(err/(ptb_dataset.__len__()-t))