import numpy as np


# 파일 읽기 및 NumPy 배열로 변환
def load_numpy_array(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        clean_lines = [line.replace('[', '').replace(']', '').strip() for line in lines]
        data = []
        for line in clean_lines:
            if line:  # 빈 줄 제외
                row = [float(x) for x in line.split()]
                data.append(row)
        # NumPy 배열로 변환
        return np.array(data)

file_path = 'arima_geant_pred.txt'
data = load_numpy_array(file_path)
print(data.shape)
data = np.expand_dims(data.reshape(2150, 1, 72), axis=-1)
data = np.swapaxes(data, 0, 1)

print(data.shape)
np.save('test_pred_geant_arima.npy', data)