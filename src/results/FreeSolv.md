## Results on FreeSolv using Graph Convolutional Network

On the test set,

| Model  | MSE | MAE| RMSE | R2 | Training Time | Inference Time |
|:------:|:---:|:--:|:----:|:--:|:-------------:|:--------------:|
| **Graph Convolutional Network (GCN)** | 1.140 | 0.792 | 1.068 | 0.918 |  21.017 s |  0.284 s |
| **GCN with MC Dropout** | 2.269 | 1.065 | 1.506 | 0.837 |  19.360 s|  1.514 s|
| **GCN with Ensemble MC Dropout** | 2.732 | 1.233 | 1.653 | 0.804 | 318.162 s| 356.191 s|
| **GCN with DeepEnsembles** | 1.701 | 0.982 | 1.304 | 0.878 | 307.463 s | 32.654 s|
| **GCN with SQR, and ONC** | 1.067 | 0.845 | 1.033 | 0.923 | 39.181 s | 0.381 s|
| **Bayesian Graph Neural Network** | - | - | - | - | s | s|

On the validation set,

| Model  | MSE | MAE| RMSE | R2 | Inference Time |
|:------:|:---:|:--:|:----:|:--:|:--------------:|
| **Graph Convolutional Network (GCN)** | 1.484 | 0.901 | 1.218 | 0.878 | 0.022 s|
| **GCN with MC Dropout** | 2.313 | 1.155 | 1.521 | 0.810 | 1.034 s|
| **GCN with Ensemble MC Dropout** | 2.843 | 1.376 | 1.686 | 0.766 | 320.530 s|
| **GCN with DeepEnsembles** | 2.396 | 1.171 | 1.548 | 0.803 | 0.238 s|
| **GCN with SQR, and ONC** | 1.994 | 1.146 | 1.412 | 0.836 | 0.055 s|
| **Bayesian Graph Neural Network** | - | - | - | - | s|

