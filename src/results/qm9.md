## Results on QM9 using Graph Convolutional Network

On the test set,

| Model  | MSE | MAE| RMSE | R2 | Training Time | Inference Time |
|:------:|:---:|:--:|:----:|:--:|:-------------:|:--------------:|
| **Graph Convolutional Network (GCN)** | 551.7446 | 5.6506 | 8.7458 | 0.9489 | 3727.264 s | 5.344 s |
| **GCN with MC Dropout** | 2686.2145 | 15.0652 | 22.7457 | 0.7062 | 2443.845 s| 237.161 s|
| **GCN with Ensemble MC Dropout** | 3608.5917 | 19.2580 | 26.6933 | 0.5899 | 23662.529 s| 65445.637 s|
| **GCN with DeepEnsembles** | 1918.4546 | 13.6765 | 17.0839 | 0.8584 | 24124.511 s | 79.940 s|
| **GCN with SQR, and ONC** | 1852.0177 | 13.1211 | 19.0315 | 0.8003 | 3036.067 s | 13.224 s|
| **Bayesian Graph Neural Network** | - | - | - | - | s | s|

On the validation set,

| Model  | MSE | MAE| RMSE | R2 | Inference Time |
|:------:|:---:|:--:|:----:|:--:|:--------------:|
| **Graph Convolutional Network (GCN)** | 580.3300 | 5.7023 | 8.9507 | 0.946 | 5.226 s|
| **GCN with MC Dropout** | 2766.0936 | 15.2489 | 23.0954 | 0.7038 | 234.212 s|
| **GCN with Ensemble MC Dropout** | 3680.7207 | 19.3682 | 26.9978 | 0.5896 | 62776.858 s|
| **GCN with DeepEnsembles** | 1919.9521 | 13.6147 | 17.1124 | 0.8603 | 48.511 s|
| **GCN with SQR, and ONC** | 1967.7104 | 13.3432 | 19.4575 | 0.79762 | 12.824 s|
| **Bayesian Graph Neural Network** | - | - | - | - | s|