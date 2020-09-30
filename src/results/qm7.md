## Results on QM7 using Graph Convolutional Network

On the test set,

| Model  | MSE | MAE| RMSE | R2 | Training Time | Inference Time |
|:------:|:---:|:--:|:----:|:--:|:-------------:|:--------------:|
| **Graph Convolutional Network (GCN)** | 15512.530 | 75.499 | 124.549 | 0.681 | 169.664 s | 0.544 s |
| **GCN with MC Dropout** | 16025.649 | 82.345 | 126.592 | 0.671 | 148.591 s| 12.996 s|
| **GCN with Ensemble MC Dropout** | 22103.611 | 113.213 | 148.672 | 0.546 | 1709.633 s| 2869.730 s|
| **GCN with DeepEnsembles** | 17170.168 | 87.521 | 131.034 | 0.647 | 1521.202 s | 35.111 s|
| **GCN with SQR, and ONC** | 23855.9678 | 112.0663 | 154.4537 | 0.5106 | 185.351 s | 1.040 s|
| **Bayesian Graph Neural Network** | - | - | - | - | s | s|

On the validation set,

| Model  | MSE | MAE| RMSE | R2 | Inference Time |
|:------:|:---:|:--:|:----:|:--:|:--------------:|
| **Graph Convolutional Network (GCN)** | 12935.578 | 69.485 | 113.734 | 0.735 | 0.233 s|
| **GCN with MC Dropout** | 12843.093 | 77.335 | 113.327 | 0.737 | 12.632 s|
| **GCN with Ensemble MC Dropout** | 18676.164 | 107.890 | 136.660 | 0.618 | 2817.519 s|
| **GCN with DeepEnsembles** | 14686.526  | 81.682 | 121.187 | 0.699  |  2.558 s|
| **GCN with SQR, and ONC** | 22509.4616 | 110.3797 | 150.0315 | 0.5396 | 0.534 s|
| **Bayesian Graph Neural Network** | - | - | - | - | s|
