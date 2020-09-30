## Results on Lipophilicity using Graph Convolutional Network

On the test set,

| Model  | MSE | MAE| RMSE | R2 | Training Time | Inference Time |
|:------:|:---:|:--:|:----:|:--:|:-------------:|:--------------:|
| **Graph Convolutional Network (GCN)** | 0.410 | 0.475 | 0.641 | 0.716 | 146.419 s | 0.553 s |
| **GCN with MC Dropout** | 0.574 | 0.529 | 0.727 | 0.635 | 125.980 s| 12.727 s|
| **GCN with Ensemble MC Dropout** | 0.488 | 0.548 | 0.698 | 0.663 | 1361.137 s| 3297.722 s|
| **GCN with DeepEnsembles** | 0.443 | 0.512 | 0.665 | 0.694 | 1357.874 s | 34.396 s|
| **GCN with SQR, and ONC** | 0.432 | 0.500 | 0.657 | 0.702 | 175.929 s | 0.919 s|
| **Bayesian Graph Neural Network** | - | - | - | - | s | s|

On the validation set,

| Model  | MSE | MAE| RMSE | R2 | Inference Time |
|:------:|:---:|:--:|:----:|:--:|:--------------:|
| **Graph Convolutional Network (GCN)** | 0.428 | 0.482 | 0.654 | 0.696 | 0.234 s|
| **GCN with MC Dropout** | 0.575 | 0.552 | 0.743 | 0.608 | 12.542 s|
| **GCN with Ensemble MC Dropout** | 0.622 | 0.614 | 0.788 | 0.559 | 3312.145 s|
| **GCN with DeepEnsembles** | 0.473 | 0.514 | 0.687 | 0.665 | 2.714 s|
| **GCN with SQR, and ONC** | 0.505 | 0.540 | 0.710 | 0.642 | 0.588 s|
| **Bayesian Graph Neural Network** | - | - | - | - | s|

