## Results on ESOL using Graph Convolutional Network

On the test set,

| Model  | MSE | MAE| RMSE | R2 | Training Time | Inference Time |
|:------:|:---:|:--:|:----:|:--:|:-------------:|:--------------:|
| **Graph Convolutional Network (GCN)** | 0.743 | 0.613 | 0.862 | 0.810 | 42.385 s |  0.316 s |
| **GCN with MC Dropout** | 0.937 | 0.738 | 0.968 | 0.761 | 32.804 s| 2.952 s|
| **GCN with Ensemble MC Dropout** | 0.726 | 0.675 | 0.852 | 0.814 | 419.915 s|  694.805 s|
| **GCN with DeepEnsembles** | 0.652 | 0.590 | 0.807 | 0.833 | 473.882 s | 33.527 s|
| **GCN with SQR, and ONC** | 0.691 | 0.608 | 0.831 | 0.823 | 55.550 s | 0.426 s|
| **Bayesian Graph Neural Network** | - | - | - | - | s | s|

On the validation set,

| Model  | MSE | MAE| RMSE | R2 | Inference Time |
|:------:|:---:|:--:|:----:|:--:|:--------------:|
| **Graph Convolutional Network (GCN)** | 0.564 | 0.542 | 0.751 | 0.867 | 0.054 s|
| **GCN with MC Dropout** | 0.735 | 0.657 | 0.857 | 0.827 | 2.738 s|
| **GCN with Ensemble MC Dropout** | 0.757 | 0.640 | 0.870 | 0.821 | 673.545 s|
| **GCN with DeepEnsembles** | 0.616 | 0.579 | 0.785 | 0.855 | 0.577 s|
| **GCN with SQR, and ONC** | 0.680 | 0.615 | 0.824 | 0.840 | 0.124 s|
| **Bayesian Graph Neural Network** | - | - | - | - | s|
