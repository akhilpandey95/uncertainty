## Results on QM8 using Graph Convolutional Network

On the test set,

| Model  | MSE | MAE| RMSE | R2 | Training Time | Inference Time |
|--------|:--------------:|:------------:|:---------:|:------:|
| **Graph Convolutional Network (GCN)** | 0.0007 | 0.0146 | 0.0239 | 0.7596 | 457.601 s | 1.108 s |
| **GCN with MC Dropout** | 0.0011 | 0.0159 | 0.0290 | 0.6318 | 492.520 s| 38.007 s|
| **GCN with Ensemble MC Dropout** | 0.0012 | 0.0162 | 0.0301 | 0.6051 | 5134.569 s| 9637.756 s|
| **GCN with DeepEnsembles** | 0.0008 | 0.0145 | 0.0239 | 0.7554 | 4905.302 s | 40.172 s|
| **GCN with SQR, and ONC** |  |  |  |  | s |  s|
| **Bayesian Graph Neural Network** | - | - | - | - | s | s|

On the validation set,

| Model  | MSE | MAE| RMSE | R2 | Inference Time |
|--------|:--------------:|:------------:|:---------:|:------:|
| **Graph Convolutional Network (GCN)** | 0.0007 | 0.0148 | 0.0234 | 0.7852 | 1.003 s|
| **GCN with MC Dropout** | 0.0012 | 0.0166 | 0.0297 | 0.6365 | 37.307 s|
| **GCN with Ensemble MC Dropout** | 0.0013 | 0.0168 | 0.0309 | 0.6079 | 9603.512 s|
| **GCN with DeepEnsembles** | 0.0007 | 0.0147 | 0.0236 | 0.7792 | 6.802 s|
| **GCN with SQR, and ONC** |  |  |  |  |   s|
| **Bayesian Graph Neural Network** | - | - | - | - | s|