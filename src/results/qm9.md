## Results on QM9 using Graph Convolutional Network

On the test set,

| Model  | MSE | MAE| RMSE | R2 | Training Time | Inference Time |
|--------|:--------------:|:------------:|:---------:|:------:|
| **Graph Convolutional Network (GCN)** |  |  |  |  |  s |  s |
| **GCN with MC Dropout** |  |  |  |  |  s|  s|
| **GCN with Ensemble MC Dropout** |  |  |  |  |  s|  s|
| **GCN with DeepEnsembles** |  |  |  |  |  s |  s|
| **GCN with SQR, and ONC** |  |  |  |  | s |  s|
| **Bayesian Graph Neural Network** | - | - | - | - | s | s|

On the validation set,

| Model  | MSE | MAE| RMSE | R2 | Inference Time |
|--------|:--------------:|:------------:|:---------:|:------:|
| **Graph Convolutional Network (GCN)** | 12935.578 | 69.485 | 113.734 | 0.735 | 0.233 s|
| **GCN with MC Dropout** | 12843.093 | 77.335 | 113.327 | 0.737 | 12.632 s|
| **GCN with Ensemble MC Dropout** | 18676.164 | 107.890 | 136.660 | 0.618 | 2817.519 s|
| **GCN with DeepEnsembles** | 14686.526  | 81.682 | 121.187 | 0.699  |  2.558 s|
| **GCN with SQR, and ONC** |  |  |  |  |   s|
| **Bayesian Graph Neural Network** | - | - | - | - | s|