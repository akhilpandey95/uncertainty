<img src="src/media/uncertainty-logo.png" alt="drawing" width="150px" style="display: block; margin-right: auto"/>

## About
The repository consists of information assosicated with the research on uncertainity quantification for graph neural network architectures. This work was part of an internship at [ANL](https://www.anl.gov/mcs) in the Mathematics and Computer Sciences (MCS) division. The repository has 5 directories:
- `data/` - The directory that consists of the datasets in the form of pickled files
- `literature/` - Summaries of all the scholarly articles that are read through the course of the project.
- `src/` - Code, documentation, and essential information about the models built for the project.
- `scripts/` - Essential scripts necessary for setting up the project.
- `tests/` - Deterministic unit tests for better replication of the results.
- `notebooks/` - Jupyter notebook(s) with sample code showcasing how to utilize the resources of the project.

## Installation

Install the appropriate python packages
```shell
pip install -r requirements.txt
```

## Data
We have used two datasets. The idea was to perform uncertainty quantification on Fully Connected Networks, and Graph Neural Networks. So, for the Fully Connected Networks we generated a `sin` dataset of sample size 1201 with gaussian noise given by, `y = 10 sin(x) + ε`. The dataset looks like this:

<img src="src/media/data.png" alt="data" style="display: block; margin-right: auto"/>

For the Graph Neural Networks, we have used the Quantum Chemistry dataset, `QM7`.

Both of the datasets have single output and would be mapped as regression problems.

## Experiments
We built simple deep neural networks for predicting the response variable. We then estimated the uncertainties in the models predicting the response values. We tried different approaches and as a result some of the methods that we used supported quantification of both aleatoric, and epistemic uncertainties, while the remaining weren't equipped for the same. The approaches we have tried could potentially be categorized into ones that are capable of estimating both aleatoric, and epistemic uncertainties, and others that can only estimate epistemic uncertainties.

*Methods that can approximate both aleatoric and epistemic uncertainty :*
- **Bayesian Neural Network (BNN)** : [Bayesian Learning for Neural Networks by R.M. Neal](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.446.9306&rep=rep1&type=pdf)
- **SQR+ONC** : [Single-Model Uncertainties for Deep Learning by N.Tagasovska, and D. Lopez-Paz](https://arxiv.org/pdf/1811.00908.pdf)
<hr>

*Methods that can approximate only epistemic uncertainty :*
- **DeepEnsembles** : [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles by B. Lakshminarayanan et.al](https://arxiv.org/pdf/1612.01474.pdf)
- **MC Dropout** : [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning by Y. Gal, and Z. Ghahramani](https://arxiv.org/pdf/1506.02142.pdf)
- **Ensemble MC Dropout** : [Understanding Measures of Uncertainty for Adversarial Example Detection by L. Smith, and Y. Gal](https://arxiv.org/pdf/1803.08533.pdf)


For the models predicting the y values of the `sin` dataset, we have used a simple one hidden layer fully connected neural network. The following are the architectural details:

| Model  | Optimum hyper parameters |
|--------|:------------------------:|
| **Epochs** | 200, <br> 1500 for BNN |
| **Batch size** | 128 |
| **Loss Function** | MSE, <br> Gaussian Loss for DeepEnsemble, <br> Negative log likelihood for BNN, <br> Pinball Loss for SQR+ONC |
| **Hidden Layers** | 1 layer with 256 neurons |
| **Optimization function** | Adam with 0.001 learning rate |
| **Activation function(s)** | ReLU for the hidden layer, <br>Linear for the o/p layer |


## Results

Linear regression on the `sin` dataset: 

| Model  | MSE | MAE| RMSE | R2 | Training time | Inference Time |
|--------|:---:|---:|-----:|---:|:-------------:|:--------------:|
| **Fully Connected Network (FCN)** | 1.0053 | 0.8078 | 1.0026 | 0.9813 | 33.4 s| 246 ms| 
| **FCN with MC Dropout** | 1.4041 | 0.9555 | 1.1849 | 0.9740 | 15.7 s| 54.7 s|
| **FCN with Ensemble MC Dropout** | 1.4430 | 0.9697 | 1.2012 | 0.9733 | 5min 56s| 32min 19s|
| **FCN with DeepEnsembles** | 0.9531 | 0.7767 | 0.9763 | 0.9820 | 10min 25s| 1min 50s|
| **FCN with SQR, and ONC** | 1.0929 | 0.8318 | 1.0454 | 0.9793 | **43s** | **10.2s**|
| **Bayesian Neural Network** | 3.3879 | 1.4489 | 1.9418 | 0.953 | 8min 39s| 44s|


Linear regression on the `QM7` dataset :

On the test set,

| Model  | MSE | MAE| RMSE | R2 | Training Time | Inference Time |
|--------|:---:|---:|-----:|---:|:-------------:|:--------------:|
| **Graph Convolutional Network (GCN)** | 15498.9595 | 75.9521 | 124.4948 | 0.6830 | 2min 30s| 444 ms|
| **GCN with MC Dropout** | 23116.5562 | 119.5583 | 152.0412 | 0.5272 | 1min 51s| 8.46 s|
| **GCN with Ensemble MC Dropout** | 23382.3769 | 121.7387 | 152.9129 | 0.5218 | 1h 45s| 6min 46s|
| **GCN with DeepEnsembles** | 15726.9933 | 76.9819 | 125.3840 | 0.6783 | 1h 3min 15s| 2min 59s|
| **GCN with SQR, and ONC** | 21182.1737 | 99.7401 | 145.5409 | 0.5668 | **2min 30s** | **579 ms**|
| **Bayesian Graph Neural Network** | - | - | - | - |

On the validation set,

| Model  | MSE | MAE| RMSE | R2 | Inference Time |
|--------|:---:|---:|-----:|---:|:--------------:|
| **Graph Convolutional Network (GCN)** | 10658.5007 | 68.3195 | 103.2400 | 0.7811 | 153 ms|
| **GCN with MC Dropout** | 17058.3652 | 106.2512 | 130.6076 | 0.6497 | 7.99 s|
| **GCN with Ensemble MC Dropout** | 18296.0994 | 112.2832 | 135.2630 | 0.6243s | 3min 36s|
| **GCN with DeepEnsembles** | 10529.2807 | 67.4484 | 102.5716 | 0.7838 | 4.01 s|
| **GCN with SQR, and ONC** | 17238.2398 | 94.7970 | 131.2944 | 0.6460 | 427 ms|
| **Bayesian Graph Neural Network** | - | - | - | - |


## Author
[Akhil Pandey](https://github.com/akhilpandey95)

## Mentor
[Prasanna Balaprakash](https://github.com/pbalapra)

