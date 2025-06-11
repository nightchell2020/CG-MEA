![header](https://capsule-render.vercel.app/api?type=Venom&color=gradient&height=200&text=CG-MEA&fontSize=80)

### CG-MEA : Confidence Guided Masked EEG Autoencoder
<img src="https://img.shields.io/badge/Python-2B2728?style=plastic&logo=Python&logoColor=3776AB"/> <img src="https://img.shields.io/badge/PyTorch-2B2728?style=plastic&logo=PyTorch&logoColor=EE4C2C"/>


## 1. Getting Started

### Requirements

- Installation of Conda (refer to <https://www.anaconda.com/products/distribution>)
- Nvidia GPU with CUDA support

> Note: we tested the code in the following environments.
>
> |    **OS**    | **Python** | **PyTorch** | **CUDA** |
> |:------------:|:----------:|:-----------:|:--------:|
> |  Windows 10  |   3.9.12   |   1.11.0    |   11.3   |
> | Ubuntu 20.04 |   3.9.21   |    2.2.1    |   12.1   |

### Installation

(optional) Create and activate a Conda environment.

```bash
  conda create -n caueeg python=3.9
  conda activate caueeg
```

Install PyTorch library (refer to <https://pytorch.org/get-started/locally/>).

```bash
  conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

Install other necessary libraries.

```bash
  pip install -r requirements.txt
```

### Preparation of the [CAUEEG](https://github.com/ipis-mjkim/caueeg-dataset) dataset

> ❗ Note: The use of the CAUEEG dataset is allowed for only academic and research purposes 👩‍🎓👨🏼‍🎓.

- For full access of the CAUEEG dataset, follow the instructions specified in <https://github.com/ipis-mjkim/caueeg-dataset>.
- Download, unzip, and move the whole dataset files into [local/datasets/](local/datasets/).

> 💡 Note: We provide `caueeg-dataset-test-only` at [[link 1]](https://drive.google.com/file/d/1P3CbLY7h9O1CoWEWsIZFbUKoGSRUkTA1/view?usp=sharing) or [[link 2]](http://naver.me/xzLCBwFp) to test our research. `caueeg-dataset-test-only` has the 'real' test splits of two benchmarks (*CAUEEG-Dementia* and *CAUEEG-Abnormal*) but includes the 'fake' train and validation splits.

---

## 2. Usage

### PreTrain Self-Supervision

Train a CG-MEA model on the training set of *CAUEEG-Dementia* from scratch using the following command:

```bash
  python run_mae_train.py data=caueeg-dementia ssl=1D-MAE-B train=base_train
```

Similarly, train a model on the training set of *CAUEEG-Abnormal* from scratch using:

```bash
  python run_mae_train.py data=caueeg-abnormal ssl=1D-MAE-B train=base_train
```

Or, you can use [this Jupyter notebook](notebook/02_Train.ipynb).

If you encounter a GPU memory allocation error or wish to adjust the balance between memory usage and training speed, you can specify the minibatch size by adding the `++model.minibatch=INTEGER_NUMBER` option to the command as shown below:

```bash
  python run_train.py data=caueeg-dementia model=1D-ResNet-18 train=base_train ++model.minibatch=32
```

```bash
  python run_train.py data=caueeg-abnormal model=1D-ResNet-18 train=base_train ++model.minibatch=32
```

Thanks to [Hydra](https://hydra.cc/) support, the model, hyperparameters, and other training details are easily tuned using or modifying config files.

```bash
python run_train.py data=caueeg-dementia model=2D-VGG-19 train=base_train
```

For speed-up, we recommend using the `PyArrow.feather` file format than using directly `EDF`, which can be done:

```bash
python ./datasets/convert_file_format.py  # it takes a few minutes
python run_train.py data=caueeg-dementia model=2D-VGG-19 train=base_train ++data.file_format=feather
```

### Evaluation

Evaluation can be conducted using [this Jupyter notebook](notebook/03_Evaluate.ipynb) (or [another notebook](notebook/03_Evaluate_Test_Only.ipynb) for `caueeg-dataset-test-only` case)

To use the pre-trained model, download the checkpoint file from [here](#model-summary), and move it to [local/checkpoint](local/checkpoint/) directory (e.g., `local/checkpoint/1vc80n1f/checkpoint.pt` for the 1D-VGG-19 model on the *CAUEEG-Dementia* benchmark).

---
