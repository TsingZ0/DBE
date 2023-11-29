# Introduction

This is the implementation of our paper *[Eliminating Domain Bias for Federated Learning in Representation Space](https://arxiv.org/abs/2311.14975)* (accepted by NeurIPS 2023). It can improve bi-directional knowledge transfer between the server and clients. We show the code of the representative FedAvg+DBE (`FedAvgDBE`). 

- [[NeurIPS] DBE.pptx](./[NeurIPS]%20DBE.pptx)
- [DBEPoster.pdf](./DBEPoster.pdf)


# Dataset

Due to the file size limitation, we only upload the fmnist dataset with the default practical setting ($\beta=0.1$). Please refer to our project [PFLlib](https://github.com/TsingZ0/PFLlib). 


# System (based on PFL-Non-IID)

- `main.py`: configurations of **FedAvgDBE**. 
- `run_me.sh`: command lines to start **FedAvgDBE**. 
- `env_linux.yaml`: python environment to run **FedAvgDBE** on Linux. 
- `./flcore`: 
    - `./clients/clientAvgDBE.py`: the code on the client. 
    - `./servers/serverAvgDBE.py`: the code on the server. 
    - `./trainmodel/models.py`: the code for models. 
- `./utils`:
    - `data_utils.py`: the code to read the dataset. 

# Simulation

## Environments
With the installed [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh), we can run this platform in a conda virtual environment called *fl_torch*. 
```
conda env create -f env_linux.yaml # for Linux
```


## Training and Evaluation

All codes corresponding to **FedAvgDBE** are stored in `./system`. Just run the following commands.

```
cd ./system
sh run_me.sh
```