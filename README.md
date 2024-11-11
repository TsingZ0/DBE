# Introduction

This is the implementation of our paper *[Eliminating Domain Bias for Federated Learning in Representation Space](https://arxiv.org/abs/2311.14975)* (accepted by NeurIPS 2023). It can improve bi-directional knowledge transfer between the server and clients. We show the code of the representative FedAvg+DBE (`FedDBE`). 

- [[NeurIPS] DBE.pptx](./[NeurIPS]%20DBE.pptx)
- [DBEPoster.pdf](./DBEPoster.pdf)

**Takeaway**: By eliminating domain bias in the feature extractor, we address catastrophic forgetting during local training, enhancing the generalization ability. Consequently, the global module can swiftly adapt to a new client. 

**Citation**

```
@inproceedings{zhang2023eliminating,
  title={Eliminating Domain Bias for Federated Learning in Representation Space},
  author={Jianqing Zhang and Yang Hua and Jian Cao and Hao Wang and Tao Song and Zhengui XUE and Ruhui Ma and Haibing Guan},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=nO5i1XdUS0}
}
```

# Datasets and Environments

Due to the file size limitation, we only upload the fmnist dataset with the default practical setting ($\beta=0.1$). Please refer to our project [PFLlib](https://github.com/TsingZ0/PFLlib) for other datasets and environments settings. 


# System

- `main.py`: configurations of **FedDBE**. 
- `run_me.sh`: command lines to start **FedDBE**. 
- `env_linux.yaml`: python environment to run **FedDBE** on Linux. 
- `./flcore`: 
    - `./clients/clientDBE.py`: the code on the client. 
    - `./servers/serverDBE.py`: the code on the server. 
    - `./trainmodel/models.py`: the code for models. 
- `./utils`:
    - `data_utils.py`: the code to read the dataset. 

# Training and Evaluation

All codes corresponding to **FedDBE** are stored in `./system`. Just run the following commands.

```
cd ./system
sh run_me.sh
```
