# SpaceMutation

The code demo of SpaceMutation.

## Installation

We have tested SpaceMutation based on Python 3.11 on Ubuntu 20.04, theoretically it should also work on other operating
systems. To get the dependencies of SpaceMutation, it is sufficient to run the following command.
`pip install -r requirements.txt`

## The structure of the repository
This directory includes implementations of GRASP, mutation operators, prompt generators, the generation of test oracles, and the implementation of dynamic power regulation.
</br>The training and testing data for MNIST, CIFAR-100, and CIFAR-10 can be directly loaded from PyTorch by simply setting up the directory.
## GRASP Strategy

We provided six heuristic algorithms to search for the optimal mutated model among all the mutated DNN models generated.

| DNN        | Ours                                                     | GA                                                       | PSO                                                       | SA                                                       | A*                                                       | HC                                                       | ACO                                                       |
|------------|----------------------------------------------------------|----------------------------------------------------------|-----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|-----------------------------------------------------------|
| AlexNet    | [AlexNet-SM](https://www.hostize.com/zh/v/9uvEpHWsfs)    | [AlexNet-GA](https://www.hostize.com/zh/v/5asRMu1s-v)    | [AlexNet-PSO](https://www.hostize.com/zh/v/yUCFJTDUDm)    | [AlexNet-SA](https://www.hostize.com/zh/v/R-xJm1PyQc)    | [AlexNet-A*](https://www.hostize.com/zh/v/2W5Dn07pvu)    | [AlexNet-HC](https://www.hostize.com/zh/v/DvihLzJD_w)    | [AlexNet-ACO](https://www.hostize.com/zh/v/L4Ld-v1Oem)    |
| LeNet-5    | [LeNet-5-SM](https://www.hostize.com/zh/v/NQ3nBCbeQQ)    | [LeNet-5-GA](https://www.hostize.com/zh/v/iZWOANMXBU)    | [LeNet-5-PSO](https://www.hostize.com/zh/v/uHzO__L0WB)    | [LeNet-5-SA](https://www.hostize.com/zh/v/FJbOpnsfmg)    | [LeNet-5-A*](https://www.hostize.com/zh/v/h3xqfWaRsR)    | [LeNet-5-HC](https://www.hostize.com/zh/v/9Q7TRP6Umc)    | [LeNet-5-ACO](https://www.hostize.com/zh/v/nbPu4FT1Ww)    |
| VGG-16     | [VGG-16-SM](https://www.hostize.com/zh/v/SHQSoVTlCj)     | [VGG-16-GA](https://www.hostize.com/zh/v/1x0VNAP-5X)     | [VGG-16-PSO](https://www.hostize.com/zh/v/L3Tr73_XHz)     | [VGG-16-SA](https://www.hostize.com/zh/v/zVZWldpt7u)     | [VGG-16-A*](https://www.hostize.com/zh/v/edm21RkGWM)     | [VGG-16-HC](https://www.hostize.com/zh/v/wLkxKz_IPh)     | [VGG-16-ACO](https://www.hostize.com/zh/v/PIbqBJeG0O)     |
| ResNet-50  | [ResNet-50-SM](https://www.hostize.com/zh/v/DSkPYfEGWM)  | [ResNet-50-GA](https://www.hostize.com/zh/v/UVzmvH-yl_)  | [ResNet-50-PSO](https://www.hostize.com/zh/v/Ekp4JIPI8p)  | [ResNet-50-SA](https://www.hostize.com/zh/v/0_gI83CsNr)  | [ResNet-50-A*](https://www.hostize.com/zh/v/93Tzhq9mBF)  | [ResNet-50-HC](https://www.hostize.com/zh/v/V6z4yEMpTA)  | [ResNet-50-ACO](https://www.hostize.com/zh/v/q0JETpgnsU)  |
| UNet       | [UNet-SM](https://www.hostize.com/zh/v/u5kZ0p84kx)       | [UNet-GA](https://www.hostize.com/zh/v/edhhq_rsrJ)       | [UNet-PSO](https://www.hostize.com/zh/v/3y4beygDby)       | [UNet-SA](https://www.hostize.com/zh/v/Ntnmp2YGbZ)       | [UNet-A*](https://www.hostize.com/zh/v/Lfjn8cWuAr)       | [UNet-HC](https://www.hostize.com/zh/v/ttV8rqf8HY)       | [UNet-ACO](https://www.hostize.com/zh/v/kncRuNGzM4)       |
| ShuffleNet | [ShuffleNet-SM](https://www.hostize.com/zh/v/vpnYKfln8V) | [ShuffleNet-GA](https://www.hostize.com/zh/v/JZyMtN_hoC) | [ShuffleNet-PSO](https://www.hostize.com/zh/v/zT3FKT5QKe) | [ShuffleNet-SA](https://www.hostize.com/zh/v/9vFYURN4_o) | [ShuffleNet-A*](https://www.hostize.com/zh/v/fv_CkP593s) | [ShuffleNet-HC](https://www.hostize.com/zh/v/TGkTc41sKx) | [ShuffleNet-ACO](https://www.hostize.com/zh/v/yuOlgST4c6) |

You can run the `python main_grasp.py` file to see the evaluation results of the generated optimal SSHOM. 
Prerequisite is to have the LLM (Qwen2-7B-Instruct) environment set up and the PyTorch dependencies installed.
</br>If you want to run experiments with other models, you just need to change the model name in `main_grasp.py`. For now, SpaceMutation has not provided an easier way to conduct experiments.

## Prompts Optimization for Test Oracle Generation
You can run `python main_grasp.py`, and during the program execution, test oracles will be generated. You just need to set the directory where the test oracles will be stored. The prerequisite is to have the LLM (Qwen2-7B-Instruct) deployed.
You can install [Ollama](https://github.com/ollama/ollama) locally and then proceed with the deployment of the LLM.
</br>After the deployment is complete, you can run the command:
</br>`ollama run qwen2:7b`
</br>We provide the original data samples and the statistical results of the generated test oracles in the data/test_oracle directory.
## Power consumption
In the power_consumption/process/run.py file, you can set the communication rounds.
Running python energy.py will initiate the power consumption monitoring during the mutation testing process. 
</br>These devices include the master node, equipped with Intel i7-13260H CPU and NVIDIA RTX 4060 GPU; 2 working nodes, equipped with Intel i7-10700K CPU and NVIDIA RTX 3080 GPU; 2 working nodes, equipped with Atlas 200I DK A2 CPU.
</br>Considering the limitations of hardware devices, we have provided the raw data of actual power consumption and the final experimental results for all experiments in the power_consumption_result directory.
