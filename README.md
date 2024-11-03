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
| AlexNet    | [AlexNet-SM](https://www.hostize.com/zh/v/R-jXKsGzrr)    | [AlexNet-GA](https://www.hostize.com/zh/v/7_HvyY8jYn)    | [AlexNet-PSO](https://www.hostize.com/zh/v/1z2OwzqNE5)    | [AlexNet-SA](https://www.hostize.com/zh/v/R-xJm1PyQc)    | [AlexNet-A*](https://www.hostize.com/zh/v/T0l3fV4Phq)    | [AlexNet-HC](https://www.hostize.com/zh/v/waZu9tAqQD)    | [AlexNet-ACO](https://www.hostize.com/zh/v/73tqnmjoRJ)    |
| LeNet-5    | [LeNet-5-SM](https://www.hostize.com/zh/v/adpd61lxOA)    | [LeNet-5-GA](https://www.hostize.com/zh/v/QoGJzEnCL2)    | [LeNet-5-PSO](https://www.hostize.com/zh/v/cX6uQB5Ggz)    | [LeNet-5-SA](https://www.hostize.com/zh/v/tAsw5tQTsY)    | [LeNet-5-A*](https://www.hostize.com/zh/v/_RVKhK0jJX)    | [LeNet-5-HC](https://www.hostize.com/zh/v/l-O8nEenIp)    | [LeNet-5-ACO](https://www.hostize.com/zh/v/Xd1yuAvscG)    |
| VGG-16     | [VGG-16-SM](https://www.hostize.com/zh/v/UbpT_qfiLy)     | [VGG-16-GA](https://www.hostize.com/zh/v/1x0VNAP-5X)     | [VGG-16-PSO](https://www.hostize.com/zh/v/L3Tr73_XHz)     | [VGG-16-SA](https://www.hostize.com/zh/v/zVZWldpt7u)     | [VGG-16-A*](https://www.hostize.com/zh/v/edm21RkGWM)     | [VGG-16-HC](https://www.hostize.com/zh/v/wLkxKz_IPh)     | [VGG-16-ACO](https://www.hostize.com/zh/v/PIbqBJeG0O)     |
| ResNet-50  | [ResNet-50-SM](https://www.hostize.com/zh/v/M2TAFS4hUw)  | [ResNet-50-GA](https://www.hostize.com/zh/v/awRd_fBbBI)  | [ResNet-50-PSO](https://www.hostize.com/zh/v/clxY5yYYNY)  | [ResNet-50-SA](https://www.hostize.com/zh/v/0_gI83CsNr)  | [ResNet-50-A*](https://www.hostize.com/zh/v/voYwLFDitT)  | [ResNet-50-HC](https://www.hostize.com/zh/v/ZGcl9p3foe)  | [ResNet-50-ACO](https://www.hostize.com/zh/v/tPb-eqjOhw)  |
| UNet       | [UNet-SM](https://www.hostize.com/zh/v/4uVIdGdEKQ)       | [UNet-GA](https://www.hostize.com/zh/v/sHuGDLyU5H)       | [UNet-PSO](https://www.hostize.com/zh/v/3y4beygDby)       | [UNet-SA](https://www.hostize.com/zh/v/1mP0HxFNzb)       | [UNet-A*](https://www.hostize.com/zh/v/rUYzxtKz3_)       | [UNet-HC](https://www.hostize.com/zh/v/ttV8rqf8HY)       | [UNet-ACO](https://www.hostize.com/zh/v/YqRBb__1PP)       |
| ShuffleNet | [ShuffleNet-SM](https://www.hostize.com/zh/v/jZcnGBsVPg) | [ShuffleNet-GA](https://www.hostize.com/zh/v/UbfZeVBLCD) | [ShuffleNet-PSO](https://www.hostize.com/zh/v/16h5weOvg6) | [ShuffleNet-SA](https://www.hostize.com/zh/v/9vFYURN4_o) | [ShuffleNet-A*](https://www.hostize.com/zh/v/4-0MLTDEZD) | [ShuffleNet-HC](https://www.hostize.com/zh/v/9vFYURN4_o) | [ShuffleNet-ACO](https://www.hostize.com/zh/v/_IX7tmk6sQ) |

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
