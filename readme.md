# PySlimDL
PySlimDL基于冗余数据识别和移除，在深度学习任务中通过识别和移除冗余的训练数据，在保证模型最终准确率的前提下尽可能减少模型训练时间。
## 项目目录结构
```
.
├── bin
├── docs
├── readme.md
├── requirements.txt
└── src # 项目源代码
    ├── __init__.py
    ├── altiny.py # alexnet training on tinyimage
    ├── dataloader # 训练数据的载入和数据预处理
    │   ├── __init__.py
    │   ├── cifarloader.py # CIFAR10数据集
    │   ├── loader.py # MNIST
    │   ├── tinyimage.py # TinyImage
    │   └── utils.py
    ├── lemnist.py # LeNet5 + MNIST 
    ├── mnist.py
    ├── models # 定义相关模型，包括LeNet5, AlexNet, ResNet-x(18,34,50,101,152),MobileNetV2, SqueezeNet
    │   ├── __init__.py
    │   ├── alexnet.py
    │   ├── lenet.py
    │   ├── mobilenet.py
    │   ├── resnet.py
    │   └── squeezenet.py
    ├── multicifar.py # 在CIFAR10上训练AlexNet，ResNet-x, MobileNetV2, SqueezeNet
    ├── redishelper # 缓存训练过程的基本信息以及模型参数和梯度
    │   ├── __init__.py
    │   ├── gossiphelper.py # TODO gossipgrad
    │   └── redishelper.py # GoSGD
    ├── run.sh # 运行脚本，需传入需要运行的python脚本以及python脚本参数等
    ├── squcifar.py # SqueezeNet + CIFAR10
    ├── stoppy.sh # 停止用run.sh启动的训练过程并清除日志
```
