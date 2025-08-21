# Project Setup and Usage Guide

## 1. Check Python Version

Check your version with:
```bash
python --version
```
or  
```bash
python3 --version
```

---

## 2. Create and Activate a Virtual Environment (venv)

### Linux / MacOS
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Required Dependencies
Install all packages from `requirements.txt`:
```bash
pip install tqdm
pip install numpy
pip install wandb
pip install torch
pip install torchvision
pip install tensorboardX
pip install tensorboard
pip install timm
pip install backpack-for-pytorch


```

---

## 4. Train the Model
Example:
```bash
python main.py --algo swinjscc --train_flag True --channel_type AWGN --lr 0.0001 --out-e 200

```
Notes:

--algo swinjscc → specifies the algorithm to use (SwinJSCC).

--train_flag True → enables training mode.

--channel_type AWGN → sets the channel type to Additive White Gaussian Noise.

--lr 0.0001 → learning rate.

--out-e 200 → number of output epochs (training runs for 200 epochs).
The trained model will be saved in: out/checkpoints/
---

## 6. Test the Model
Example:
```bash
python3 test_cifar.py
```

---
## 7. Calling python
### Install dependencies:

```bash
pip install pybind11
```

### Compile C++ source with embedded Python
```bash
g++ -O3 -Wall -std=c++17 -fPIC   $(python3 -m pybind11 --includes)   main.cpp -o main   $(python3-config --ldflags --embed)
```

### Run
```bash
./main
```

---

## 8. Deactivate the Virtual Environment
```bash
deactivate
```

---
