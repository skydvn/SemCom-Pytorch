# Project Setup and Usage Guide

## 1. Check Python Version
This project requires **Python 3.8+**.  
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

### Windows (Command Prompt or PowerShell)
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / MacOS
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Deactivate the Virtual Environment
```bash
deactivate
```

---

## 4. Install Required Dependencies
Install all packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 5. Train the Model
Example:
```bash
python train.py --config configs/train_config.yaml
```
Or, depending on your project:
```bash
python main.py --mode train
```

---

## 6. Test the Model
Example:
```bash
python test.py --model_path saved_models/model.pth
```
Or:
```bash
python main.py --mode test
```

---

## 7. Notes
- Always activate the virtual environment before training or testing.
- If a package is missing, install it manually:
```bash
pip install <package_name>
```
- It is recommended to use **Python >= 3.8** to avoid compatibility issues.
