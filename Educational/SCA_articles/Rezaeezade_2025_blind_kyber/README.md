# Notebook Setup (VS Code)

This folder contains a simple Python notebook:

- `hello_world_numpy.ipynb`

Use the steps below to create a virtual environment, install dependencies, and run the notebook in VS Code.

## Prerequisites

- Python 3 installed
- VS Code installed
- VS Code extensions:
  - `Python` (by Microsoft)
  - `Jupyter` (by Microsoft)

## 1. Open this folder

After cloning/downloading the repository, open a terminal in this folder (the folder that contains `requirements.txt` and `hello_world_numpy.ipynb`).

## 2. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Open in VS Code

```bash
code .
```

Then in VS Code:

1. Open `hello_world_numpy.ipynb`.
2. Click the kernel/interpreter selector in the top-right of the notebook.
3. Choose the Python interpreter from `.venv`.

## 5. Run the notebook

Use `Run All` in the notebook toolbar, or run cells one by one.

## Troubleshooting

If `.venv` does not appear in the kernel list:

```bash
python -m ipykernel install --user --name cissa-venv --display-name "Python (.venv)"
```

Then restart VS Code and select `Python (.venv)` as the notebook kernel.
