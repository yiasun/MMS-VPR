#!/usr/bin/env python3
import os, glob, sys
from nbformat import read, write
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(path, nb_path, timeout=600):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': path}})
    out_path = os.path.join(path, 'executed', os.path.basename(nb_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        write(nb, f)
    print(f"✔️  {nb_path} → {out_path}")

def main():
    root = os.path.abspath(os.path.dirname(__file__))
    nb_dir = root
    notebooks = glob.glob(os.path.join(nb_dir, '*.ipynb'))
    for nb in notebooks:
        run_notebook(nb_dir, nb)
    print("🎉 All done.")

if __name__ == '__main__':
    main()
