#!/usr/bin/env python3
"""
This script executes Jupyter notebooks in the `models/` folder with optional parameter injection via Papermill.
It will automatically tag the first code cell as 'parameters' if missing, so you do not need manual JSON edits.

Supported args:
  --model          notebook name(s) without .ipynb
  --in_channels    input channels override (int)
  --embed_dim      embedding dimension override (int)
  --num_layers     number of transformer layers override (int)
  --num_heads      number of attention heads override (int)
  --mlp_dim        MLP hidden dimension override (int)
  --dropout        dropout rate override (float)
  --text_dim       text vector dimension override (int)
  --lr             learning rate override (float)
  --lr_bert        BERT learning rate override (float)
  --gamma          learning rate decay factor override (float)
  --n_estimators   random forest n_estimators override (int)
  --C              SVM regularization C override (float)
  --k              k-NN neighbors override (int)
  --hidden         MLP hidden units override (int)
  --hidden_channels GNN hidden channels override (int)
  --test_size      test split override (float)
  --batch_size     batch size override (int)
  --epochs         epochs override (int)
  --timeout/-t     execution timeout
"""
import os
import glob
import sys
import argparse
import tempfile
import papermill as pm
import nbformat


def preprocess_notebook(nb_path, work_dir):
    nb = nbformat.read(nb_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            tags = cell.metadata.get('tags', [])
            if 'parameters' not in tags:
                tags.append('parameters')
                cell.metadata['tags'] = tags
            break
    tmp_handle, tmp_path = tempfile.mkstemp(suffix='.ipynb', dir=work_dir)
    os.close(tmp_handle)
    nbformat.write(nb, tmp_path)
    return tmp_path


def run_notebook(path, nb_path, out_path, timeout, parameters):
    cwd = os.getcwd()
    os.chdir(path)
    try:
        tmp_nb = preprocess_notebook(nb_path, path)
        pm.execute_notebook(
            input_path=tmp_nb,
            output_path=out_path,
            parameters=parameters,
            kernel_name='python3',
            timeout=timeout
        )
        os.remove(tmp_nb)
        print(f"✔️  {nb_path} → {out_path}  (params: {parameters})")
    finally:
        os.chdir(cwd)


def main():
    project_root = os.path.abspath(os.path.dirname(__file__))
    nb_dir = os.path.join(project_root, 'models')

    notebooks = sorted(glob.glob(os.path.join(nb_dir, '*.ipynb')))
    model_names = [os.path.splitext(os.path.basename(nb))[0] for nb in notebooks]

    parser = argparse.ArgumentParser(
        description="Run or parameterize notebooks under models/")
    parser.add_argument('--model',          '-m', nargs='+', choices=model_names,
                        help="Which model(s) to run (without .ipynb)")
    parser.add_argument('--in_channels',   type=int,   default=None, help="Input channels")
    parser.add_argument('--embed_dim',     type=int,   default=None, help="Embedding dimension")
    parser.add_argument('--num_layers',    type=int,   default=None, help="Number of transformer layers")
    parser.add_argument('--num_heads',     type=int,   default=None, help="Number of attention heads")
    parser.add_argument('--mlp_dim',       type=int,   default=None, help="MLP hidden dimension")
    parser.add_argument('--dropout',       type=float, default=None, help="Dropout rate")
    parser.add_argument('--text_dim',      type=int,   default=None, help="Text vector dimension")
    parser.add_argument('--lr',            type=float, default=None, help="Learning rate override")
    parser.add_argument('--lr_bert',       type=float, default=None, help="BERT learning rate override")
    parser.add_argument('--gamma',         type=float, default=None, help="LR decay factor")
    parser.add_argument('--n_estimators',  type=int,   default=None, help="RandomForest n_estimators")
    parser.add_argument('--C',             type=float, default=None, help="SVM regularization C")
    parser.add_argument('--k',             type=int,   default=None, help="k-NN neighbors")
    parser.add_argument('--hidden',        type=int,   default=None, help="MLP hidden units")
    parser.add_argument('--hidden_channels', type=int, default=None, help="GNN hidden channels")
    parser.add_argument('--test_size',     type=float, default=None, help="Test split override")
    parser.add_argument('--batch_size',    type=int,   default=None, help="Batch size override")
    parser.add_argument('--num_epochs',        type=int,   default=None, help="Epochs override")
    parser.add_argument('--timeout',       '-t', type=int,   default=600,
                        help="Execution timeout per notebook")
    args = parser.parse_args()

    params = {}
    for attr in ['in_channels','embed_dim','num_layers','num_heads','mlp_dim','dropout',
                 'text_dim','lr','lr_bert','gamma','n_estimators','C','k','hidden',
                 'hidden_channels','test_size','batch_size','num_epochs']:
        val = getattr(args, attr)
        if val is not None:
            params[attr] = val

    to_run = []
    if args.model:
        for name in args.model:
            to_run.append(os.path.join(nb_dir, f"{name}.ipynb"))
    else:
        to_run = notebooks
    if not to_run:
        print("⚠️  No notebooks to run.")
        sys.exit(1)

    out_root = os.path.join(project_root, 'executed')
    os.makedirs(out_root, exist_ok=True)

    for nb_path in to_run:
        out_path = os.path.join(out_root, os.path.basename(nb_path))
        run_notebook(nb_dir, nb_path, out_path, timeout=args.timeout, parameters=params)

    print("🎉 All done.")

if __name__ == '__main__':
    main()
