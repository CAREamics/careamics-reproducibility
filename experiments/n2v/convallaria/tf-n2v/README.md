# N2V TensorFlow Implementation

- [Original repository](https://github.com/juglab/n2v)

## Environment

```bash
conda create -n tf-n2v python=3.9
conda activate tf-n2v
conda install -c conda-forge cudatoolkit=11.3 cudnn=8.1
pip install tensorflow=2.10 n2v microscopy-portfolio
```

## Running the script

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
python n2v_tf_convallaria.py
```

## Configuration
