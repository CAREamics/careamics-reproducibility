# N2V TensorFlow Implementation

- [Original repository](https://github.com/juglab/n2v)
- [Original notebook](https://github.com/juglab/n2v/blob/main/examples/2D/denoising2D_BSD68/BSD68_reproducibility.ipynb)


## Environment

```bash
conda create -n tf-n2v python=3.9
conda activate tf-n2v
conda install -c conda-forge cudatoolkit=11.3 cudnn=8.1
pip install tensorflow=3.10 n2v microscopy-portfolio
```

## Running the script

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
python n2v_tf_BSD68.py
```

## Configuration

```python
{
  'means': ['110.72957232412905'],
  'stds': ['63.656060106500874'],
  'n_dim': 2,
  'axes': 'YXC',
  'n_channel_in': 1,
  'n_channel_out': 1,
  'unet_residual': True,
  'unet_n_depth': 2,
  'unet_kern_size': 3,
  'unet_n_first': 96,
  'unet_last_activation': 'linear',
  'unet_input_shape': (None, None, 1),
  'train_loss': 'mse',
  'train_epochs': 10,
  'train_steps_per_epoch': 400,
  'train_learning_rate': 0.0004,
  'train_batch_size': 128,
  'train_tensorboard': True,
  'train_checkpoint': 'weights_best.h5',
  'train_reduce_lr': {'factor': 0.5, 'patience': 10},
  'batch_norm': True,
  'n2v_perc_pix': 0.198,
  'n2v_patch_shape': (64, 64),
  'n2v_manipulator': 'uniform_withCP',
  'n2v_neighborhood_radius': 2,
  'single_net_per_channel': False,
  'blurpool': False,
  'skip_skipone': False,
  'structN2Vmask': None,
  'probabilistic': False
}
```

## Results

From paper:
27.71

From notebook:
27.81

From script:
26.7 +/- 2.53
