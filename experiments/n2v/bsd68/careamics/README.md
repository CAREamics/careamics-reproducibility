# N2V CAREamics

- [Original repository](https://github.com/CAREamics/careamics)


## Environment

```bash
conda create -n careamics python=3.9
conda activate torch-n2v
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Configuration

```python
{   
    'unet_n_depth': 2,
    'unet_n_first': 96,
    'train_epochs': 10,
    'train_steps_per_epoch': 400,
    'train_batch_size': 128,
    'train_learning_rate': 0.0004,
}
```

## Results

From paper:
TODO

From script:
TODO

