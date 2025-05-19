# N2V CAREamics - Convallaria

This script demonstrates Noise2Void (N2V) denoising on the Convallaria dataset using the CAREamics API.

- [Original repository](https://github.com/CAREamics/careamics)

## Environment
```bash
conda create -n careamics python=3.9
conda activate careamics
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install careamics
```

## Configuration

```python
config = create_n2v_configuration(
    experiment_name="convallaria_n2v",
    data_type="array",
    axes="YX",
    patch_size=(64, 64),
    batch_size=16,
    num_epochs=100,
    masked_pixel_percentage=0.2,
    struct_n2v_axis="none",
)
```

## Results 
todo