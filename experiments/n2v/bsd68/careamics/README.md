# N2V CAREamics - BSD68
This script demonstrates Noise2Void (N2V) denoising on the BSD68 dataset using the CAREamics API.
- [Original repository](https://github.com/CAREamics/careamics)

## Environment
```bash
conda create -n careamics python=3.11
conda activate careamics
conda install pytorch torchvision pytorch-cuda=12.8 -c pytorch -c nvidia
pip install careamics careamics-portfolio
```

## Configuration
```python
config = create_n2v_configuration(
    experiment_name="n2v_BSD",
    data_type="tiff",
    axes="SYX",
    patch_size=(64, 64),
    batch_size=128,
    num_epochs=50,
    masked_pixel_percentage=0.2,
    struct_n2v_axis="none",
)
```

## Results
Average PSNR: 26.44 ± 2.68
Average MicroSSIM: 0.75 ± 0.07