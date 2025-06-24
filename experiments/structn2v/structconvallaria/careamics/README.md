# StructN2V CAREamics - StructConvallaria

This script demonstrates Structured Noise2Void (StructN2V) denoising on the StructConvallaria dataset using the CAREamics API. StructN2V is designed to handle structured noise patterns that regular N2V cannot remove effectively.

- [Original repository](https://github.com/CAREamics/careamics)

## Environment

```bash
conda create -n careamics python=3.11
conda activate careamics
conda install pytorch torchvision pytorch-cuda=12.8 -c pytorch -c nvidia
pip install careamics careamics-portfolio
```

## Dataset

The StructConvallaria dataset consists of 100 manipulated noisy images with enhanced structured noise patterns, derived from the original Convallaria dataset. It's automatically downloaded via careamics-portfolio.

## Configuration

```python
config = create_n2v_configuration(
    experiment_name="structconvallaria_structn2v",
    data_type="array",
    axes="YX",
    patch_size=(64, 64),
    batch_size=16,
    num_epochs=50,
    masked_pixel_percentage=0.2,
    # StructN2V parameters for handling structured noise
    struct_n2v_axis="horizontal",
    struct_n2v_span=11,
)
```
## Results
From script:
Average PSNR: 28.89
Average MicroSSIM: 0.70