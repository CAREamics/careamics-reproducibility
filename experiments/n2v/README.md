# N2V

## Summary

Reproducibility between:
- [Original TF](https://github.com/juglab/n2v)
- [PPN2V torch](https://github.com/juglab/ppn2v)
- [CAREamics](https://github.com/CAREamics/careamics-restoration)

The various implementations have certains differences which complicate
reproducing exactly the same training conditions. In particular parameters
can have different definition or default values:

- Model architecture:
    - TF: UNet with batch norm
    - torch (PPN2V): UNet without batch norm layers
    - CAREamics:
- Number of masked pixels:
    - TF: Percentage of pixels in patch (default 0.198%)
    - torch (PPN2VApp): Approximate number from a grid (default 312.5)
    - CAREamics:
- N2V neighborhood:
    - TF:
    - torch (PPN2V):
    - CAREamics:
- Epoch: 
    - TF: An epoch is a number of steps
    - torch (PPN2V): An epoch is a number of steps
    - CAREamics: An epoch is an iteration through all patches


## Set-up

- tf-n2v: Reproduce the [original notebook](https://github.com/juglab/n2v/blob/main/examples/2D/denoising2D_BSD68/BSD68_reproducibility.ipynb).
- torch-ppn2v: Use the default number of pixels but use a number of steps per
epoch to make the number of masked pixels seen during each epoch similar to
tf-n2v.


## Results

Using the scripts in this repository.

|  dataset   |    TF(PSNR)   | torch (PPN2V)(PSNR) | CAREamics(PSNR) |CAREamics (MicroSSIM)|
|------------|---------------|---------------------|-----------------|---------------------|     
| BSD68      | 26.7 +/- 2.53 |  27.34 +/- 3        |                 |                     |
| Convallaria| 26.7 +/- 2.53 |  27.34 +/- 3        |      36.45      |         0.93        |           