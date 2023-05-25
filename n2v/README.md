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
    - torch (PPN2V): Grid ??
    - CAREamics:
- N2V neighborhood:
    - TF:
    - torch (PPN2V):
    - CAREamics:
- Epoch: 
    - TF: An epoch is a number of steps
    - torch (PPN2V): An epoch is a number of steps
    - CAREamics: An epoch is an iteration through all patches


## Results

Using the scripts in this repository.

|       |       TF      | torch (PPN2V) | CAREamics |
|-------|---------------|---------------|-----------|
| BSD68 | 26.7 +/- 2.53 |               |           |