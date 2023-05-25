# N2V

## Summary

Reproducibility between:
- [Original TF](https://github.com/juglab/n2v)
- [PPN2V torch](https://github.com/juglab/ppn2v)
- [CAREamics](https://github.com/CAREamics/careamics-restoration)

The various implementations have certains differences which complicate
reproducing exactly the same training conditions. In particular parameters
can have different definition or default values:

- Number of masked pixels:
    - TF:
    - torch (PPN2V):
    - CAREamics:
- N2V neighborhood
- Epoch: 
    - TF:
    - torch (PPN2V):
    - CAREamics:


## Results

Using the scripts in this repository.

|       |       TF      | torch (PPN2V) | CAREamics |
|-------|---------------|---------------|-----------|
| BSD68 | 26.7 +/- 2.53 |               |           |