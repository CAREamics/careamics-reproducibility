# CAREamics reproducibility repository

Start an experiment on the HPC by running the following command:

```bash
./run-experiment -l <library> -m <method> -d <dataset>
```

The script will run a job on the HPC and save the following files:
- Log of the job in `logs/`
- Efficiency of the job in `logs/efficiency`
- Performances of the model in its own folder