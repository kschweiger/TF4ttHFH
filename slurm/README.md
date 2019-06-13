# Running tensorflow on slurm (relevant for usage on T3@PSI)

## Commands
| command | usage                              | options                                  |
| ------- | ---------------------------------- | ---------------------------------------- |
| [sbatch](https://slurm.schedmd.com/sbatch.html#index)  | submit a script to the slurm batch | -p [wn/gpu] --> can also be set in script |
| squeue  | Check the queues                   |                                          |
| scancel | kill jobs                          | -u [USERNAME]                            |
|         |                                    |  -n [JOBNAME]                            |

## Scripts
Configuration about the submission a listed in the top of the file **with a leading #SBATCH** like this:

```
#SBATCH -p wn
#SBATCH --time 01:00:00
#SBATCH -w t3wn60
#SBATCH -e cn-test.err 
#SBATCH -o cn-test.out  # replace default slurm-SLURM_JOB_ID.out
```

Important options:

| option             | effect                    | note                                                     |
| ------------------ | ------------------------- | -------------------------------------------------------- |
| -p [node]          | sets the node             | Use wn for CPU and gpu for GPU node                      |
| -o [filename]      | Sets std out file         | Use `%A` to insert job id - Folders will not be created! |
| -e [filename]      | Sets std err file         | Use `%A` to insert job id - Folders will not be created! |
| --ntasks=XX        | Number of CPUs requested  | On GPU nodes: balance between CPU and GPU : 5CPU/1GPU    |
| --mem=XXXM         | sets node memory          | Set memory **per node**                                  |
| --time=XX-XX.XX    | Time limit                | Set in DD-HH:MM format                                   |
| --account=gpu_gres | access gpu resources      | Only required when running on PGU nodes                  |
| --gres=gpu:[1,2]   | Number of GPUs requiested | Only required when running on PGU nodes                  |

### General notes

- Each worker has a local scratch space at `/scratch/$USER/${SLURM_JOB_ID}` that can be used druing operation. Output files still have to be transferred to the `/work/$USER` (or the storage element) at the end of the job.

- When running a job the setup virtual environment has to be set. 
  - With anaconda installation do: `source activate [VENVNAME]`
  - With regular python (via pyenv) `source /t3home/$HOME/.pyenv/versions/[VENVNAME]/bin/activate` 

### Notes for running on GPU nodes
Allocated GPU numbers are saved in the `CUDA_VISIBLE_DEVICES` environment variable. Since this is dynamically allocated the gpu device number should be set according to this variable when running a script on the node.

## Test

In order to check if all permissions are set and see how a submission scrpit for GPUs need to look check `scripts/testscript_gpu.sh`. Submit with 

```
mkdir output #If not present
sbatch scripts/testscript_gpu.sh
```

