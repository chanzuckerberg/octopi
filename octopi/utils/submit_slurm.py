def create_shellsubmit(
    job_name,
    output_file,
    shell_name,
    conda_path,
    command,
    num_gpus = 1,
    gpu_constraint = 'h100',
    time_limit = '18:00:00',
    cpus_per_task = 4,
    mem_per_cpu = '16G',
    partition = None):
    """
    Create a SLURM submission script.

    Args:
        job_name: Name of the SLURM job
        output_file: Path for job output log
        shell_name: Name of the shell script to create
        conda_path: Path to conda environment
        command: Command to execute
        num_gpus: Number of GPUs (0 for CPU-only jobs)
        gpu_constraint: GPU type (h100, a100, a6000, h200)
        time_limit: Job time limit in HH:MM:SS format (default: '18:00:00')
        cpus_per_task: Number of CPUs per task (default: 4)
        mem_per_cpu: Memory per CPU (default: '16G')
        partition: SLURM partition (default: auto-detect based on num_gpus)
    """

    # Determine partition
    if partition is None:
        partition = 'gpu' if num_gpus > 0 else 'cpu'

    # GPU configuration
    if num_gpus > 0:
        slurm_gpus = f'#SBATCH --partition={partition}\n#SBATCH --gpus={gpu_constraint}:{num_gpus}'
    else:
        slurm_gpus = f'#SBATCH --partition={partition}'

    shell_script_content = f"""#!/bin/bash

{slurm_gpus}
#SBATCH --time={time_limit}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --job-name={job_name}
#SBATCH --output={output_file}

ml anaconda
conda activate {conda_path}
{command}
"""

    # Save to file
    with open(shell_name, "w") as file:
        file.write(shell_script_content)

    print(f"\nShell script has been created successfully as {shell_name}\n")

def create_shellsubmit_array(
    job_name,
    output_file,
    shell_name,
    conda_path,
    command,
    job_array,
    time_limit = '18:00:00',
    cpus_per_task = 4,
    mem_per_cpu = '16G',
    partition = 'gpu',
    num_gpus = 1,
    gpu_constraint = 'h100'):
    """
    Create a SLURM array job submission script.

    Args:
        job_name: Name of the SLURM job
        output_file: Path for job output log
        shell_name: Name of the shell script to create
        conda_path: Path to conda environment
        command: Command to execute
        job_array: Tuple/list of (min, max) for array range
        time_limit: Job time limit in HH:MM:SS format (default: '18:00:00')
        cpus_per_task: Number of CPUs per task (default: 4)
        mem_per_cpu: Memory per CPU (default: '16G')
        partition: SLURM partition (default: 'gpu')
        num_gpus: Number of GPUs per array task (default: 1)
        gpu_constraint: GPU type (h100, a100, a6000, h200)
    """

    # GPU configuration
    if num_gpus > 0:
        slurm_gpus = f'#SBATCH --partition={partition}\n#SBATCH --gpus={gpu_constraint}:{num_gpus}'
    else:
        slurm_gpus = f'#SBATCH --partition={partition}'

    shell_script_content = f"""#!/bin/bash

{slurm_gpus}
#SBATCH --time={time_limit}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --job-name={job_name}
#SBATCH --output={output_file}
#SBATCH --array={job_array[0]}-{job_array[1]}

ml anaconda
conda activate {conda_path}
{command}
"""

    # Save to file
    with open(shell_name, "w") as file:
        file.write(shell_script_content)

    print(f"\nShell script has been created successfully as {shell_name}\n")

def create_multiconfig_shellsubmit(
    job_name, 
    output_file,
    shell_name,
    conda_path,
    base_inputs,
    config_inputs,
    command):

    multiconfig = f"""#! /bin/bash

#SBATCH --job-name={job_name}
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --partition=cpu
#SBATCH --output={output_file}

ml anaconda 
{conda_path}

{base_inputs}

{config_inputs}

{command}
"""

    # Save to file
    with open(shell_name, "w") as file:
        file.write(multiconfig)

    print(f"\nShell script has been created successfully as {shell_name}\n")
