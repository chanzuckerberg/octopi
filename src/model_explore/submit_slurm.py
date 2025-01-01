def create_shellsubmit(
    job_name, 
    output_file,
    shell_name,
    conda_path,
    command,
    num_gpus = 1, 
    gpu_constraint = 'H100'):

    if num_gpus > 0:
        slurm_gpus = f'#SBATCH --partition=gpu\n#SBATCH --gpus={num_gpus}\n#SBATCH --constraint="{gpu_constraint}"'
    else:
        slurm_gpus = f'#SBATCH --partition=cpu'

    shell_script_content = f"""#!/bin/bash

{slurm_gpus}
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=16G
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
#SBATCH --cpus-per-task=64
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
