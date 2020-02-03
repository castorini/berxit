import os
import sys
import subprocess

slurm_command = "srun --mem=12G -c 2 --gres=gpu:1 -p gpu {} &> {} &"

target_command = ""
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        if line.endswith('\\'):
            line = line[:-1].strip()
        target_command += line + ' '


log_id = sys.argv[2]
print(log_id)

log_folder = "logs"
output = log_folder + "/{}.slurm_out".format(log_id)

final_command = slurm_command.format(target_command, output)
print(final_command)

with open(output, 'a') as fout:
    subprocess.run(final_command, shell=True, stdout=fout, stderr=fout)