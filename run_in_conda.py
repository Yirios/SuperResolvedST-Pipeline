import subprocess

def run_command_in_conda_env(env_name, command, log_file):
    cmd = f'source $(conda info --base)/etc/profile.d/conda.sh && conda activate {env_name} && {command}'
    with open(log_file, 'w') as file:
        subprocess.run(f'bash -c "{cmd}"', shell=True, text=True, stdout=file, stderr=subprocess.STDOUT)
