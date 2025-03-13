import subprocess
import time

def run_command_in_conda_env(env_name, command):
    cmd = f'source $(conda info --base)/etc/profile.d/conda.sh && conda activate {env_name} && {command}'
    start = time.time()
    print(cmd)
    subprocess.run(
        ['bash', '-c', cmd],
        check=True,
        text=True
    )
    return time.time()-start
