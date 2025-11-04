import os
from time import sleep, time
import subprocess
import sys
import numpy as np
import re
'''
 This script is used to create a batch of parameter files for varying sigma_e.
'''

N = 300000

sigma_e_list = np.arange(0.01, 0.301, 0.01)

if len(sys.argv) < 2:
    print("Usage: python srun_sigma.py <base_param_name>")
    sys.exit(1)

parname_base = sys.argv[1]

base_param_file = os.path.join('sim_parameter_files', parname_base + '.m')
try:
    with open(base_param_file, 'r') as f:
        param_content_template = f.read()
except FileNotFoundError:
    print(f"Error: Base parameter file not found at {base_param_file}")
    sys.exit(1)
    
# Create a directory for batch parameter files if it doesn't exist
batch_param_dir = os.path.join('sim_parameter_files', 'batch')
os.makedirs(batch_param_dir, exist_ok=True)


for sigma_e in sigma_e_list:
    new_parname_nosubdir = f'{parname_base}_sE_{sigma_e:.2f}'.replace('.', 'p')
    new_param_file_path = os.path.join(batch_param_dir, new_parname_nosubdir + '.m')

    # Modify parameter file content
    # sigma_ee, sigma_pe, sigma_se, sigma_ve are set to sigma_e
    new_content = re.sub(r'^(sigmaee\s*=\s*).*;', f'\\g<1>{sigma_e:.10f};', param_content_template, flags=re.MULTILINE)
    new_content = re.sub(r'^(sigmape\s*=\s*).*;', f'\\g<1>{sigma_e:.10f};', new_content, flags=re.MULTILINE)
    new_content = re.sub(r'^(sigmase\s*=\s*).*;', f'\\g<1>{sigma_e:.10f};', new_content, flags=re.MULTILINE)
    new_content = re.sub(r'^(sigmave\s*=\s*).*;', f'\\g<1>{sigma_e:.10f};', new_content, flags=re.MULTILINE)
    
    with open(new_param_file_path, 'w') as f:
        f.write(new_content)