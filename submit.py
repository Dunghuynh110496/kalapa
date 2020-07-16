import os

data_version = "16-07-2020"
code_version = os.popen('git rev-parse HEAD').read().strip()
weight_version = "abcasdasdasd"

os.system(f"python run.py -d {data_version} -c {code_version} -w {weight_version}")

