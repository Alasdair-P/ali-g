import os
import time

base_cmd = "python main.py --dataset cifar100 --wrn --depth 16 --width 4 --batch-size 128 --opt {opt} --eta {eta} --max-norm {ma\
x_norm} --l2 {l2} --momentum 0.9 --epochs 160 --dropout 0.4 --no-data-augmentation --no-visdom --no-tqdm"

jobs = [
    base_cmd.format(opt='nrgd', eta=0.001, max_norm=50, l2=0),
    base_cmd.format(opt='nrgd', eta=0.01, max_norm=50, l2=0),
    base_cmd.format(opt='nrgd', eta=0.1, max_norm=50, l2=0),
]


if __name__ == "__main__":
    for job in jobs:
        print(job)
    print("Total of {} jobs to launch".format(len(jobs)))
    launch(jobs, 5)

 
def run_command(command, noprint=True):
    command = " ".join(command.split())
    print(command)
    os.system(command)

def launch(jobs, interval):
    for i, job in enumerate(jobs):
        print("\nJob {} out of {}".format(i + 1, len(jobs)))
        run_command(job)
        time.sleep(interval)
