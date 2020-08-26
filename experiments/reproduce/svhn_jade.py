import os
import time
import yaml

def create_jobs():
    template = """python main.py --dataset svhn-extra --model wrn --depth 16
        --width 4 --batch_size 128 --momentum 0 --epochs 160
        --dropout 0.4 --no_data_augmentation --no_visdom --no_tqdm --no_tb --jade --tag 0 """

    with open("reproduce/hparams/svhn_borat.yaml", "r") as f:
        hparams = yaml.safe_load(f)

    jobs = []
    for hparam in hparams:
        command = template + " ".join("--{} {}".format(key, value) for key, value in hparam.items())
        jobs.append(command)
    return jobs

def run_command(command, noprint=True):
    command = " ".join(command.split())
    print(command)
    os.system(command)

def launch(jobs, interval):
    for i, job in enumerate(jobs):
        print("\nJob {} out of {}".format(i + 1, len(jobs)))
        run_command(job)
        time.sleep(interval)

if __name__ == "__main__":
    jobs = create_jobs()
    for job in jobs:
        print(job)
    print("Total of {} jobs to launch".format(len(jobs)))
    launch(jobs, 5)
