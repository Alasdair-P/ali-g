import os
import time
import yaml

def create_jobs():

    # template = "python main.py "
    template = "python main.py --no_data_augmentation "
    wrn_opts = " --depth 40 --width 4 --no_tb --epochs 200 "
    rn_opts = " --width 1 --no_tb "
    dn_opts = " --depth 40 --growth 40 --no_tb --epochs 300 "
    # gcn_opts = " --depth 3 --width 50 --epochs 200"
    gcn_opts = " "
    # gcn_opts = " --depth 5 --width 300 "

    # with open("reproduce/hparams/code_test.yaml", "r") as f:
    with open("reproduce/hparams/global_cifar.yaml", "r") as f:
        hparams = yaml.safe_load(f)
    jobs = []
    for hparam in hparams:
        command = template + " ".join("--{} {}".format(key, value) for key, value in hparam.items())
        if hparam['model'] == "wrn":
            command += wrn_opts
        elif hparam['model'] == "rn":
            command += rn_opts
        elif hparam['model'] == "dn":
            command += dn_opts
        elif "gcn" in hparam['model']:
            command += gcn_opts
        else:
            raise ValueError("Model {} not recognized".format(hparam["model"]))
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


