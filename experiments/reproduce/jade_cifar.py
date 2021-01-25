import os
import time
import yaml


def create_jobs():
    template = "python main.py --no_visdom --no_tqdm --jade "
    wrn_opts = " --depth 40 --width 4 --epochs 200"
    # wsrn_opts = " --depth 40 --width 4 "
    dn_opts = " --depth 40 --growth 40 --epochs 300"
    gcn_opts = " --depth 5 --growth 300 --epochs 500"

    with open("reproduce/hparams/test.yaml", "r") as f:
        hparams = yaml.safe_load(f)

    jobs = []
    for hparam in hparams:
        command = template + " ".join("--{} {}".format(key, value) for key, value in hparam.items())
        if hparam['model'] == "wrn":
            command += wrn_opts
        elif hparam['model'] == "dn":
            command += dn_opts
        elif hparam['model'] == "gcn":
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


