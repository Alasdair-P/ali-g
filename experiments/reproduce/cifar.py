import yaml
from scheduling import launch


def create_jobs():
    template = "python main.py "
    wrn_opts = " --depth 40 --width 4 --no_tb "
    dn_opts = " --depth 40 --growth 40 --epochs 300 "

    # with open("reproduce/hparams/svm_1e.yaml", "r") as f:
    with open("reproduce/hparams/apsmalleta.yaml", "r") as f:
        hparams = yaml.safe_load(f)

    jobs = []
    for hparam in hparams:
        command = template + " ".join("--{} {}".format(key, value) for key, value in hparam.items())
        if hparam['model'] == "wrn":
            command += wrn_opts
        elif hparam['model'] == "dn":
            command += dn_opts
        else:
            raise ValueError("Model {} not recognized".format(hparam["model"]))
        jobs.append(command)
    return jobs


if __name__ == "__main__":
    jobs = create_jobs()
    launch(jobs)
