import yaml
from scheduling import launch


def create_jobs():
    template = "python main.py "
    wrn_opts = " --depth 40 --width 4 --no_tb --epochs 200 "
    dn_opts = " --depth 40 --growth 40 --no_tb --epochs 300 "
    gcn_opts = " --depth 5 --width 300 --epochs 500 --tag 500e"
    with open("reproduce/hparams/mol.yaml", "r") as f:
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


if __name__ == "__main__":
    jobs = create_jobs()
    launch(jobs)
