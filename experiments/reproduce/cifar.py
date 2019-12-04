import yaml
from scheduling import launch


def create_jobs():
    template = "python main.py --no_tqdm "
    # template = "python main.py --no_tqdm --no_visdom "
    wrn_opts = " --depth 40 --width 4 --port 9007"
    dn_opts = " --depth 40 --growth 40 --port 9007"
    # resnet_opts = " --depth 20 --epochs 200 --momentum 0.9 --teacher /data0/binary-networks-data/cifar100/cifar100_resnet32/model.pkl --batch_size 128 --run teacher32 --loss kl"
    resnet_opts = " --depth 20 --epochs 200 --port 9025"

    # with open("reproduce/hparams/cifar.yaml", "r") as f:
    # with open("reproduce/hparams/cross_val_cgd_0_1_large_batch.yaml", "r") as f:
    # with open("reproduce/hparams/distill.yaml", "r") as f:
    # with open("reproduce/hparams/cifar_segd.yaml", "r") as f:
    with open("reproduce/hparams/cifar_segd_1.yaml", "r") as f:
        hparams = yaml.safe_load(f)

    jobs = []
    for hparam in hparams:
        command = template + " ".join("--{} {}".format(key, value) for key, value in hparam.items())
        if hparam['model'] == "wrn":
            command += wrn_opts
        elif hparam['model'] == "dn":
            command += dn_opts
        elif hparam['model'] == "resnet":
            command += resnet_opts
        else:
            raise ValueError("Model {} not recognized".format(hparam["model"]))
        jobs.append(command)
    return jobs


if __name__ == "__main__":
    jobs = create_jobs()
    launch(jobs)
