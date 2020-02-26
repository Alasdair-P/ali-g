import os
import time




# jobs = ["python main.py --batch_size 128 --dataset cifar100 --eta 1.0 --max_norm 175 --model wrn --momentum 0.9 --opt sbdf2 --epochs 400 --depth 40 --width 4 --port 9026 --run n-equals-2-solve-forward-no-momentum --k 2 --solve_forward --no_visdom --no_tqdm --xp_dir /jmain01/home/JAD035/pkm01/shared/models/alasdair --tb /jmain01/home/JAD035/pkm01/shared/tensorboard/aap21", ]
# jobs = ["python main.py --batch_size 128 --dataset cifar100 --eta 1.0 --max_norm 175 --model wrn --momentum 0.9 --opt sbdf2 --epochs 400 --depth 40 --width 4 --port 9026 --run n-equals-2-solve-forward-no-momentum --k 2 --solve_forward --no_visdom --no_tqdm --xp_dir /jmain01/home/JAD035/pkm01/shared/models/alasdair" ]
jobs = ["python main.py --batch_size 128 --dataset cifar100 --eta 1.0 --max_norm 175 --model wrn --momentum 0.9 --opt sbdf2 --epochs 400 --depth 40 --width 4 --port 9026 --run n-equals-2-solve-forward-no-momentum --k 2 --solve_forward --no_visdom --no_tqdm --xp_dir /jmain01/home/JAD035/pkm01/shared/models/alasdair --debug"]



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
    for job in jobs:
        print(job)
    print("Total of {} jobs to launch".format(len(jobs)))
    launch(jobs, 5)


