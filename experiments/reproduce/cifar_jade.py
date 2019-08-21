import os
import time

# jobs = ["python main.py --batch_size 256 --dataset imagenet --eta 0.001 --model ResNet18 --momentum 0.9 --opt sgd --pochs 50 --no_tqdm --no_visdom --T 1 2 3 4 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 --decay_factor 1.14816 --xp_name start_point"]

jobs = ["python main.py --batch_size 256 --dataset tiny_imagenet --eta 0.1 --model ResNet18 --momentum 0.9 --opt sgd --epochs 60 --no_tqdm --no_visdom --T 20 40 --decay_factor 0.1 --xp_name fp_tiny"]

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


