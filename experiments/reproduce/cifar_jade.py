import os
import time


# jobs = ["python main.py --dataset tiny_imagenet --batch_size 128 --opt sgd --momentum 0.9 --epochs 200 --weight_decay 0.0001 --decay_factor 0.1 --eta 0.1 --xp_name tinyimagenet_resnet110 --loss ce --T 100 150 --model ResNet110 --reg --no_visdom"]
# jobs = ["python main.py --dataset tiny_imagenet --batch_size 128 --opt sgd --momentum 0.9 --epochs 200 --weight_decay 0.0001 --decay_factor 0.1 --eta 0.1 --xp_name tinyimagenet_resnet56 --loss ce --T 100 150 --model ResNet56 --reg --no_visdom"]
# jobs = ["python main.py --dataset tiny_imagenet --batch_size 128 --opt sgd --momentum 0.9 --epochs 200 --weight_decay 0.0001 --decay_factor 0.1 --eta 0.1 --xp_name tinyimagenet_resNet20 --loss ce --T 100 150 --model ResNet20 --reg"]
# jobs = ["python main.py --dataset tiny_imagenet --opt sgd --epochs 50 --batch-size 128 --eta 0.001 --T 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 --momentum 0.9 --xp-name tinyimagenet_resnet20_start_point --loss ce --model ResNet20 --load-model /jmain01/home/JAD035/pkm01/aap21-pkm01/code/ali-g/experiments/tinyimagenet_resnet20/best_model.pkl --decay_factor 1.14816"]
jobs = ["python main.py --dataset tiny_imagenet --opt adam --epochs 300 --batch-size 128 --eta 0.02 --hq_epoch 200 --temp_rate 2e-6 --xp-name distill_base_tau_3 --loss ce --model ResNet20 --tau 3 --load-model /jmain01/home/JAD035/pkm01/aap21-pkm01/code/ali-g/experiments/tinyimagenet_resnet20/model.pkl --teacher /jmain01/home/JAD035/pkm01/aap21-pkm01/code/ali-g/experiments/tinyimagenet_resnet20/model.pkl"]
# jobs = ["python main.py --dataset tiny_imagenet --opt adam --epochs 300 --batch-size 128 --eta 0.02 --hq_epoch 200--temp_rate 2e-6 --xp-name distill_base_tau_1 --loss ce --model ResNet20 --tau 1 --load-model /jmain01/home/JAD035/pkm01/aap21-pkm01/code/ali-g/experiments/tinyimagenet_resnet20/best_model.pkl --teacher /jmain01/home/JAD035/pkm01/aap21-pkm01/code/ali-g/experiments/tinyimagenet_resnet20/best_model.pkl"]

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


