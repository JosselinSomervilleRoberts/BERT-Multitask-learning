#This file compares the learning of multiple schedulers : PalScheduler, RandomScheduler, RoundRobinScheduler
from multitask_classifier import *


if __name__ == '__main__':
    args = get_args()
    args.epochs = 12
    args.learning_rate = 0.001

    schedulers = ['random', 'round_robin', 'pal']

    for scheduler in schedulers:
        print(f'Running {scheduler} scheduler')
        args.task_scheduler = scheduler
        args.save_loss_logs = True
        seed_everything(args.seed)  # fix the seed for reproducibility
        train_multitask(args)
        print(f'Finished {scheduler} scheduler')
