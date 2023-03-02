#This file compares the learning of multiple schedulers : PalScheduler, RandomScheduler, RoundRobinScheduler
from multitask_classifier import *


if __name__ == '__main__':
    #First, we pretrain the model
    args = get_args()
    args.epochs = 12
    args.learning_rate = 0.001
    args.use_gpu = True
    args.save_loss_logs = True
    seed_everything(args.seed)  # fix the seed for reproducibility
    #args.option == "pretrain"
    #train_multitask(args)

    #Now, we fine-tune the model with different schedulers
    schedulers = ['random', 'round_robin', 'pal']
    args.learning_rate = 0.0005
    args.option == "finetune"
    for scheduler in schedulers:
        #We load the pretrained model
        print(f'Running {scheduler} scheduler')
        args.task_scheduler = scheduler    
        args.filepath = f'{args.option}-{args.epochs}-{args.lr}-{args.task_scheduler}-multitask.pt' # save path
        train_multitask(args)
        print(f'Finished {scheduler} scheduler')
