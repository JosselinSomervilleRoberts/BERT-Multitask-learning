#This file compares the learning of multiple schedulers : PalScheduler, RandomScheduler, RoundRobinScheduler
import matplotlib.pyplot as plt

def read_logs(path):
    """
    Read the loss logs of a scheduler.
    :param path: path to the log file.
    :return: list of loss values.
    """

    with open(path, 'r') as f:
        logs = f.readlines()
    logs = logs[1:]
    logs = [float(log.split(',')[1]) for log in logs]
    return logs

def compare_schedulers_logs(list_logs):
    """
    Compare the loss logs of multiple schedulers and generates a single graph with the loss of each scheduler.
    :param list_logs: list of loss logs for each scheduler.
    :return: None
    """
    pal_log, random_log, round_robin_log = list_logs
    plt.figure(figsize=(10, 5))
    plt.legend(['PalScheduler', 'RandomScheduler', 'RoundRobinScheduler'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    pal_log_0 = read_logs('train_loss_logs_epochs_pal_0.txt')
    random_log_0 = read_logs('train_loss_logs_epochs_random_0.txt')
    round_robin_log_0 = read_logs('train_loss_logs_epochs_round_robin_0.txt')

    pal_log_1 = read_logs('train_loss_logs_epochs_pal_1.txt')
    random_log_1 = read_logs('train_loss_logs_epochs_random_1.txt')
    round_robin_log_1 = read_logs('train_loss_logs_epochs_round_robin_1.txt')

    print(pal_log_0)
    #compare_schedulers_logs([pal_log, random_log, round_robin_log])