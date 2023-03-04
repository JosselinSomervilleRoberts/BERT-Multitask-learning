#This file compares the learning of multiple schedulers : PalScheduler, RandomScheduler, RoundRobinScheduler
#import matplotlib.pyplot as plt

def read_train_loss_logs(path):
    """
    Read the loss logs of a scheduler.
    :param path: path to the log file.
    :return: dictionnay of loss values per task.
    """
    # We read the file and remove the first line
    with open(path, 'r') as f:
        logs = f.readlines()
    logs = logs[1:]
    # We extract the necessary data from each line
    data = {}
    for line in logs:
        if line.startswith("sst:"):
            line = line[6:-2]
            data["sst"] = [float(x) for x in line.split(",")]
        elif line.startswith("para:"):
            line = line[7:-2]
            data["para"] = [float(x) for x in line.split(",")]
        elif line.startswith("sts:"):
            line = line[6:-2]
            data["sts"] = [float(x) for x in line.split(",")]

    return data

def read_dev_acc_logs(path):
    """
    Read the dev accuracy logs of a scheduler.
    :param path: path to the log file.
    :return: dictionnay of dev accuracy values per task.
    """
    # We read the file and remove the first line
    with open(path, 'r') as f:
        logs = f.readlines()
    logs = logs[1:]
    # We extract the necessary data from each line
    data = {}
    for line in logs:
        if line.startswith("sst:"):
            line = line[6:-2]
            data["sst"] = [float(x) for x in line.split(",")]
        elif line.startswith("para:"):
            line = line[7:-2]
            data["para"] = [float(x) for x in line.split(",")]
        elif line.startswith("sts:"):
            line = line[6:-2]
            data["sts"] = [float(x) for x in line.split(",")]
    return data

# def compare_schedulers_logs(list_logs):
#     """
#     Compare the loss logs of multiple schedulers and generates a single graph with the loss of each scheduler.
#     :param list_logs: list of loss logs for each scheduler.
#     :return: None
#     """
#     pal_log, random_log, round_robin_log = list_logs
#     plt.figure(figsize=(10, 5))
#     plt.legend(['PalScheduler', 'RandomScheduler', 'RoundRobinScheduler'])
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.show()


if __name__ == '__main__':
    pal_loss_log_0 = read_train_loss_logs('train_loss_logs_epochs_pal_0.txt')
    random_loss_log_0 = read_train_loss_logs('train_loss_logs_epochs_random_0.txt')
    round_robin_loss_log_0 = read_train_loss_logs('train_loss_logs_epochs_round_robin_0.txt')

    pal_log_1 = read_train_loss_logs('train_loss_logs_epochs_pal_1.txt')
    random_log_1 = read_train_loss_logs('train_loss_logs_epochs_random_1.txt')
    round_robin_log_1 = read_train_loss_logs('train_loss_logs_epochs_round_robin_1.txt')

    print(pal_loss_log_0)

    #compare_schedulers_logs([pal_log, random_log, round_robin_log])