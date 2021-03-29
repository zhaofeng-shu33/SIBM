import os
from datetime import datetime
import pickle
import logging

from sklearn import metrics

def set_up_log():
    LOGGING_FILE = 'simulation.log'
    logFormatter = logging.Formatter('%(asctime)s %(message)s')
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(os.path.join('build', LOGGING_FILE))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    if os.environ.get("LOGLEVEL"):
        rootLogger.setLevel(os.environ.get("LOGLEVEL"))
    else:
        rootLogger.setLevel(logging.INFO)

def get_ground_truth(graph):
    label_list = []
    for n in graph.nodes(data=True):
        label_list.append(n[1]['block'])
    return label_list

def compare(label_0, label_1):
    '''
    get acc using adjusted rand index
    '''
    return metrics.adjusted_rand_score(label_0, label_1)

def save_data_to_pickle(file_name_prefix, data):
    # save the data in pickle format
    file_name = file_name_prefix + '-' + datetime.now().strftime('%Y-%m-%d') + '.pickle'
    with open(os.path.join('build', file_name), 'wb') as f:
        pickle.dump(data, f)