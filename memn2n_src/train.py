from __future__ import absolute_import
import os
import json
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from data_utils import load_candidates

class chatBot(object):
    def __init__(self, data_dir, model_dir, task_id, isInteractive=True, OOV=False, memory_size=50, random_state=None, batch_size=32, learning_rate=0.001, epsilon=1e-8, max_grad_norm=40.0, evaluation_interval=10, hops=3, epochs=200, embedding_size=20):
        self.data_dir = data_dir
        self.task_id = task_id
        self.model_dir = model_dir
        # self.isTrain=isTrain
        self.isInteractive = isInteractive
        self.OOV = OOV
        self.memory_size = memory_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.evaluation_interval = evaluation_interval
        self.hops = hops
        self.epochs = epochs
        self.embedding_size = embedding_size

        candidates, self.candid2indx = load_candidates(self.data_dir, self.task_id)
        print candidates


def main(params):
    model_dir = "task" + str(params['task_id']) + "_" + params['model_dir']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    chatbot = chatBot(data_dir=params['data_dir'], model_dir=model_dir, task_id=params['task_id'], isInteractive=params['interactive'], OOV=params['OOV'], memory_size=params['memory_size'], random_state=params['random_state'], batch_size=params['batch_size'], learning_rate=params['learning_rate'], epsilon=params['epsilon'], max_grad_norm=params['max_grad_norm'], evaluation_interval=params['evaluation_interval'], hops=params['hops'], epochs=params['epochs'], embedding_size=params['embedding_size'])
    # if params['train']:
    #     chatbot.train()
    # else:
    #     chatbot.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate for Adam Optimizer')
    parser.add_argument('--epsilon', default=1e-8, type=float, help='Epsilon value for Adam Optimizer')
    parser.add_argument('--max_grad_norm', default=40.0, type=float, help='Clip gradients to this norm')
    parser.add_argument('--evaluation_interval', default=10, type=int, help='Evaluate and print results every x epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--hops', default=3, type=int, help='Number of hops in the Memory Network')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train for')
    parser.add_argument('--embedding_size', default=20, type=int, help='Embedding size for embedding matrices')
    parser.add_argument('--memory_size', default=50, type=int, help='Maximum size of memory')
    parser.add_argument('--task_id', default=6, type=int, help='bAbI task id, 1 <= id <= 6')
    parser.add_argument('--random_state', default=None, help='Random state')
    parser.add_argument('--data_dir', default='data/dialog-bAbI-tasks/', help='Directory containing bAbI tasks')
    parser.add_argument('--model_dir', default='model/', help='Directory containing memn2n model checkpoints')
    parser.add_argument('--train', default=True, type=bool, help='Train if True, test if False')
    parser.add_argument('--interactive', default=False, type=bool, help='if True, interactive')
    parser.add_argument('--OOV', default=False, type=bool, help='if True, use OOV test set')
    parser.add_argument('--print_params', default=True, type=bool, help='pass False to turn off printing input parameters')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    if params['print_params']:
        print('parsed input parameters:')
        print json.dumps(params, indent = 2)
    main(params)


