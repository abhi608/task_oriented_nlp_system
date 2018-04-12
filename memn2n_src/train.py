from __future__ import absolute_import
import os
import json
import torch
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from DataLoader import CDATA
from main_model import MemN2NDialog
from data_utils import load_candidates, load_dialog_task, vectorize_candidates

class chatBot(object):
    def __init__(self, data_dir, model_dir, task_id, isInteractive=True, OOV=False, memory_size=50, random_state=None, batch_size=32, learning_rate=0.001, epsilon=1e-8, max_grad_norm=40.0, evaluation_interval=10, hops=3, epochs=200, embedding_size=20):
        self.data_dir = data_dir
        self.task_id = task_id
        self.model_dir = model_dir
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

        self.train_dataset = CDATA(data_dir=self.data_dir, task_id=self.task_id, memory_size=self.memory_size, train=0, batch_size=self.batch_size) #0->train, 1->validate, 2->test
        self.model = MemN2NDialog(self.batch_size, self.train_dataset.getParam('vocab_size'), self.train_dataset.getParam('sentence_size'), self.embedding_size, self.train_dataset.getParam('candidates_vec'),
                                hops=self.hops, learning_rate=self.learning_rate, max_grad_norm=self.max_grad_norm, task_id=self.task_id)
        # criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        trainS, trainQ, trainA = self.train_dataset.getData()
        assert len(trainS) == len(trainQ) and len(trainQ) == len(trainA)
        n_train = len(trainS)
        batches = zip(range(0, n_train - self.batch_size, self.batch_size), range(self.batch_size, n_train, self.batch_size))
        batches = [(start, end) for start, end in batches]

        for epoch in range(self.epochs):
            np.random.shuffle(batches)
            for start, end in batches:
                s = trainS[start:end]
                q = trainQ[start:end]
                a = trainA[start:end]
                # print "S: ", s
                # self.model.batch_train(s, q, a)

    def build_vocab(self, data, candidates):
        vocab = reduce(lambda x, y: x | y, (set(
            list(chain.from_iterable(s)) + q) for s, q, a in data))
        vocab |= reduce(lambda x, y: x | y, (set(candidate)
                                             for candidate in candidates))
        vocab = sorted(vocab)
        self.word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        max_story_size = max(map(len, (s for s, _, _ in data)))
        mean_story_size = int(np.mean([len(s) for s, _, _ in data]))
        self.sentence_size = max(
            map(len, chain.from_iterable(s for s, _, _ in data)))
        self.candidate_sentence_size = max(map(len, candidates))
        query_size = max(map(len, (q for _, q, _ in data)))
        self.memory_size = min(self.memory_size, max_story_size)
        self.vocab_size = len(self.word_idx) + 1  # +1 for nil word
        self.sentence_size = max(
            query_size, self.sentence_size)  # for the position
        # params
        print("vocab size:", self.vocab_size)
        print("Longest sentence length", self.sentence_size)
        print("Longest candidate sentence length",
              self.candidate_sentence_size)
        print("Longest story length", max_story_size)
        print("Average story length", mean_story_size)


def main(params):
    model_dir = "task" + str(params['task_id']) + "_" + params['model_dir']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    chatbot = chatBot(data_dir=params['data_dir'], model_dir=model_dir, task_id=params['task_id'], isInteractive=params['interactive'], OOV=params['OOV'], memory_size=params['memory_size'], random_state=params['random_state'], batch_size=params['batch_size'], learning_rate=params['learning_rate'], epsilon=params['epsilon'], max_grad_norm=params['max_grad_norm'], evaluation_interval=params['evaluation_interval'], hops=params['hops'], epochs=params['epochs'], embedding_size=params['embedding_size'])
    if params['train']:
        chatbot.train()
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


