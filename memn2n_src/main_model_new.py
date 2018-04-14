import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable as Var

dtype = torch.FloatTensor


class MemN2NDialog(nn.Module):
    """End-To-End Memory Network."""

    def __init__(self, batch_size, vocab_size, candidate_size, sentence_size, candidates_vec,
                 embedding_size=20, hops=3, learning_rate=1e-6, max_grad_norm=40.0, task_id=1):
        super(MemN2NDialog, self).__init__()
        # Constants
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._learn_rate = learning_rate
        self._candidate_size = candidate_size
        self.candidates = candidates_vec

        # Weight matrices
        self.fc1 = nn.Linear(self._sentence_size, self._embedding_size, bias=False) #A
        self.fc2 = nn.Linear(self._sentence_size, self._embedding_size, bias=False) #C
        self.fc3 = nn.Linear(self._embedding_size, self._embedding_size, bias=False) #H
        self.fc4 = nn.Linear(self._embedding_size, int(self.candidates.shape[0]), bias=False) #W
        self.softmax = torch.nn.Softmax(dim=0)

        
    def single_pass(self, stories, queries):
        # Initialize predictions
        a_pred = []

        # Iterate over batch_size
        for b in range(self._batch_size):
            # Get Embeddings
            # print('query size: ', queries[b].shape)
            u = self.fc1(queries[b].view(1, self._sentence_size)).view(self._embedding_size)
            # print('Shape of u: ', u.shape)
            m = self.fc1(stories[b].view(1, int(stories[b].data.shape[0]), self._sentence_size)).view(int(stories[b].data.shape[0]), self._embedding_size)
            # print('Shape of m: ', m.shape)
            c = self.fc2(stories[b].view(1, int(stories[b].data.shape[0]), self._sentence_size)).view(int(stories[b].data.shape[0]), self._embedding_size)
            # print('Shape of c: ', c.shape)

            # Pass through Memory Network
            for _ in range(self._hops):
                o = Var(torch.zeros(self._embedding_size).type(dtype), requires_grad=False)

                # Iterate over m_i to get p_i
                for i in range(int(m.data.shape[0])):
                    prob = self.softmax(u.dot(m[i]))  # probability of each possible output
                    o += prob * c[i]                  # generate embedded output

                # Update next input
                u = torch.nn.functional.normalize(o + self.fc3(u.view(1, self._embedding_size)).view(self._embedding_size), dim=0)

            # Get prediction
            a_pred.append(self.softmax(self.fc4(u.view(1, self._embedding_size)).view(int(self.candidates.shape[0]))))

        return a_pred

    def batch_train(self, stories, queries, answers):
        a_pred = self.single_pass(stories, queries)
        loss = -answers[0].dot(torch.log(a_pred[0]))

        for b in range(1, self._batch_size):
            loss += -answers[b].dot(torch.log(a_pred[b]))

        # Backprop and update weights
        loss.backward()
        return float(loss.data)

    def batch_test(self, stories, queries, answers):
        a_pred = self.single_pass(stories, queries)

        loss = -(answers * torch.log(a_pred)).sum()

        return a_pred, loss
