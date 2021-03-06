from __future__ import division
import torch
import numpy as np
from torch.autograd import Variable as Var

dtype = torch.FloatTensor
if torch.cuda.device_count() > 0:
    dtype = torch.cuda.FloatTensor
    print("Running on GPU")


class MemN2NDialog(object):
    """End-To-End Memory Network."""

    def __init__(self, batch_size, vocab_size, candidate_size, sentence_size, candidates_vec,
                 embedding_size=20, hops=3, learning_rate=1e-6, max_grad_norm=40.0, task_id=1):

        # Constants
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._learn_rate = learning_rate
        self._candidate_size = candidate_size
        self._candidates = candidates_vec

        # Weight matrices
        self.A = Var(torch.randn(self._sentence_size,
                                 self._embedding_size).type(dtype), requires_grad=True)
        self.C = Var(torch.randn(self._sentence_size,
                                 self._embedding_size).type(dtype), requires_grad=True)
        self.H = Var(torch.randn(self._embedding_size,
                                 self._embedding_size).type(dtype), requires_grad=True)
        self.W = Var(torch.randn(self._embedding_size,
                                 self._candidates.shape[0]).type(dtype), requires_grad=True)

        # Functions
        self.softmax = torch.nn.Softmax(dim=0)

    def single_pass(self, stories, queries):
        # Initialize predictions
        # a_pred = Var(torch.randn(self._candidate_size).type(dtype), requires_grad=False)
        a_pred = []

        # Iterate over batch_size
        for b in range(len(queries)):
            # Get Embeddings
            # print('query size: ', queries[b].shape)
            u = queries[b].matmul(self.A)  # query embeddings
            # print('Shape of u: ', u.shape)
            m = stories[b].matmul(self.A)  # memory vectors
            # print('Shape of m: ', m.shape)
            c = stories[b].matmul(self.C)  # possible outputs
            # print('Shape of c: ', c.shape)

            # Pass through Memory Network
            for _ in range(self._hops):
                o = Var(torch.zeros(self._embedding_size).type(dtype), requires_grad=False)

                # Iterate over m_i to get p_i
                for i in range(int(m.data.shape[0])):
                    prob = self.softmax(u.dot(m[i]))  # probability of each possible output
                    o += prob * c[i]                  # generate embedded output

                # Update next input
                u = torch.nn.functional.normalize(o + u.matmul(self.H), dim=0)

            # Get prediction
            a_pred.append(self.softmax(u.matmul(self.W)))

        return a_pred

    def batch_train(self, stories, queries, answers):
        a_pred = self.single_pass(stories, queries)

        loss = -answers[0].dot(torch.log(a_pred[0]))
        for b in range(1, self._batch_size):
            loss += -answers[b].dot(torch.log(a_pred[b]))
        # print("loss: ", loss.data)

        # Backprop and update weights
        loss.backward()
        self.update_weights()

        return float(loss.data)

    def test(self, stories, queries, answers):
        a_pred = self.single_pass(stories, queries)
        assert len(a_pred) == len(answers)

        loss = -answers[0].dot(torch.log(a_pred[0]))
        for b in range(1, len(a_pred)):
            loss += -answers[b].dot(torch.log(a_pred[b]))

        acc = 0
        for b in range(len(a_pred)):
            pred = np.argmax(a_pred[b].data.numpy())
            actual = np.argmax(answers[b].data.numpy())
            if pred == actual:
                acc += 1
        acc /= len(a_pred)

        return acc, loss

    def predict(self, story, query):
        a_pred = self.single_pass([story], [query])

        pred_index = np.argmax(a_pred[0].data.numpy())
        pred_out = self._candidates[pred_index]

        return pred_out

    def update_weights(self):
        # print self.A.grad.data
        self.A.data -= self._learn_rate * self.A.grad.data
        self.H.data -= self._learn_rate * self.H.grad.data
        self.C.data -= self._learn_rate * self.C.grad.data
        self.W.data -= self._learn_rate * self.W.grad.data

        # Manually zero the gradients after updating weights
        self.A.grad.data.zero_()
        self.H.grad.data.zero_()
        self.C.grad.data.zero_()
        self.W.grad.data.zero_()

    def save_weights(self, filename='model_weights.tar'):
        weights = {'W': self.W, 'A': self.A, 'C': self.C, 'H': self.H}
        torch.save(weights, filename)

    def load_weights(self, filename='model_weights.tar'):
        weights = torch.load(filename)
        self.W = weights['W']
        self.A = weights['A']
        self.H = weights['H']
        self.C = weights['C']


class MemN2NDialog_2(MemN2NDialog):
    """End-To-End Memory Network."""

    def __init__(self, batch_size, vocab_size, candidate_size, sentence_size, candidates_vec,
                 embedding_size=20, hops=3, learning_rate=1e-6, max_grad_norm=40.0, task_id=1):

        # Constants
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._learn_rate = learning_rate
        self._candidate_size = candidate_size
        self._candidates = candidates_vec

        # Weight matrices
        self.A = Var(torch.randn(self._sentence_size,
                                 self._embedding_size).type(dtype), requires_grad=True)
        self.C = []
        for _ in range(hops):
            self.C.append(Var(torch.randn(self._sentence_size,
                                          self._embedding_size).type(dtype), requires_grad=True))
        self.W = Var(torch.randn(self._embedding_size,
                                 self._candidates.shape[0]).type(dtype), requires_grad=True)

        # Functions
        self.softmax = torch.nn.Softmax(dim=0)

    def single_pass(self, stories, queries):
        # Initialize predictions
        a_pred = []

        # Iterate over batch_size
        for b in range(len(queries)):
            # Get Embeddings
            u = queries[b].matmul(self.A)     # query embeddings
            m = stories[b].matmul(self.A)     # memory vectors

            # Pass through Memory Network
            for hop in range(self._hops):
                c = stories[b].matmul(self.C[hop])  # possible outputs

                o = Var(torch.zeros(self._embedding_size).type(dtype), requires_grad=False)
                # Iterate over m_i to get p_i
                for i in range(int(m.data.shape[0])):
                    prob = self.softmax(u.dot(m[i]))  # probability of each possible output
                    o += prob * c[i]                  # generate embedded output

                # Update next input
                u = torch.nn.functional.normalize(o + u, dim=0)
                m = c

            # Get prediction
            a_pred.append(self.softmax(u.matmul(self.W)))

        return a_pred

    def update_weights(self):
        self.A.data -= self._learn_rate * self.A.grad.data
        self.A.grad.data.zero_()

        for hop in range(self._hops):
            self.C[hop].data -= self._learn_rate * self.C[hop].grad.data
            self.C[hop].grad.data.zero_()

        self.W.data -= self._learn_rate * self.W.grad.data
        self.W.grad.data.zero_()

    def save_weights(self, filename='model_weights.tar'):
        weights = {'W': self.W, 'C': self.C, 'A': self.A}
        torch.save(weights, filename)

    def load_weights(self, filename='model_weights.tar'):
        weights = torch.load(filename)
        self.W = weights['W']
        self.A = weights['A']
        self.C = weights['C']
