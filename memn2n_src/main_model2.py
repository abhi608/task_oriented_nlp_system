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
        nil_word_slot = torch.zeros([1, self._embedding_size])
        self.q_embed = torch.nn.Embedding(self._vocab_size, self._embedding_size)
        self.q_embed.weight.data[0] = nil_word_slot
        # self.c_embed = torch.nn.Embedding.from_pretrained(A)
        self.H = Var(torch.randn(self._embedding_size,
                                 self._embedding_size).type(dtype), requires_grad=True)
        self.out_embed = torch.nn.Embedding(self._vocab_size, self._embedding_size)
        self.out_embed.weight.data[0] = nil_word_slot

        # Functions
        self.softmax = torch.nn.Softmax(dim=0)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def single_pass(self, stories, queries):
        # Get Embeddings
        q_emb = self.q_embed(queries)  # Get vocab embedding
        u = torch.sum(q_emb, 1)      # Reduce sentence size

        # Pass through Memory Network
        for _ in range(self._hops):
            m_emb = self.q_embed(stories)  # memory vectors
            m = torch.sum(m_emb, 2)        # Reduce sentence size

            # Calculate probabilities
            p = self.softmax(m.bmm(u.view(self._batch_size, self._embedding_size, 1)))

            # # Uncomment when calculating separate answer embedding
            # c_emb = self.c_embed(stories)  # possible outputs
            # c = c_emb.sum(2)

            # Calculate possible output
            p_temp = p.view(self._batch_size, 1, self._embedding_size)
            o = p_temp.bmm(m).view(self._batch_size, self._embedding_size)

            # Update next input
            u = self.u.matmul(self.H) + o

        # Get prediction
        candidates_emb = self.out_embed(self._candidates)
        candidates_emb = candidates_emb.sum(1)  # reducing sentence size
        a_pred = u.matmul(candidates_emb.transpose())

        return a_pred

    def batch_train(self, stories, queries, answers):
        a_pred = self.single_pass(stories, queries)

        loss = self.loss_fn(a_pred, answers)
        loss = loss.sum()

        # Backprop and update weights
        loss.backward()
        self.update_weights()

        return float(loss.data)

    def test(self, stories, queries, answers):
        a_pred = self.single_pass(stories, queries)
        assert len(a_pred) == len(answers)

        loss = self.loss_fn(a_pred, answers)
        loss = loss.sum()

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
        self.q_embed.weight.data -= self._learn_rate * self.q_embed.weight.grad.data
        self.H.data -= self._learn_rate * self.H.grad.data
        # self.c_embed.weight.data -= self._learn_rate * self.c_embed.weight.grad.data
        self.out_embed.weight.data -= self._learn_rate * self.out_embed.weight.grad.data

        # Manually zero the gradients after updating weights
        self.q_embed.weight.grad.data.zero_()
        self.H.grad.data.zero_()
        # self.c_embed.weight.grad.data.zero_()
        self.out_embed.weight.grad.data.zero_()

    def save_weights(self, filename='model_weights.tar'):
        weights = {'q_embed': self.q_embed, 'H': self.H, 'out_embed': self.out_embed}
        torch.save(weights, filename)

    def load_weights(self, filename='model_weights.tar'):
        weights = torch.load(filename)
        self.q_embed = weights['q_embed']
        self.H = weights['H']
        # self.c_embed = weights['c_embed']
        self.out_embed = weights['out_embed']


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
        nil_word_slot = torch.zeros([1, self._embedding_size])
        self.q_embed = torch.nn.Embedding(self._vocab_size, self._embedding_size)
        self.q_embed.weight.data[0] = nil_word_slot
        self.c_embed = []
        for _ in range(hops):
            self.C.append(torch.nn.Embedding(self._vocab_size, self._embedding_size))
            self.C[-1].weight.data[0] = nil_word_slot
        self.out_embed = torch.nn.Embedding(self._vocab_size, self._embedding_size)
        self.out_embed.weight.data[0] = nil_word_slot

        # Functions
        self.softmax = torch.nn.Softmax(dim=0)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def single_pass(self, stories, queries):
        # Get Embeddings
        u = self.q_embed(queries)  # Get vocab embedding
        u = torch.sum(u, 1)        # Reduce sentence size
        m = self.q_embed(stories)  # Get memory vectors
        m = torch.sum(m, 2)        # Reduce sentence size

        # Pass through Memory Network
        for hop in range(self._hops):
            c = self.C[hop](stories)  # possible outputs
            c = torch.sum(c, 2)

            # Calculate probabilities
            p = self.softmax(m.bmm(u.view(self._batch_size, self._embedding_size, 1)))

            # Calculate possible output
            p_temp = p.view(self._batch_size, 1, self._embedding_size)
            o = p_temp.bmm(c).view(self._batch_size, self._embedding_size)

            # Update next input
            u = u + o
            m = c

        # Get prediction
        candidates_emb = self.out_embed(self._candidates)
        candidates_emb = candidates_emb.sum(1)  # reducing sentence size
        a_pred = u.matmul(candidates_emb.transpose())

        return a_pred

    def update_weights(self):
        self.q_embed.weight.data -= self._learn_rate * self.q_embed.weight.grad.data
        self.q_embed.weight.grad.data.zero_()

        for hop in range(self._hops):
            self.C[hop].weight.data -= self._learn_rate * self.C[hop].weight.grad.data
            self.C[hop].weight.grad.data.zero_()

        self.out_embed.weight.data -= self._learn_rate * self.out_embed.weight.grad.data
        self.out_embed.weight.grad.data.zero_()

    def save_weights(self, filename='model_weights.tar'):
        weights = {'q_embed': self.q_embed, 'C': self.C, 'out_embed': self.out_embed}
        torch.save(weights, filename)

    def load_weights(self, filename='model_weights.tar'):
        weights = torch.load(filename)
        self.q_embed = weights['q_embed']
        self.C = weights['C']
        self.out_embed = weights['out_embed']
