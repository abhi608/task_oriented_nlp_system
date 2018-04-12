import torch
from torch.autograd import Variable as Var

dtype = torch.FloatTensor


class MemN2NDialog(object):
    """End-To-End Memory Network."""

    def __init__(self, batch_size, vocab_size, sentence_size, embedding_size,
                 candidates_vec, hops=3, learning_rate=1e-6, max_grad_norm=40.0, task_id=1):

        # Constants
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._learn_rate = learning_rate

        # Weight matrices
        self.A = Var(torch.randn(self._sentence_size,
                                 self._embedding_size).type(dtype), requires_grad=True)
        self.C = Var(torch.randn(self._sentence_size,
                                 self._embedding_size).type(dtype), requires_grad=True)
        self.H = Var(torch.randn(self._embedding_size,
                                 self._embedding_size).type(dtype), requires_grad=True)
        self.W = Var(torch.randn(self._embedding_size).type(
            dtype), requires_grad=True)

        # Functions
        self.softmax = torch.nn.Softmax()

    def single_pass(self, stories, queries):
        # Get Embeddings
        u = queries.matmul(self.A)  # query embeddings
        m = stories.matmul(self.A)  # memory vectors
        c = stories.matmul(self.C)  # possible outputs

        # Initialize output that will be generated
        o = Var(torch.randn(self._batch_size,
                            self._sentence_size).type(dtype), requires_grad=False)

        # Iterate over batch_size, through Memory Network
        for i in self._batch_size:
            story = m[i]
            u_temp = u[i]
            exp_out = c[i]

            for _ in range(self._hops):
                net_out = torch.zeros(self._sentence_size)

                # Iterate over x_i to get p_i
                for j in story.shape[0]:
                    sentence = story[j]
                    probs = self.softmax(u_temp.dot(sentence))
                    net_out += probs * exp_out[j]

                u_temp = net_out + u_temp
                o.data[i] = net_out

        # Get prediction
        a_pred = Var(torch.randn(self._batch_size).type(dtype), requires_grad=False)
        for i in self._batch_size:
            a_pred.data[i] = self.softmax(self.W.dot(u[i] + o.data[i]))

        return a_pred

    def batch_train(self, stories, queries, answers):
        a_pred = self.single_pass(stories, queries)

        loss = -(answers * torch.log(a_pred)).sum()

        # Backprop and update weights
        loss.backward()
        self.update_weights()

    def batch_test(self, stories, queries, answers):
        a_pred = self.single_pass(stories, queries)

        loss = -(answers * torch.log(a_pred)).sum()

        return a_pred, loss

    def update_weights(self):
        self.A.data -= self._learn_rate * self.A.grad.data
        self.H.data -= self._learn_rate * self.H.grad.data
        self.C.data -= self._learn_rate * self.C.grad.data
        self.W.data -= self._learn_rate * self.W.grad.data

        # Manually zero the gradients after updating weights
        self.A.grad.data.zero_()
        self.H.grad.data.zero_()
        self.C.grad.data.zero_()
        self.W.grad.data.zero_()
