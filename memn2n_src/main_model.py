import torch
from torch.autograd import Variable as Var

dtype = torch.FloatTensor


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
        self.softmax = torch.nn.Softmax(dim=0)

    def single_pass(self, stories, queries):
        # Initialize predictions
        a_pred = Var(torch.randn(self._batch_size).type(dtype), requires_grad=False)

        # Iterate over batch_size
        for b in range(self._batch_size):
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
                # print('\nNumber of sentences: ', m.shape[0])
                for i in range(int(m.data.shape[0])):
                    prob = self.softmax(u.dot(m[i]))  # probability of each possible output
                    # print('Probability shape: ', prob.shape)
                    # print('c shape: ', c[i].shape)
                    o += prob * c[i]                  # generate embedded output

                # Update next input
                u = o + u.matmul(self.H)

            # Get prediction
            a_pred[b] = self.softmax(u.dot(self.W))

        return a_pred

    def batch_train(self, stories, queries, answers):
        a_pred = self.single_pass(stories, queries)
        # print(answers)
        # answers = Var(dtype(answers), requires_grad=False)

        loss = -answers.dot(torch.log(a_pred))

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
