import torch as T
import torch.nn as nn
import torch.autograd as autograd
import numpy as np



class LSTM(nn.Module):

    def __init__(self, obs_size, nb_hidden=128, action_size=16):
        super(LSTM, self).__init__()

        self.obs_size = obs_size
        self.nb_hidden = nb_hidden
        self.action_size = action_size

        # entry points
        self.init_state_c_, self.init_state_h_ = self.init_hidden()

        # input projection
        # add relu/tanh here if necessary
        self.projected_features_layer = nn.Linear(obs_size,nb_hidden) 

        self.lstm_f = nn.LSTM(nb_hidden,nb_hidden)

        self.hidden2logits = nn.Linear(2*nb_hidden,action_size)

        self.logitSoftmax = nn.Softmax(dim=2)


    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(T.zeros( 1,1, self.nb_hidden)),
                autograd.Variable(T.zeros(1,1, self.nb_hidden)))


    def forward(self, features, action_mask_):

        projected_features = self.projected_features_layer(features)
        output, state = self.lstm_f(projected_features.view(1,1,-1), (self.init_state_c_, self.init_state_h_))

        self.init_state_c_ = state[0]
        self.init_state_h_ = state[1]

        # reshape LSTM's state tuple (2,128) -> (1,256)
        state_reshaped = T.cat((state[0], state[1]),dim=2)
        # output projection
        # get logits
        logits = self.hidden2logits(state_reshaped)
        # print(logits.size())

        softmax_logits = self.logitSoftmax(logits)
        # probabilities
        #  normalization : elemwise multiply with action mask
        probs = T.mul(T.squeeze(softmax_logits), action_mask_)

        # prediction
        _, prediction = T.max(probs, dim=0)

        return logits,probs,prediction


class LSTM_wrapper():

    def __init__(self, obs_size, nb_hidden=128, action_size=16):

        self.model = LSTM(obs_size,nb_hidden,action_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = T.optim.Adagrad(self.model.parameters(),lr=0.01)

    # training
    def train_step(self, features, action, action_mask):
        self.model.zero_grad()
        self.model.init_state_c_, self.model.init_state_h_ = self.model.init_hidden()
        logits,probs,prediction = self.model(features,action_mask)
        # print(logits)
        # print(action)
        loss = self.loss_function(logits.view(1,-1),action)
        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def reset_state(self):
        self.model.init_state_c_, self.model.init_state_h_ = self.model.init_hidden()

    def forward(self, features, action_mask_):
        return self.model.forward(features,action_mask_)



    # save session to checkpoint
    # def save(self):
        # saver = tf.train.Saver()
        # saver.save(self.sess, 'ckpt/hcn.ckpt', global_step=0)
        # print('\n:: saved to ckpt/hcn.ckpt \n')

    # restore session from checkpoint
    # def restore(self):
        # saver = tf.train.Saver()
        # ckpt = tf.train.get_checkpoint_state('ckpt/')
        # if ckpt and ckpt.model_checkpoint_path:
            # print('\n:: restoring checkpoint from', ckpt.model_checkpoint_path, '\n')
            # saver.restore(self.sess, ckpt.model_checkpoint_path)
        # else:
            # print('\n:: <ERR> checkpoint not found! \n')
