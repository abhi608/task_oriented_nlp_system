from modules.entities import EntityTracker
from modules.bow import BoW_encoder
from  modules.lstm_net import LSTM_wrapper
from modules.embed import UtteranceEmbed
from modules.actions import ActionTracker
from modules.data_utils import Data
import modules.util as util

import numpy as np
import torch
import sys
import random


class Trainer():

    def __init__(self):

        et = EntityTracker()
        self.bow_enc = BoW_encoder()
        self.emb = UtteranceEmbed()
        at = ActionTracker(et)

        self.dataset, dialog_indices = Data(et, at).trainset
        random.shuffle(dialog_indices)
        self.dialog_indices_tr = dialog_indices[:200]
        self.dialog_indices_dev = dialog_indices[200:250]

        obs_size = self.emb.dim + self.bow_enc.vocab_size + et.num_features
        self.action_templates = at.get_action_templates()
        action_size = at.action_size
        nb_hidden = 128

        self.net = LSTM_wrapper(obs_size=obs_size,
                       action_size=action_size,
                       nb_hidden=nb_hidden)


    def train(self):

        print('\n:: training started\n')
        epochs = 80
        for j in range(epochs):
            # iterate through dialogs
            num_tr_examples = len(self.dialog_indices_tr)
            loss = 0.
            for i,dialog_idx in enumerate(self.dialog_indices_tr):
                # get start and end index
                start, end = dialog_idx['start'], dialog_idx['end']
                # train on dialogue
                loss += self.dialog_train(self.dataset[start:end])
                # print #iteration
                sys.stdout.write('\r{}.[{}/{}]'.format(j+1, i+1, num_tr_examples))

            print('\n\n:: {}.tr loss {}'.format(j+1, loss/num_tr_examples))
            # evaluate every epoch
            accuracy = self.evaluate()
            print(':: {}.dev accuracy {}\n'.format(j+1, accuracy))

            if accuracy > 0.99:
                self.net.save()
                break

    def dialog_train(self, dialog):
        # create entity tracker
        et = EntityTracker()
        # create action tracker
        at = ActionTracker(et)
        # reset network
        self.net.reset_state()

        loss = 0.
        # iterate through dialog
        for (u,r) in dialog:
            # print("Here in the dialog loop")
            u_ent = et.extract_entities(u)
            u_ent_features = et.context_features()
            u_emb = self.emb.encode(u)
            u_bow = self.bow_enc.encode(u)
            # concat features
            features = torch.autograd.Variable(torch.from_numpy(np.concatenate((u_ent_features, u_emb, u_bow), axis=0))).float()
            # print(features)
            # get action mask
            action_mask = torch.autograd.Variable(torch.from_numpy(at.action_mask()))
            r = torch.autograd.Variable(torch.LongTensor([r]))
            # print(r)
            # forward propagation
            #  train step
            loss += self.net.train_step(features, r, action_mask)
        return loss/len(dialog)

    def evaluate(self):
        # create entity tracker
        et = EntityTracker()
        # create action tracker
        at = ActionTracker(et)
        # reset network
        self.net.reset_state()

        dialog_accuracy = 0.
        r_count = 0 #Count of task 15
        count = 0 # Total count of rewards
        for dialog_idx in self.dialog_indices_dev:

            start, end = dialog_idx['start'], dialog_idx['end']
            dialog = self.dataset[start:end]
            num_dev_examples = len(self.dialog_indices_dev)

            # create entity tracker
            et = EntityTracker()
            # create action tracker
            at = ActionTracker(et)
            # reset network
            self.net.reset_state()

            # iterate through dialog
            correct_examples = 0
            for (u,r) in dialog:
                # encode utterance
                u_ent = et.extract_entities(u)
                u_ent_features = et.context_features()
                u_emb = self.emb.encode(u)
                u_bow = self.bow_enc.encode(u)
                # concat features
                # get action mask
                features = torch.autograd.Variable(torch.from_numpy(np.concatenate((u_ent_features, u_emb, u_bow), axis=0))).float()
                # print(features)
                # get action mask
                action_mask = torch.autograd.Variable(torch.from_numpy(at.action_mask()))
                # r = torch.autograd.Variable(torch.LongTensor([r]))
                # forward propagation
                #  train step
                logits,probs,prediction = self.net.forward(features, action_mask)
                # print("logits", logits)
                # print("probs", probs)
                # print("prediction", logits,probs,prediction)
                # print(prediction,r)
                correct_examples += int(prediction == r)
                if r==15:
                    r_count += 1
                count += 1
            # get dialog accuracy
            dialog_accuracy += correct_examples/len(dialog)

        print("task 15 was the answer with freq",r_count/count)

        return dialog_accuracy/num_dev_examples



if __name__ == '__main__':
    # setup trainer
    trainer = Trainer()
    # start training
    trainer.train()
