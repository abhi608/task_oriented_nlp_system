import torch
import numpy as np
from itertools import chain
from data_utils import load_candidates, load_dialog_task, vectorize_candidates, vectorize_data
from functools import reduce

class CDATA(object):
    def __init__(self, data_dir, task_id, OOV=False, memory_size=50, train=0, batch_size=32, nn=False):
        self.data_dir = data_dir
        self.task_id = task_id
        self.OOV = OOV
        self.memory_size = memory_size
        self.train = train
        self.batch_size = batch_size
        self.nn = nn
        candidates, self.candid2indx = load_candidates(self.data_dir, self.task_id)
        self.n_cand = len(candidates)
        print("Candidate Size", self.n_cand)
        self.indx2candid = dict((self.candid2indx[key], key) for key in self.candid2indx)
        self.trainData, self.testData, self.valData = load_dialog_task(
            self.data_dir, self.task_id, self.candid2indx, self.OOV)
        data = self.trainData + self.testData + self.valData
        self.build_vocab(data, candidates)
        self.candidates_vec = vectorize_candidates(
            candidates, self.word_idx, self.candidate_sentence_size)
        self.params = {
            'n_cand': self.n_cand,
            'indx2candid': self.indx2candid,
            'candid2indx': self.candid2indx,
            'candidates_vec': self.candidates_vec,
            'word_idx': self.word_idx,
            'sentence_size': self.sentence_size,
            'candidate_sentence_size': self.candidate_sentence_size,
            'vocab_size': self.vocab_size
        }

        if self.nn:
            if self.train == 0:
                self.S, self.Q, self.A = vectorize_data(
                    self.trainData, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size, nn=self.nn)
            elif self.train == 1:
                self.S, self.Q, self.A = vectorize_data(
                    self.valData, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size, nn=self.nn)
            elif self.train == 2:
                self.S, self.Q, self.A = vectorize_data(
                    self.testData, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size, nn=self.nn)
        else:
            if self.train == 0:
                self.S, self.Q, self.A = vectorize_data(
                    self.trainData, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
            elif self.train == 1:
                self.S, self.Q, self.A = vectorize_data(
                    self.valData, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)
            elif self.train == 2:
                self.S, self.Q, self.A = vectorize_data(
                    self.testData, self.word_idx, self.sentence_size, self.batch_size, self.n_cand, self.memory_size)

    def getData(self):
        return self.S, self.Q, self.A

    def getParam(self, parameter):
        return self.params[parameter]

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
