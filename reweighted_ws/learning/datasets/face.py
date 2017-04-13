"""

Access to the FACE dataset of handwritten digits.

"""

from __future__ import division

import os
import logging
import cPickle as pickle
import gzip

import numpy as np

import theano
import theano.tensor as T

from learning.datasets import DataSet, datapath

_logger = logging.getLogger(__name__)

floatX = theano.config.floatX

#-----------------------------------------------------------------------------

class FACE(DataSet):
    def __init__(self, which_set='train', n_datapoints=None, fname="FACE.pkl.gz", preproc=[]):
        super(FACE, self).__init__(preproc)

        _logger.info("Loading FACE data")
        fname = datapath(fname)


        if which_set == 'train':
            data_name = 'toronto_face_train.npy'
            label_name = 'toronto_face_train_label.npy'

        elif which_set == 'valid':
            data_name = 'toronto_face_train.npy'
            label_name = 'toronto_face_train_label.npy'

        elif which_set == 'test':
            data_name = 'toronto_face_train.npy'
            label_name = 'toronto_face_train_label.npy'

        elif which_set == 'salakhutdinov_train':
            raise ValueError("Unknown dataset %s" % which_set)

        elif which_set == 'salakhutdinov_valid':
            raise ValueError("Unknown dataset %s" % which_set)

        else:
            raise ValueError("Unknown dataset %s" % which_set)

        x = np.load(data_name)
        y = np.load(label_name)
        self.X, self.Y = self.prepare(x, y, n_datapoints)

        self.n_datapoints = self.X.shape[0]

    def prepare(self, x, y, n_datapoints):
        N = x.shape[0]
        assert N == y.shape[0]

        if n_datapoints is not None:
            N = n_datapoints

        x = x[:N]
        y = y[:N]

        one_hot = np.zeros((N, 10), dtype=floatX)
        for n in xrange(N):
            one_hot[n, y[n]] = 1.

        return x.astype(floatX), one_hot.astype(floatX)

