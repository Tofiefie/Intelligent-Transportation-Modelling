# !/usr/bin/env python3
# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import itertools
from typing import Optional
import logging

import numpy as np
from paddle.io import Sampler

from utils import comm
logger = logging.getLogger(__name__)

class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, dataset, shuffle=True, seed=None, dp_group=None, moe_group=None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = len(dataset)
        assert self._size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        # self._rank = comm.get_rank()
        # self._world_size = comm.get_world_size()
        if dp_group is None: 
            self._rank = comm.get_rank()
            self._world_size = comm.get_world_size()
        else:
            self._rank = comm.get_rank() // moe_group.nranks
            self._world_size = dp_group.nranks
        logger.info("dataset {}: rank {} is mapped to _rank {} under the real local world size {}".format(dataset.dataset_name, comm.get_rank(), self._rank, self._world_size))
        self._generate_indices()

    def _generate_indices(self,):
        indices = self._finite_indices()
        
        local_indices = []
        for i in range(self._rank, len(indices), self._world_size):
            local_indices.append(indices[i])
        self.local_indices = local_indices

    def __iter__(self):
        # while True:
        #     for indice in self.local_indices:
        #         yield indice
        #     self._seed += 1
        #     self._generate_indices()
        # yield from self.local_indices
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)
    
    def __len__(self,):
        return 0 #len(self.local_indices)

    def _finite_indices(self,):
        np.random.seed(self._seed)
        if self._shuffle:
            ret_indices =  np.random.permutation(s