import os
import traceback
import six
import sys
if sys.version_info >= (3, 0):
    pass
else:
    pass
import numpy as np
from collections import defaultdict

from paddle.io import DataLoader, DistributedBatchSampler
from paddle.fluid.dataloader.collate import default_collate_fn

from ppdet.core.workspace import register
from .. import transforms as transform
    
from ppdet.utils.logger import setup_logger
from collections import Counter
logger = setup_logger('reader')

MAIN_PID = os.getpid()


class Compose(object):
    def __init__(self, transforms, num_classes=80):
        self.transforms = transforms
        self.transforms_cls = []
        self.transforms_det = []
        for t in self.transforms:
            for k, v in t.items():
                op_cls = getattr(transform, k)
                f = op_cls(**v)
                if hasattr(f, 'num_classes'):
                    f.num_classes = num_classes

                self.transforms_cls.append(f)
                if k not in ['RandomErasing', 'TimmAutoAugment']:
                    self.transforms_det.append(f)

    def __call__(self, data):
        if 'is_cls' in data and data['is_cls']:
            for f in self.transforms_cls:
                # skip TimmAutoAugment for vehicle color data
                if type(f).__name__ == "TimmAutoAugment" and data['cls_label_1'] == -1:
                    continue
                try:
                    data = f(data)
                except Exception as e:
                    stack_info = traceback.format_exc()
                    logger.warning("fail to map sample transform [{}] "
                                "with error: {} and stack:\n{}".format(
                                    f, e, str(stack_info)))
                    raise e
        else:
            for f in self.transforms_det:
                try:
                    data = f(data)
                except Exception as e:
                    stack_info = traceback.format_exc()
                    logger.warning("fail to map sample transform [{}] "
                                "with error: {} and stack:\n{}".format(
                                    f, e, str(stack_info)))
                    raise e
        return data


class BatchCompose(Compose):
    def __init__(self, transforms, num_classes=80, collate_batch=True):
        super(BatchCompose, self).__init__(transforms, num_classes)
        self.collate_batch = collate_batch

    def __call__(self, data):
        for f in self.transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warning("fail to map batch transform [{}] "
                               "with error: {} and stack:\n{}".format(
                                   f, e, str(stack_info)))
                raise e

        # remove keys which is not needed by model
        extra_key = ['h', 'w', 'flipped']
        for k in extra_key:
            for sample in data:
                if k in sample:
                    sample.pop(k)

        # batch data, if user-define batch function needed
        # use user-defined here
        if self.collate_batch:
            batch_data = default_collate_fn(data)
        else:
            batch_data = {}
            for k in data[0].keys():
                tmp_data = []
                for i in range(len(data)):
                    tmp_data.append(data[i][k])
                if not 'gt_' in k and not 'is_crowd' in k and not 'difficult' in k:
                    tmp_data = np.stack(tmp_data, axis=0)
                batch_data[k] = tmp_data
        return batch_data


class BaseDataLoader(object):
    """
    Base DataLoader implementation for detection models

    Args:
        sample_transforms (list): a list of transforms to perform
                                  on each sample
        batch_transforms (list): a list of transforms to perform
                                 on batch
        batch_size (int): batch size for batch collating, default 1.
        shuffle (bool): whether to shuffle samples
        drop_last (bool): whether to drop the last incomplete,
                          default False
        num_classes (int): class number of dataset, default 80
        collate_batch (bool): whether to collate batch in dataloader.
            If set to True, the samples will collate into batch according
            to the batch size. Otherwise, the ground-truth will not collate,
            which is used when the number of ground-truch is different in 
            samples.
        use_shared_memory (bool): whether to use shared memory to
                accelerate data loading, enable this only if you
                are sure that the shared memory size of your OS
                is larger than memory cost of input datas of model.
                Note that shared memory will be automatically
                disabled if the shared memory of OS is less than
                1G, which is not enough for detection models.
                Default False.
    """

    def __init__(self,
                 sample_transforms=[],
                 batch_transforms=[],
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 num_classes=80,
                 collate_batch=True,
                 use_shared_memory=False,
                 cls_aware_sample=False,
                 multi_task_sample=False,
                 **kwargs):
        # sample transform
        self._sample_transforms = Compose(
            sample_transforms, num_classes=num_classes)

        # batch transfrom 
        self._batch_transforms = BatchCompose(batch_transforms, num_classes,
                                              collate_batch)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.use_shared_memory = use_shared_memory
        self.kwargs = kwargs
        self.cls_aware_sample = cls_aware_sample
        self.multi_task_sample = multi_task_sample

    def __call__(self,
                 dataset,
                 worker_num,
                 batch_sampler=None,
                 return_list=False):
        self.dataset = dataset
        self.dataset.check_or_download_dataset()
        self.dataset.parse_dataset()
        # get data
        self.dataset.set_transform(self._sample_transforms)
        # set kwargs
        self.dataset.set_kwargs(**self.kwargs)
        # batch sampler
        if batch_sampler is None:
            if self.cls_aware_sample:
                print('using class aware sampler')
                self._batch_sampler = ClassAwareSampler(
                    self.dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    drop_last=self.drop_last)
            elif self.multi_task_sample:
                print('using multi task sampler')
                self._batch_sampler = MultiTaskSampler(
                    self.dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    drop_last=self.drop_last)
            else:
                self._batch_sampler = DistributedBatchSampler(
                    self.dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    drop_last=self.drop_last)
        else:
            self._batch_sampler = batch_sampler

        # DataLoader do not start sub-process in Windows and Mac
        # system, do not need to use shared memory
        use_shared_memory = self.use_shared_memory and \
                            sys.platform not in ['win32', 'darwin']

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_sampler=self._batch_sampler,
            collate_fn=self._batch_transforms,
            num_workers=worker_num,
            return_list=return_list,
            use_shared_memory=use_shared_memory)
        self.loader = iter(self.dataloader)

        return self

    def __len__(self):
        return len(self._batch_sampler)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.loader)
        except StopIteration:
            self.loader = iter(self.dataloader)
            six.reraise(*sys.exc_info())

    def next(self):
        # python2 compatibility
        return self.__next__()


class ClassAwareSampler(DistributedBatchSampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        super(ClassAwareSampler, self).__init__(dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
        self.batch_size = batch_size
        self.category_imgids = self._classaware_sampler(dataset.roidbs)
        
    def __iter__(self):
        for _ in range(len(self)):
            batch = []
            # # plate head detection
            random_categories = list(np.random.choice([1,    2 ], self.batch_size, replace=True, 
                                                    p=[0.5, 0.5]))
            category_counts = defaultdict(int)
            for cls in random_categories:
                category_counts[cls] += 1
            for cls, c