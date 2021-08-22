# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>annToMask, COCO>showAnns

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import copy
import json
import os
import sys
import time
from collections import defaultdict
from typing import Optional, Callable, List, Any, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools import mask as maskUtils
# from torchvision.datasets import VisionDataset

PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def hdf5_bunch_get(hdf5_dataset, indices, sorted=False):
    if sorted:
        items = [p for p in hdf5_dataset[indices]]
        items = np.array(items)
    else:
        img_idxs_sorted, reverse_indices = np.unique(indices, return_inverse=True)
        items = [p for p in hdf5_dataset[img_idxs_sorted.tolist()]]
        items = np.array(items)[reverse_indices]
    return items


class COCO:
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        # self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        # self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.catToImgs = defaultdict(list)
        self.catToImgsSet = None
        self.num_categ = None
        self.num_image = None
        self.num_annos = None

        if not annotation_file == None:
            self.anno_file = annotation_file
            self.anno_hdf5 = None
            print('loading annotations into memory...')
            tic = time.time()
            # dataset = json.load(open(annotation_file, 'r'))
            # assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            # self.dataset = dataset
            self.createIndex()
            print('Done (t={:0.2f}s)'.format(time.time() - tic))

    def open_hdf5(self):
        # we need this function to open anno_hdf5 file after the object is forked
        if self.anno_hdf5 is None:
            self.anno_hdf5 = h5py.File(self.anno_file, 'r')

    def createIndex(self):
        # create index
        print('creating index...')

        # don't set self.anno_hdf5 in this function since h5py file can't be forked
        # in multiple processes
        anno_hdf5 = h5py.File(self.anno_file, 'r')
        self.num_categ = anno_hdf5['name'].shape[0]
        self.num_image = anno_hdf5['file_name'].shape[0]
        self.num_annos = anno_hdf5['image_id'].shape[0]

        # build catToImgs: category id => [image_id0 image_id1]
        catToImgs = defaultdict(list)

        # get all items will be much faster compared to access one by one
        category_ids = anno_hdf5['category_id'][:].tolist()
        image_ids = anno_hdf5['image_id'][:].tolist()
        for idx, cat_id in enumerate(category_ids):
            image_id = image_ids[idx]
            catToImgs[cat_id].append(image_id)
        self.catToImgs = catToImgs  # category id => [image_id0 image_id1]

        catToImgsSet = {}
        for key, value in catToImgs.items():
            catToImgsSet[key] = set(value)
        self.catToImgsSet = catToImgsSet

        # build cats
        cat_ids = anno_hdf5['id'][:].tolist()
        cat_names = anno_hdf5['name'][:].tolist()
        cat_supercategories = anno_hdf5['supercategory'][:].tolist()
        cats = {}
        for cid, name, supercategory in zip(cat_ids, cat_names, cat_supercategories):
            cats[cid] = {'id': cid, 'name': name, 'supercategory': supercategory}
        self.cats = cats

        # anns, cats, imgs = {}, {}, {}
        # imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        # if 'annotations' in self.dataset:
        #     for ann in self.dataset['annotations']:
        #         imgToAnns[ann['image_id']].append(ann)
        #         anns[ann['id']] = ann
        #
        # if 'images' in self.dataset:
        #     for img in self.dataset['images']:
        #         imgs[img['id']] = img
        #
        # if 'categories' in self.dataset:
        #     for cat in self.dataset['categories']:
        #         cats[cat['id']] = cat
        #
        # if 'annotations' in self.dataset and 'categories' in self.dataset:
        #     for ann in self.dataset['annotations']:
        #         catToImgs[ann['category_id']].append(ann['image_id'])
        # create class members
        # self.anns = anns  # anno id => anno
        # self.imgToAnns = imgToAnns  # image id => [ann0 anno1 anno2]
        # self.catToImgs = catToImgs  # category id => [image_id0 image_id1]
        # self.imgs = imgs  # image id => image_info
        # self.cats = cats  # cat id => cat info

        print('index created!')

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        raise NotImplementedError
        # for key, value in self.dataset['info'].items():
        #     print('{}: {}'.format(key, value))

    # needed by v2x
    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        self.open_hdf5()

        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anno_ids = range(self.num_annos)
            # anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                anno_ids = []
                for imgId in imgIds:
                    if imgId < self.num_image:
                        start_idx = self.anno_hdf5['start_idx'][imgId]
                        end_idx = self.anno_hdf5['start_idx'][imgId + 1]
                        for _id in range(start_idx, end_idx):
                            anno_ids.append(_id)
                # lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                # anns = list(itertools.chain.from_iterable(lists))
            else:
                anno_ids = range(self.num_annos)
                # anns = self.dataset['annotations']

            if len(catIds) != 0:
                cate_ids = hdf5_bunch_get(self.anno_hdf5['category_id'], anno_ids)
                anno_ids = [_id for _id, cid in zip(anno_ids, cate_ids) if cid in catIds]

            if len(areaRng) != 0:
                areas = hdf5_bunch_get(self.anno_hdf5['area'], anno_ids)
                anno_ids = [_id for _id, area in zip(anno_ids, areas) if area > areaRng[0] and area < areaRng[1]]

            # anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            # anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]

        if not iscrowd == None:
            iscrowds = hdf5_bunch_get(self.anno_hdf5['iscrowd'], anno_ids)
            anno_ids = [id_ for id_, iscrowd_ in zip(anno_ids, iscrowds) if iscrowd_ == iscrowd]
            # ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            anno_ids = anno_ids
            # ids = [ann['id'] for ann in anns]
        return anno_ids

    # needed by v2x
    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        self.open_hdf5()

        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats_ids = self.anno_hdf5['id'][:].tolist()
            # cats = self.dataset['categories']
        else:
            cats_ids = self.anno_hdf5['id'][:].tolist()
            # cats = self.dataset['categories']
            if len(catNms) != 0:
                cats_ids = [id_ for id_ in cats_ids if self.cats[id_]['name'] in catNms]

            if len(supNms) != 0:
                cats_ids = [id_ for id_ in cats_ids if self.cats[id_]['supercategory'] in supNms]

        # ids = [cat['id'] for cat in cats]
        return cats_ids

    # needed by v2x
    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        self.open_hdf5()

        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = range(self.num_image)
            # ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = self.catToImgsSet[catId]
                else:
                    ids &= self.catToImgsSet[catId]
        return list(ids)

    def process_attr(self, key, value):
        if key in ['file_name', 'name', 'supercategory', 'seg_direction', 'seg_sub_type']:
            value = value.decode()
        elif key in ['bbox', 'lights', 'vis_line']:
            value = value.tolist()
        elif key == 'mid_line':
            a, b = value.tolist()
            a = int(a)  # the first value of mid_line is int
            value = [a, b]
        elif key == 'box_3d':
            value = value.tolist()
            new_value = []
            for idx in range(len(value) // 2):
                new_value.append({'y': value[2 * idx], 'x': value[2 * idx + 1]})
            value = new_value
        elif key in ['rotation_ori', 'rotation_3d', 'rotation_flip']:
            value = value.tolist()
            value = {'theta': value[0], 'phi': value[1], 'psi': value[2]}
        elif key in ['center_3d']:
            value = value.tolist()
            value = {'y': value[0], 'x': value[1], 'z': value[2]}
        elif key in ['transform']:
            value = value.tolist()
            value = {
                'rotation': {'y': value[0], 'x': value[1], 'z': value[2], 'w': value[3]},
                'translation': {'y': value[4], 'x': value[5], 'z': value[6]}
            }

        return value

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        self.open_hdf5()

        if type(ids) == int:
            ids = [ids]
        if len(ids) == 0:
            return []

        # this op should be fast, since it will be frequently invoked
        attr_pos = hdf5_bunch_get(self.anno_hdf5['attr_pos'], ids)

        annos = [{} for _ in range(len(ids))]
        for idx, key in enumerate(self.anno_hdf5.attrs['anno_keys']):
            one_attr_pos = attr_pos[:, idx]
            # remove negative index
            one_attr_pos_exist = one_