# !/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import types
from collections import UserDict
from typing import List

from detectron2.utils.logger import log_first_n

__all__ = ["DatasetCatalog", "MetadataCatalog", "Metadata"]


class _DatasetCatalog(UserDict):
    """
    A global dictionary that stores information about the datasets and how to obtain them.

    It contains a mapping from strings
    (which are names that identify a dataset, e.g. "coco_2014_train")
    to a function which parses the dataset and returns the samples in the
    format of `list[dict]`.

    The returned dicts should be in Detectron2 Dataset format (See DATASETS.md for details)
    if used with the data loader functionalities in `data/build.py,data/detection_transform.py`.

    The purpose of having this catalog is to make it easy to choose
    different datasets, by just using the strings in the config.
    """

    def register(self, name, func):
        """
        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".
            func (callable): a callable which takes no arguments and returns a list of dicts.
                It must return the same results if called multiple times.
        """
        assert callable(func), "You must register a function with `DatasetCatalog.register`!"
        assert name not in self, "Dataset '{}' is already registered!".format(name)
        self[name] = func

    def get(self, name):
        """
        Call the registered function and return its results.

        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".

        Returns:
            list[dict]: dataset annotations.
        """
        try:
            f = self[name]
        except KeyError as e:
            raise KeyError(
                "Dataset '{}' is not registered! Available datasets are: {}".format(
                    name, ", ".join(list(self.keys()))
                )
            ) from e
        return f()

    def list(self) -> List[str]:
        """
        List all registered datasets.

        Returns:
            list[str]
        """
        return list(self.keys())

    def remove(self, name):
        """
        Alias of ``pop``.
        """
        self.pop(name)

    def __str__(self):
        return "DatasetCatalog(registered datasets: {})".format(", ".join(self.keys()))

    __repr__ = __str__


DatasetCatalog = _DatasetCatalog()
DatasetCatalog.__doc__ = (
    _DatasetCatalog.__doc__
    + """
    .. automethod:: detectron2.data.catalog.DatasetCatalog.register
    .. automethod:: detectron2.data.catalog.DatasetCatalog.get
"""
)


class Metadata(types.SimpleNamespace):
    """
    A class that supports simple attribute setter/getter.
    It is intended for storing metadata of a dataset and make it accessible globally.

    Examples:
    ::
        # somewhere when you load the data:
        MetadataCatalog.get("mydataset").thing_classes = ["person", "dog"]

        # somewhere when you print statistics or visualize:
        classes = MetadataCatalog.get("mydataset").thing_classes
    """

    # the name of the dataset
    # set default to N/A so that `self.name` in the errors will not trigger getattr again
    name: str = "N/A"

    _RENAMED = {
        "class_names": "thing_classes",
        "dataset_id_to_contiguous_id": "thing_dataset_id_to_contiguous_id",
        "stuff_class_names": "stuff_classes",
    }

    def __getattr__(self, key):
        if key in self._RENAMED:
            log_first_n(
                logging.WARNING,
                "Metadata '{}' was renamed to '{}'!".format(key, self._RENAMED[key]),
                n=10,
            )
            return getattr(self, self._RENAMED[key])

        # "name" exists in every metadata
        if len(self.__dict__) > 1:
            raise AttributeError(
                "Attribute '{}' does not exist in the metadata of dataset '{}'. Available "
                "keys are {}.".format(key, self.name, str(self.__dict__.keys()))
            )
        else:
            raise AttributeError(
                "Attribute '{key}' does not exist in the metadata of dataset '{self.name}': "
                "metadata is empty."
            )

    def __setattr__(self, key, val):
        if key in self._RENAMED:
            log_first_n(
                logging.WARNING,
                "Metadata '{}' was renamed to '{}'!".format(key, self._RENAMED[key]),
                n=10,
            )
            setattr(self, self._RENAMED[key], val)

        # Ensure that metadata of the same name stays consistent
        try:
            oldval = getattr(self, key)
            assert oldval == val, (
                "Attribute '{}' in the metadata of '{}' cannot be set "
                "to a different value!\n{} != {}".format(key, self.name, oldval, val)
            )
        except AttributeError:
            super().__setattr__(key, val)

    def as_dict(self):
        """
        Returns all the m