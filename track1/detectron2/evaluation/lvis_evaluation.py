# !/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import json
import logging
import os
import pickle
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from .coco_evaluation import instances_to_coco_json
from .evaluator import DatasetEvaluator


class LVISEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal and instance detection/segmentation outputs using
    LVIS's metrics and evaluation API.
    """

    def __init__(
        self,
        dataset_name,
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                "json_file": the path to the LVIS format annotation
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
            max_dets_per_image (None or int): limit on maximum detections per image in evaluating AP
                This limit, by default of the LVIS dataset, is 300.
        """
        from lvis import LVIS

        self._logger = logging.getLogger(__name__)

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warn(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._distributed = distributed
        self._output_dir = output_dir
        self._max_dets_per_image = max_dets_per_image

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(self._metadata.json_file)
        self._lvis_api = LVIS(json_file)
        # Test set json files do not contain annotations (evaluation must be
        # performed using the LVIS evaluation server).
        self._do_evaluation = len(self._lvis_api.get_ann_ids()) > 0

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a LVIS model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a LVIS model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[LVISEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _tasks_from_predictions(self, predictions):
        for pred in predictions:
            if "segmentation" in pred:
                return ("bbox", "segm")
        return ("bbox",)

    def _eval_predictions(self, predictions):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.

        Args:
            predictions (list[dict]): list of outputs from the model
        """
        self._logger.info("Preparing results in the LVIS format ...")
        lvis_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(lvis_results)

        # LVIS evaluator can be used to evaluate results for COCO dataset categories.
        # In this case `_metadata` variable will have a field with COCO-specific category mapping.
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
       