
# !/usr/bin/env python3
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


class SegLoss(nn.Layer):
    def __init__(self):
        super(SegLoss, self).__init__()
        self.logits_list = []
        self.loss1 = CrossEntropyLoss()
        self.loss2 = RMILoss()
        self.loss21 = CrossEntropyLoss()
        self.loss3 = CrossEntropyLoss()
        self.loss4 = CrossEntropyLoss()

        self.losses = [self.loss1, self.loss2, self.loss21, self.loss3, self.loss4]
        self.coef = [0.4, 1.0, 1.0, 0.05, 0.05]
    
    def forward(self, output, labels):
        outputs = [output[0], output[1], output[1], output[2], output[3]]
        loss_list = self.loss_computation(outputs, labels, losses=self.losses, coef=self.coef)
        sum_loss = sum(loss_list)
        loss = dict()
        loss['crossentropy_loss'] = sum_loss[0] 
        return  loss


    def check_logits_losses(self, logits_list, losses):
        len_logits = len(logits_list)
        len_losses = len(losses)
        if len_logits != len_losses:
            raise RuntimeError(
                'The length of logits_list should equal to the types of loss config: {} != {}.'
                .format(len_logits, len_losses))


    def loss_computation(self, logits_list, labels, losses, coef):
        self.check_logits_losses(logits_list, losses)
        loss_list = []
        for i in range(len(logits_list)):
            logits = logits_list[i]
            loss_i = losses[i]
            coef_i = coef[i]
            loss_list.append(coef_i * loss_i(logits, labels))
        return loss_list


class SegBDD100KLoss(nn.Layer):
    def __init__(self, thresh=0.5, min_kept=10000, ignore_index=255):
        super(SegBDD100KLoss, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index

        self.logits_list = []
        self.loss1 = OhemCrossEntropyLoss(thresh=self.thresh, min_kept=self.min_kept, ignore_index=self.ignore_index)
        self.loss2 = OhemCrossEntropyLoss(thresh=self.thresh, min_kept=self.min_kept, ignore_index=self.ignore_index)
        self.loss3 = OhemCrossEntropyLoss(thresh=self.thresh, min_kept=self.min_kept, ignore_index=self.ignore_index)

        self.losses = [self.loss1, self.loss2, self.loss3, ]
        self.coef = [1, 1, 1]
    
    def forward(self, outputs, labels):
        loss_list = self.loss_computation(outputs, labels, losses=self.losses, coef=self.coef)
        sum_loss = sum(loss_list)
        loss = dict()
        loss['OhemCrossEntropyLoss'] = sum_loss[0] 
        return  loss


    def check_logits_losses(self, logits_list, losses):
        len_logits = len(logits_list)
        len_losses = len(losses)
        if len_logits != len_losses:
            raise RuntimeError(
                'The length of logits_list should equal to the types of loss config: {} != {}.'
                .format(len_logits, len_losses))


    def loss_computation(self, logits_list, labels, losses, coef):
        self.check_logits_losses(logits_list, losses)
        loss_list = []
        for i in range(len(logits_list)):
            logits = logits_list[i]
            loss_i = losses[i]
            coef_i = coef[i]
            loss_list.append(coef_i * loss_i(logits, labels))
        return loss_list



class SegDMNetLoss(nn.Layer):
    def __init__(self, ):
        super(SegDMNetLoss, self).__init__()
  
        self.loss1 = CrossEntropyLoss()
        self.loss2 = CrossEntropyLoss()
        # self.loss3 = CrossEntropyLoss()

        self.losses = [self.loss1, self.loss2]
        self.coef = [1, 0.4]
    
    def forward(self, outputs, labels):
        loss_list = self.loss_computation(outputs, labels, losses=self.losses, coef=self.coef)
        sum_loss = sum(loss_list)
        # loss = dict()
        # loss['DMNetLoss1'] = loss_list[0] 
        # loss['DMNetLoss2'] = loss_list[1]
        # loss['DMNetLoss3'] = loss_list[2]
        return  sum_loss


    def check_logits_losses(self, logits_list, losses):
        len_logits = len(logits_list)
        len_losses = len(losses)
        if len_logits != len_losses:
            raise RuntimeError(
                'The length of logits_list should equal to the types of loss config: {} != {}.'
                .format(len_logits, len_losses))


    def loss_computation(self, logits_list, labels, losses, coef):
        self.check_logits_losses(logits_list, losses)
        loss_list = []
        for i in range(len(logits_list)):
            logits = logits_list[i]
            loss_i = losses[i]
            coef_i = coef[i]
            loss_list.append(coef_i * loss_i(logits, labels))
        return loss_list



class SegSETRLoss(nn.Layer):
    def __init__(self, ):
        super(SegSETRLoss, self).__init__()
  
        self.loss1 = CrossEntropyLoss()
        self.loss2 = CrossEntropyLoss()
        self.loss3 = CrossEntropyLoss()
        self.loss4 = CrossEntropyLoss()
        self.loss5 = CrossEntropyLoss()
        self.losses = [self.loss1, self.loss2, self.loss5, self.loss5, self.loss5]
        self.coef = [1, 0.4, 0.4, 0.4, 0.4]
    
    def forward(self, outputs, labels):
        loss_list = self.loss_computation(outputs, labels, losses=self.losses, coef=self.coef)
        loss = dict()
        # loss['SegSETRLoss1'] = loss_list[0] 
        # loss['SegSETRLoss2'] = loss_list[1]
        # loss['SegSETRLoss3'] = loss_list[2]
        # loss['SegSETRLoss4'] = loss_list[3]
        # loss['SegSETRLoss5'] = loss_list[4]
        loss['seg_setr_loss'] = sum(loss_list)
        return  loss


    def check_logits_losses(self, logits_list, losses):
        len_logits = len(logits_list)
        len_losses = len(losses)
        if len_logits != len_losses:
            raise RuntimeError(
                'The length of logits_list should equal to the types of loss config: {} != {}.'
                .format(len_logits, len_losses))


    def loss_computation(self, logits_list, labels, losses, coef):
        self.check_logits_losses(logits_list, losses)
        loss_list = []
        for i in range(len(logits_list)):
            logits = logits_list[i]
            loss_i = losses[i]
            coef_i = coef[i]
            loss_list.append(coef_i * loss_i(logits, labels))
        return loss_list


class CrossEntropyLoss(nn.Layer):
    """
    Implements the cross entropy loss function.

    Args:
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        top_k_percent_pixels (float, optional): the value lies in [0.0, 1.0].
            When its value < 1.0, only compute the loss for the top k percent pixels
            (e.g., the top 20% pixels). This is useful for hard pixel mining. Default ``1.0``.
        data_format (str, optional): The tensor format to use, 'NCHW' or 'NHWC'. Default ``'NCHW'``.
    """

    def __init__(self,
                 weight=None,
                 ignore_index=255,
                 top_k_percent_pixels=1.0,
                 data_format='NCHW'):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.top_k_percent_pixels = top_k_percent_pixels
        self.EPS = 1e-8
        self.data_format = data_format
        if weight is not None:
            self.weight = paddle.to_tensor(weight, dtype='float32')
        else:
            self.weight = None

    def forward(self, logit, label, semantic_weights=None):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
            semantic_weights (Tensor, optional): Weights about loss for each pixels,
                shape is the same as label. Default: None.
        Returns:
            (Tensor): The average loss.
        """
        channel_axis = 1 if self.data_format == 'NCHW' else -1
        if self.weight is not None and logit.shape[channel_axis] != len(
                self.weight):
            raise ValueError(
                'The number of weights = {} must be the same as the number of classes = {}.'
                .format(len(self.weight), logit.shape[channel_axis]))

        if channel_axis == 1:
            logit = paddle.transpose(logit, [0, 2, 3, 1])
        label = label.astype('int64')

        loss = F.cross_entropy(
            logit,
            label,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.weight)

        return self._post_process_loss(logit, label, semantic_weights, loss)

    def _post_process_loss(self, logit, label, semantic_weights, loss):
        """
        Consider mask and top_k to calculate the final loss.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
            semantic_weights (Tensor, optional): Weights about loss for each pixels,
                shape is the same as label.
            loss (Tensor): Loss tensor which is the output of cross_entropy. If soft_label
                is False in cross_entropy, the shape of loss should be the same as the label.
                If soft_label is True in cross_entropy, the shape of loss should be
                (N, D1, D2,..., Dk, 1).
        Returns:
            (Tensor): The average loss.
        """
        mask = label != self.ignore_index
        mask = paddle.cast(mask, 'float32')
        label.stop_gradient = True
        mask.stop_gradient = True

        if loss.ndim > mask.ndim:
            loss = paddle.squeeze(loss, axis=-1)
        loss = loss * mask
        if semantic_weights is not None:
            loss = loss * semantic_weights

        if self.weight is not None:
            _one_hot = F.one_hot(label * mask, logit.shape[-1])
            coef = paddle.sum(_one_hot * self.weight, axis=-1)
        else:
            coef = paddle.ones_like(label)

        if self.top_k_percent_pixels == 1.0:
            avg_loss = paddle.mean(loss) / (paddle.mean(mask * coef) + self.EPS)
        else:
            loss = loss.reshape((-1, ))
            top_k_pixels = int(self.top_k_percent_pixels * loss.numel())
            loss, indices = paddle.topk(loss, top_k_pixels)
            coef = coef.reshape((-1, ))
            coef = paddle.gather(coef, indices)
            coef.stop_gradient = True
            coef = coef.astype('float32')
            avg_loss = loss.mean() / (paddle.mean(coef) + self.EPS)

        return avg_loss


class DistillCrossEntropyLoss(CrossEntropyLoss):
    """
    The implementation of distill cross entropy loss.

    Args:
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        top_k_percent_pixels (float, optional): the value lies in [0.0, 1.0].
            When its value < 1.0, only compute the loss for the top k percent pixels
            (e.g., the top 20% pixels). This is useful for hard pixel mining.
            Default ``1.0``.
        data_format (str, optional): The tensor format to use, 'NCHW' or 'NHWC'.
            Default ``'NCHW'``.
    """

    def __init__(self,
                 weight=None,
                 ignore_index=255,
                 top_k_percent_pixels=1.0,
                 data_format='NCHW'):
        super().__init__(weight, ignore_index, top_k_percent_pixels,
                         data_format)

    def forward(self,
                student_logit,
                teacher_logit,
                label,
                semantic_weights=None):
        """
        Forward computation.

        Args:
            student_logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            teacher_logit (Tensor): Logit tensor, the data type is float32, float64. The shape
                is the same as the student_logit.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
            semantic_weights (Tensor, optional): Weights about loss for each pixels,
                shape is the same as label. Default: None.
        """

        if student_logit.shape != teacher_logit.shape:
            raise ValueError(
                'The shape of student_logit = {} must be the same as the shape of teacher_logit = {}.'
                .format(student_logit.shape, teacher_logit.shape))

        channel_axis = 1 if self.data_format == 'NCHW' else -1
        if self.weight is not None and student_logit.shape[channel_axis] != len(
                self.weight):
            raise ValueError(
                'The number of weights = {} must be the same as the number of classes = {}.'
                .format(len(self.weight), student_logit.shape[channel_axis]))

        if channel_axis == 1:
            student_logit = paddle.transpose(student_logit, [0, 2, 3, 1])
            teacher_logit = paddle.transpose(teacher_logit, [0, 2, 3, 1])

        teacher_logit = F.softmax(teacher_logit)

        loss = F.cross_entropy(
            student_logit,
            teacher_logit,
            weight=self.weight,
            reduction='none',
            soft_label=True)

        return self._post_process_loss(student_logit, label, semantic_weights,
                                       loss)

class MixedLoss(nn.Layer):
    """
    Weighted computations for multiple Loss.
    The advantage is that mixed loss training can be achieved without changing the networking code.

    Args:
        losses (list[nn.Layer]): A list consisting of multiple loss classes
        coef (list[float|int]): Weighting coefficient of multiple loss

    Returns:
        A callable object of MixedLoss.
    """

    def __init__(self, losses, coef):
        super(MixedLoss, self).__init__()
        if not isinstance(losses, list):
            raise TypeError('`losses` must be a list!')
        if not isinstance(coef, list):
            raise TypeError('`coef` must be a list!')
        len_losses = len(losses)
        len_coef = len(coef)
        if len_losses != len_coef:
            raise ValueError(
                'The length of `losses` should equal to `coef`, but they are {} and {}.'
                .format(len_losses, len_coef))

        self.losses = losses
        self.coef = coef

    def forward(self, logits, labels):
        loss_list = []
        for i, loss in enumerate(self.losses):
            output = loss(logits, labels)
            loss_list.append(output * self.coef[i])
        return loss_list




_euler_num = 2.718281828
_pi = 3.14159265
_ln_2_pi = 1.837877
_CLIP_MIN = 1e-6
_CLIP_MAX = 1.0
_POS_ALPHA = 5e-4
_IS_SUM = 1


class RMILoss(nn.Layer):
    """
    Implements the Region Mutual Information(RMI) Loss（https://arxiv.org/abs/1910.12037） for Semantic Segmentation.
    Unlike vanilla rmi loss which contains Cross Entropy Loss, we disband them and only
    left the RMI-related parts.
    The motivation is to allow for a more flexible combination of losses during training.
    For example, by employing mixed loss to merge RMI Loss with Boostrap Cross Entropy Loss,
    we can achieve the online mining of hard examples together with attention to region information.
    Args:
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self,
                 num_classes=19,
                 rmi_radius=3,
                 rmi_pool_way=0,
                 rmi_pool_size=3,
                 rmi_pool_stride=3,
                 loss_weight_lambda=0.5,
                 ignore_index=255):
        super(RMILoss, self).__init__()

        self.num_classes = num_classes
        assert rmi_radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.rmi_radius = rmi_radius
        assert rmi_pool_way in [0, 1, 2, 3]
        self.rmi_pool_way = rmi_pool_way
        assert rmi_pool_size == rmi_pool_stride
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride