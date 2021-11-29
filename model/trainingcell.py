"""
TrainingCell
"""
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import ParameterTuple


class CriterionWithNet(nn.Cell):
    """
    Mindpore module includes network and loss function
    """
    def __init__(self, backbone, ce_loss, tri_loss, kl_div, t=1, loss_func='id'):
        super(CriterionWithNet, self).__init__()
        self._backbone = backbone
        self._ce_loss = ce_loss
        self._tri_loss = tri_loss
        self._kl_div = kl_div
        self.t = t
        self.softmax = nn.Softmax()
        self.loss_func = loss_func
        self.id_loss = 0
        self.tri_loss = 0
        self.acc = 0
        self.scale = 0.0

        self.cat = P.Concat()
        self.cast = P.Cast()
        self.sum = P.ReduceSum()
        self.max = P.ArgMaxWithValue(axis=1)
        self.eq = P.Equal()

    def _get_acc(self, logits, label):
        predict, _ = self.max(logits)
        correct = self.eq(predict, label)
        return np.where(correct)[0].shape[0] / label.shape[0]


    def construct(self, img1, img2, label1, label2):
        """
        v_observation[0] 是logits， v_observation[1]是feature
        """

        v_observation, v_representation, v_ms_observation, v_ms_representation, \
        i_observation, i_representation, i_ms_observation, i_ms_representation =\
        self._backbone(img1, x2=img2, mode=0)

        label = self.cat((label1, label2))
        label_ = self.cast(label, ms.int32)

        loss_id = 0.5 * (self._ce_loss(v_observation[1], label_) +\
                        self._ce_loss(v_representation[1], label_)) +\
                  0.5 * (self._ce_loss(i_observation[1], label_) +\
                        self._ce_loss(i_representation[1], label_)) +\
                  0.25 * (self._ce_loss(v_ms_observation[1], label_) +\
                        self._ce_loss(v_ms_representation[1], label_)) +\
                  0.25 * (self._ce_loss(i_ms_observation[1], label_) +\
                        self._ce_loss(i_ms_representation[1], label_))

        loss_tri = 0.5 * (self._tri_loss(v_observation[0], label_) +\
                        self._tri_loss(v_representation[0], label_))\
                 + 0.5 * (self._tri_loss(i_observation[0], label_) +\
                        self._tri_loss(i_representation[0], label_)) \
                 + 0.25 * (self._tri_loss(v_ms_observation[0], label_) +\
                        self._tri_loss(v_ms_representation[0], label_)) \
                 + 0.25 * (self._tri_loss(i_ms_observation[0], label_) +\
                        self._tri_loss(i_ms_representation[0], label_))

        loss_total = 0
        for k in self.loss_func.split("+"):

            if k == 'tri':
                loss_total += loss_tri
            if k == 'id':
                loss_total += loss_id
            if k == 'kldiv':
                loss_vsd = \
                self._kl_div(self.softmax(v_observation[1] / self.t),\
                    self.softmax(v_representation[1] / self.t)) +\
                self._kl_div(self.softmax(i_observation[1] / self.t),\
                    self.softmax(i_representation[1] / self.t))

                loss_vcd =\
                0.5 * self._kl_div(self.softmax(v_ms_observation[1] / self.t),\
                        self.softmax(i_ms_representation[1] / self.t)) \
                +0.5 * self._kl_div(self.softmax(i_ms_observation[1] / self.t),\
                        self.softmax(v_ms_representation[1] / self.t))

                loss_total += loss_vsd + loss_vcd

        self.acc =\
        self._get_acc(v_observation[1], label_) + self._get_acc(v_representation[1], label_) \
        +self._get_acc(i_observation[1], label_) + self._get_acc(i_representation[1], label_) \
        +self._get_acc(v_ms_observation[1], label_) + self._get_acc(v_ms_representation[1], label_)\
        +self._get_acc(i_ms_observation[1], label_) + self._get_acc(i_ms_representation[1], label_)

        self.acc = self.acc / 8.0

        self.loss_id = loss_id
        self.loss_tri = loss_tri

        return loss_total

    @property
    def backbone_network(self):
        """
        return backbone
        """
        return self._backbone

class OptimizerWithNetAndCriterion(nn.Cell):
    """
    Mindspore Cell incldude Network, Optimizer and loss function.
    """
    def __init__(self, network, optimizer, sens=1.0):
        super(OptimizerWithNetAndCriterion, self).__init__(auto_prefix=True)
        self.network = network
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = P.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

    def set_sens(self, value):
        """
        set sens
        """
        self.sens = value

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        return P.depend(loss, self.optimizer(grads))
