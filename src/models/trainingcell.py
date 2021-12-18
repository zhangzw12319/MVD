"""
TrainingCell
"""
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.numpy as msnp
from mindspore import Parameter, Tensor, ParameterTuple


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
        self.loss_id = Parameter(Tensor([0.0], ms.float32))
        self.loss_tri = Parameter(Tensor([0.0], ms.float32))
        self.loss_total = Parameter(Tensor([0.0], ms.float32))
        self.acc = Parameter(Tensor([0.0], ms.float32))

        self.cat = P.Concat()
        self.cast = P.Cast()
        self.sum = P.ReduceSum()
        self.max = P.ArgMaxWithValue(axis=1)
        self.eq = P.Equal()

    def get_acc(self, logits, label):
        predict, _ = self.max(logits)
        correct = self.eq(predict, label)
        return P.Div()(msnp.where(correct, 1.0, 0.0).sum() , label.shape[0])


    def construct(self, img1, img2, label1, label2):
        """
        v_observation[0] 是logits， v_observation[1]是feature
        """

        v_observation, v_representation, v_ms_observation, v_ms_representation, \
        i_observation, i_representation, i_ms_observation, i_ms_representation =\
        self._backbone(img1, x2=img2, mode=0)

        label = self.cat((label1, label2))
        label_ = self.cast(label, ms.int32)

        # loss_id = 0.5 * (self._ce_loss(v_observation[1], label_) +\
        #                 self._ce_loss(v_representation[1], label_)) +\
        #           0.5 * (self._ce_loss(i_observation[1], label_) +\
        #                 self._ce_loss(i_representation[1], label_)) +\
        #           0.25 * (self._ce_loss(v_ms_observation[1], label_) +\
        #                 self._ce_loss(v_ms_representation[1], label_)) +\
        #           0.25 * (self._ce_loss(i_ms_observation[1], label_) +\
        #                 self._ce_loss(i_ms_representation[1], label_))
        
        ##################### Only for debug1 rm representation ###########
        loss_id = 0.5 * self._ce_loss(v_observation[1], label_) +\
                  0.5 * self._ce_loss(i_observation[1], label_) +\
                  0.25 * self._ce_loss(v_ms_observation[1], label_) +\
                  0.25 * self._ce_loss(i_ms_observation[1], label_)
        ###################################################################

        ##################### Only for debug1 rm representation ###########
        # loss_id = 0.5 * self._ce_loss(v_representation[1], label_) +\
        #           0.5 * self._ce_loss(i_representation[1], label_) +\
        #           0.25 * self._ce_loss(v_ms_representation[1], label_) +\
        #           0.25 * self._ce_loss(i_ms_representation[1], label_)
        ###################################################################
        loss_total = 0

        if self.loss_func == "id":
            loss_total = loss_id
        elif self.loss_func == "id+tri":
            # loss_tri = 0.5 * (self._tri_loss(v_observation[0], label) +\
            #             self._tri_loss(v_representation[0], label))\
            #      + 0.5 * (self._tri_loss(i_observation[0], label) +\
            #             self._tri_loss(i_representation[0], label)) \
            #      + 0.25 * (self._tri_loss(v_ms_observation[0], label) +\
            #             self._tri_loss(v_ms_representation[0], label)) \
            #      + 0.25 * (self._tri_loss(i_ms_observation[0], label) +\
            #             self._tri_loss(i_ms_representation[0], label))
            ##################### Only for debug1 rm representation ###########
            loss_tri = 0.5 * self._tri_loss(v_observation[0], label) +\
                 + 0.5 * self._tri_loss(i_observation[0], label) +\
                 + 0.25 * self._tri_loss(v_ms_observation[0], label) +\
                 + 0.25 * self._tri_loss(i_ms_observation[0], label)
            ###################################################################
            loss_total = loss_id + loss_tri

            P.Depend()(loss_tri, P.Assign()(self.loss_tri, loss_tri))

        acc_tmp =\
        self.get_acc(v_observation[1], label_) + self.get_acc(v_representation[1], label_) \
        +self.get_acc(i_observation[1], label_) + self.get_acc(i_representation[1], label_) \
        +self.get_acc(v_ms_observation[1], label_) + self.get_acc(v_ms_representation[1], label_)\
        +self.get_acc(i_ms_observation[1], label_) + self.get_acc(i_ms_representation[1], label_)

        P.Depend()(acc_tmp, P.Assign()(self.acc, acc_tmp / 8.0))
        P.Depend()(loss_id, P.Assign()(self.loss_id, loss_id))
        P.Depend()(loss_total, P.Assign()(self.loss_total, loss_total))

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
    def __init__(self, network, optimizer):
        super(OptimizerWithNetAndCriterion, self).__init__(auto_prefix=True)
        self.network = network
        self.weights = ParameterTuple(optimizer.parameters)
        self.optimizer = optimizer
        self.grad = P.GradOperation(get_by_list=True)


    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        grads = self.grad(self.network, weights)(*inputs)
        P.Depend()(loss, self.optimizer(grads))
        return loss