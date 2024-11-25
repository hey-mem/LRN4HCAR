import torch
import pickle
import time
import os
from torch.nn import functional as F
from torch.utils import data as torch_data

from torch_scatter import scatter_add, scatter_mean

from torchdrug import core, tasks, metrics
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("task.LogicalQuery")
class LogicalQuery(tasks.Task, core.Configurable):
    """
    Logical query task.

    Parameters:
        model (nn.Module): logical query model
        criterion (str, list or dict, optional): training criterion(s). Only ``bce`` is available for now.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mrr``, ``hits@K``, ``mape`` and ``spearmanr``.
        adversarial_temperature (float, optional): temperature for self-adversarial negative sampling.
            Set ``0`` to disable self-adversarial negative sampling.
        sample_weight (bool, optional): whether to weight each query by its number of answers
    """

    _option_members = ["criterion", "metric", "query_type_weight"]

    def __init__(self, model,dataset="PWA", criterion="bce", metric=("mrr",), adversarial_temperature=0.1, sample_weight=False):
        super(LogicalQuery, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.dataset = dataset
        self.adversarial_temperature = adversarial_temperature
        self.sample_weight = sample_weight

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        self.id2type = dataset.id2type
        self.type2id = dataset.type2id

        self.register_buffer("fact_graph", dataset.fact_graph)
        self.register_buffer("graph", dataset.graph)

        return train_set, valid_set, test_set

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

                is_positive = target > 0.5
                is_negative = target <= 0.5
                num_positive = is_positive.sum(dim=-1)
                num_negative = is_negative.sum(dim=-1)
                neg_weight = torch.zeros_like(pred)
                neg_weight[is_positive] = (1 / num_positive.float()).repeat_interleave(num_positive)
                if self.adversarial_temperature > 0:
                    with torch.no_grad():
                        logit = pred[is_negative] / self.adversarial_temperature
                        neg_weight[is_negative] = functional.variadic_softmax(logit, num_negative)
                else:
                    neg_weight[is_negative] = (1 / num_negative.float()).repeat_interleave(num_negative)
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)

            if self.sample_weight:
                sample_weight = target.sum(dim=-1).float()
                loss = (loss * sample_weight).sum() / sample_weight.sum()
            else:
                loss = loss.mean()

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict_and_target(self, batch, all_loss=None, metric=None):
        query = batch["query"]
        type = batch["type"]
        answer = batch["answer"]
        pred = self.model(self.fact_graph, query, all_loss, metric)

        if all_loss is None:
            target = (type, answer)
            ranking,order = self.batch_evaluate(pred, target)

            prob = F.sigmoid(pred)
            num_pred = (prob * (prob > 0.5)).sum(dim=-1)
            num = answer.sum(dim=-1)
            return (ranking, num_pred,order), (type, num,query,answer)
        else:
            target = answer.float()

        return pred, target

    def batch_evaluate(self, pred, target):
        type, answer = target

        num = answer.sum(dim=-1)

        num_answer = num

        order = pred.argsort(dim=-1, descending=True)
        range = torch.arange(self.num_entity, device=self.device)
        ranking = scatter_add(range.expand_as(order), order, dim=-1)
        ranking = ranking[answer]
        return ranking,order


    def evaluate(self, pred, target):
        ranking, num_pred, order = pred
        type, num,query,answer = target
        metric = {}

        query_list = []

        num_rows, num_cols = query.size()

        if self.dataset == "PWA":
            path1 = "/home/528lab/xdh/data/LRN4HCAR/model/PWA-apiTag.pkl"
            f1 = open(path1, 'rb')
            apiTag = pickle.load(f1)

            path1 = "/home/528lab/xdh/data/LRN4HCAR/model/PWA-CC-SC-Tag-123order.pkl"
            f1 = open(path1, 'rb')
            CC_SC_Tag = pickle.load(f1)

            path1 = "/home/528lab/xdh/data/LRN4HCAR/model/PWA-long_tail_api.pkl"
            f1 = open(path1, 'rb')
            long_tail_api = pickle.load(f1)

            ll = 1000

        elif self.dataset == "HGA":
            path1 = "/home/528lab/xdh/data/LRN4HCAR/model/HGA-apiTag.pkl"
            f1 = open(path1, 'rb')
            apiTag = pickle.load(f1)

            path1 = "/home/528lab/xdh/data/LRN4HCAR/model/HGA-CC-SC-Tag-123order.pkl"
            f1 = open(path1, 'rb')
            CC_SC_Tag = pickle.load(f1)

            path1 = "/home/528lab/xdh/data/LRN4HCAR/model/HGA-long_tail_api.pkl"
            f1 = open(path1, 'rb')
            long_tail_api = pickle.load(f1)

            ll = 10000

        for i in range(num_rows):
            row_values = query[i]
            values = list(set(row_values[row_values < ll].tolist()))
            query_list.append(values)
        ans_list = []
        for i in range(num_rows):
            row_values = answer[i]
            cur_ans = torch.nonzero(row_values).squeeze().tolist()
            if isinstance(cur_ans,int):
                cur_ans = [cur_ans]
            ans_list.append(cur_ans)
        pred_list = []
        order = order.tolist()
        for i in range(num_rows):
            row_values = order[i]
            remaining_values = []
            for v in row_values:
                if v not in query_list[i]:
                    remaining_values.append(v)

            pred_list.append(remaining_values)


        for _metric in self.metric:
            if _metric.startswith("mrr@"):
                threshold = int(_metric[4:])
                query_score_list = []
                for i in range(0, len(query_list)):
                    cur_pred = pred_list[i][:threshold]
                    cur_ans = ans_list[i]
                    cur = 1 / 1000000000
                    ii = 1
                    for v in cur_pred:
                        if v in cur_ans:
                            cur = 1 / ii
                            break
                        ii += 1
                    ratio = cur
                    query_score_list.append(ratio)
                query_score = torch.tensor(query_score_list)
                query_score = query_score.to('cuda')
                type_score = scatter_mean(query_score, type, dim_size=len(self.id2type))
            elif _metric.startswith("hits@"):
                threshold = int(_metric[5:])
                query_score_list = []
                for i in range(0, len(query_list)):
                    cur_pred = pred_list[i][:threshold]
                    cur_ans = ans_list[i]
                    cur = 0.0
                    for v in cur_ans:
                        if v in cur_pred:
                            cur = 1.0
                            break
                    ratio = cur
                    query_score_list.append(ratio)
                query_score = torch.tensor(query_score_list)
                query_score = query_score.to('cuda')
                type_score = scatter_mean(query_score, type, dim_size=len(self.id2type))
            elif _metric.startswith("sd@"):
                threshold = int(_metric[3:])
                query_score_list = []
                for i in range(0,len(query_list)):
                    cur_query = query_list[i]
                    cur_pred = pred_list[i][:threshold]
                    common_count = 0
                    for n1 in cur_query:
                        for n2 in cur_pred:
                            if apiTag[n1] == apiTag[n2]:
                                common_count = common_count + 1
                    ratio = common_count / (len(cur_query) * len(cur_pred))
                    query_score_list.append(ratio)
                query_score = torch.tensor(query_score_list)
                query_score = query_score.to('cuda')
                type_score = scatter_mean(query_score, type, dim_size=len(self.id2type))
            elif _metric.startswith("fhit@"):
                threshold = int(_metric[5:])
                query_score_list = []
                for i in range(0,len(query_list)):
                    cur_query = query_list[i]
                    cur_pred = pred_list[i][:threshold]
                    if len(cur_query) == 1:
                        key = cur_query[0]
                    elif len(cur_query) == 2:
                        key = max(cur_query[0], cur_query[1]) * ll + min(cur_query[0], cur_query[1])
                    else:
                        a1 = max(cur_query[0], max(cur_query[1], cur_query[2]))
                        a3 = min(cur_query[0], min(cur_query[1], cur_query[2]))
                        a2 = cur_query[0] + cur_query[1] + cur_query[2] - a1 - a3
                        key = a1 * ll* ll + a2 * ll + a3
                    tag_set = CC_SC_Tag[key]
                    cur = 0
                    all = threshold
                    for v in cur_pred:
                        if apiTag[v] in tag_set:
                            cur = cur + 1
                    ratio = cur / all
                    query_score_list.append(ratio)
                query_score = torch.tensor(query_score_list)
                query_score = query_score.to('cuda')
                type_score = scatter_mean(query_score, type, dim_size=len(self.id2type))

            elif _metric.startswith("lt@"):
                threshold = int(_metric[3:])
                query_score_list = []
                for i in range(0,len(query_list)):
                    cur_pred = pred_list[i][:threshold]
                    cur = 0.0
                    all = threshold
                    for v in cur_pred:
                        if v in long_tail_api:
                            cur = cur + 1.0
                    ratio = cur / all
                    query_score_list.append(ratio)
                query_score = torch.tensor(query_score_list)
                query_score = query_score.to('cuda')
                type_score = scatter_mean(query_score, type, dim_size=len(self.id2type))
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            is_neg = torch.tensor(["n" in t for t in self.id2type], device=self.device)
            is_epfo = ~is_neg
            name = tasks._get_metric_name(_metric)
            for i, query_type in enumerate(self.id2type):
                metric["[%s] %s" % (query_type, name)] = type_score[i]


        return metric

    def visualize(self, batch):
        query = batch["query"]
        return self.model.visualize(self.fact_graph, self.graph, query)
