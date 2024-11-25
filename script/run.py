import os
import sys
import math
import pprint

import torch

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import dataset, gnn, model, task, util


def train_and_validate(cfg, solver):
    if cfg.train.num_epoch == 0:
        return
    step = 2

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.train(**kwargs)
        solver.save("model_epoch_%d.pth" % solver.epoch)
        solver.model.metric = ("mrr@20",  "hits@20",  "sd@20",  "fhit@20","lt@20","mrr@10",  "hits@10",  "sd@10",  "fhit@10", "lt@10")
        solver.evaluate("test")

    return solver


def test(cfg, solver):
    solver.model.metric = ( "hits@3", "hits@5", "hits@10", "hits@20", "hits@30","hits@40", "hits@50",  "mrr@3", "mrr@5", "mrr@10", "mrr@20", "mrr@30", "mrr@40", "mrr@50",
                            "fhit@3", "fhit@5", "fhit@10", "fhit@20", "fhit@30", "fhit@40", "fhit@50",  "sd@3", "sd@5", "sd@10", "sd@20", "sd@30", "sd@40", "sd@50",
                            "lt@3", "lt@5", "lt@10", "lt@20", "lt@30", "lt@40", "lt@50")
    solver.evaluate("test")


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))


    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)
    if args.type == 0:
        train_and_validate(cfg, solver)
    else:
        solver.load(args.checkpoint)
        test(cfg, solver)
