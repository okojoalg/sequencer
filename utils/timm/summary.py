#  Copyright (c) 2022. Yuki Tatsunami
#  Licensed under the Apache License, Version 2.0 (the "License");

import csv
from collections import OrderedDict


def update_summary(epoch, train_metrics, eval_metrics, filename, write_header=False, log_wandb=False, log_clearml=False):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    if log_wandb:
        import wandb
        wandb.log(rowd)
    if log_clearml:
        from clearml import Logger
        for k, v in train_metrics.items():
            Logger.current_logger().report_scalar(
                "train", k, iteration=epoch, value=v)
        for k, v in eval_metrics.items():
            Logger.current_logger().report_scalar(
                "eval", k, iteration=epoch, value=v)

    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)
