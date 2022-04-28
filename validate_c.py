#!/usr/bin/env python3

#  Copyright (c) 2022. Yuki Tatsunami
#  Licensed under the Apache License, Version 2.0 (the "License");

import argparse
import os
import csv
import glob
import logging

import numpy as np
import torch
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

import torchvision
from timm.data.loader import PrefetchLoader, create_loader
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, resolve_data_config, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils import natural_key, setup_default_logging, set_jit_legacy

import models

has_apex = False
try:
    from apex import amp

    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')

distortions = dict(
    gaussian_noise=0.886428,
    shot_noise=0.894468,
    impulse_noise=0.922640,
    defocus_blur=0.819880,
    glass_blur=0.826268,
    motion_blur=0.785948,
    zoom_blur=0.798360,
    snow=0.866816,
    frost=0.826572,
    fog=0.819324,
    brightness=0.564592,
    contrast=0.853204,
    elastic_transform=0.646056,
    pixelate=0.717840,
    jpeg_compression=0.606500,
    # speckle_noise=0.845388,
    # gaussian_blur=0.787108,
    # spatter=0.717512,
    # saturate=0.658248,
)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true',
                    help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True, verbose=True)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    results = OrderedDict()
    un_ces = []
    ces = []
    for distortion_name, distortion_alex_value in distortions.items():
        errs = []
        for severity in range(1, 6):
            correct = 0
            dataset = create_dataset(
                root=os.path.join(args.data, distortion_name, str(severity)), name=args.dataset, split=args.split,
                download=args.dataset_download, load_bytes=args.tf_preprocessing, class_map=args.class_map)

            if args.valid_labels:
                with open(args.valid_labels, 'r') as f:
                    valid_labels = {int(line.rstrip()) for line in f}
                    valid_labels = [i in valid_labels for i in range(args.num_classes)]
            else:
                valid_labels = None

            loader = create_loader(
                dataset,
                input_size=data_config['input_size'],
                batch_size=args.batch_size,
                use_prefetcher=args.prefetcher,
                interpolation=data_config['interpolation'],
                mean=data_config['mean'],
                std=data_config['std'],
                num_workers=args.workers,
                crop_pct=1.,
                pin_memory=args.pin_mem,
                tf_preprocessing=args.tf_preprocessing)

            model.eval()
            with torch.no_grad():
                # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
                input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).cuda()
                if args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)
                model(input)
                for batch_idx, (input, target) in enumerate(loader):
                    if args.no_prefetcher:
                        target = target.cuda()
                        input = input.cuda()
                    if args.channels_last:
                        input = input.contiguous(memory_format=torch.channels_last)

                    # compute output
                    with amp_autocast():
                        output = model(input)

                    if valid_labels is not None:
                        output = output[:, valid_labels]

                    pred = output.data.max(1)[1]
                    correct += pred.eq(target).sum().cpu().detach().numpy()

            errs.append(1 - 1. * correct / len(dataset))

        un_ce = np.mean(errs)
        ce = un_ce / distortion_alex_value
        results[distortion_name] = round(ce.item(), 4)
        ces.append(ce)
        un_ces.append(un_ce)
        _logger.info('Distortion: {:20s} | CE un-normalized (%): {:.3f} | CE (%): {:.3f}'.format(distortion_name, 100 * un_ce, 100 * ce))

    mce = 100 * np.mean(ces)
    un_mce = 100 * np.mean(un_ces)
    results["mCE_un_normalized"] = un_mce
    results["mCE"] = mce
    results["param_count"] = round(param_count / 1e6, 2)
    results["img_size"] = data_config['input_size'][-1]
    _logger.info('mCE un-normalized (%): {:.3f} | mCE (%): {:.3f}'.format(un_mce, mce))

    return results


def main():
    setup_default_logging()
    args = parser.parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k', '*_dino'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        results_file = args.results_file or './results-all.csv'
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= args.num_gpu:
                    torch.cuda.empty_cache()
                    try:
                        args.batch_size = batch_size
                        print('Validating with batch size: %d' % args.batch_size)
                        r = validate(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print("Validation failed with no ability to reduce batch size. Exiting.")
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                result.update(r)
                if args.checkpoint:
                    result['checkpoint'] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        validate(args)


def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == '__main__':
    main()
