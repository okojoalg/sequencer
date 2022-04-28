#!/usr/bin/env python3

#  Copyright (c) 2022. Yuki Tatsunami
#  Licensed under the Apache License, Version 2.0 (the "License");

import argparse
import os
import glob
import logging

import numpy as np
from matplotlib import pyplot as plt
from timm.utils import setup_default_logging

from erf.scaler import MinMaxScaler

parser = argparse.ArgumentParser(description='PyTorch ImageNet ERF Visualizer')

parser.add_argument('--result-npy-dir', default='./erf_results/224/npy', type=str,
                    help='path to save npys of ERF')
parser.add_argument('--result-png-dir', default='./erf_results/224/img', type=str,
                    help='path to save plotted images (png) of ERF')
parser.add_argument('--result-pdf-dir', default='./erf_results/224/pdf', type=str,
                    help='path to save plotted images (pdf) of ERF')


def main():
    setup_default_logging()
    args = parser.parse_args()

    os.makedirs(args.result_png_dir, exist_ok=True)
    os.makedirs(args.result_pdf_dir, exist_ok=True)

    npy_paths = glob.glob(os.path.join(args.result_npy_dir, "*.npy"))
    npys = [np.load(p) for p in npy_paths]
    scores = np.stack(npys, axis=0)
    scaler = MinMaxScaler()
    scores = scaler(scores)
    for p, s in zip(npy_paths, scores):
        file_base = os.path.basename(p).rsplit('.', 1)[0]
        png_path = os.path.join(args.result_png_dir, f'{file_base}.png')
        pdf_path = os.path.join(args.result_pdf_dir, f'{file_base}.pdf')

        plt.imsave(png_path, s, cmap='pink', format="png")
        plt.imsave(pdf_path, s, cmap='pink', format="pdf")

if __name__ == '__main__':
    main()
