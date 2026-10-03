#!/usr/bin/python3

import argparse
import logging
import os
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def make_parser(params_help, pooling=False, recurrent=False):
    parser = argparse.ArgumentParser(description='PyTorch model generator')
    parser.add_argument('params', type=int, nargs='+', help=params_help)
    if recurrent:
        parser.add_argument('--lstm', action='store_true', default=False,
                            help='For using LSTM layer')
        parser.add_argument('--gru', action='store_true', default=False,
                            help='For using GRU layer')
    else:
        parser.add_argument('--bn', action='store_true', default=False,
                            help='For using batch norm layer')
    if pooling:
        parser.add_argument('--maxpool', action='store_true', default=False,
                            help='For using max pool layer')
        parser.add_argument('--avgpool', action='store_true', default=False,
                            help='For using average pool layer')
    parser.add_argument('--v', action='store_true', default=False,
                        help='For verbose mode')
    return parser


def model_name(base, bsize, use_bn=False, use_maxpool=False, use_avgpool=False):
    name = base
    if use_bn:
        name += "_BN"
    if use_maxpool:
        name += "_MAXP"
    if use_avgpool:
        name += "_AVGP"
    return name + "_B" + str(bsize)


def export_onnx(model, xinput, name, use_bn=False, dynamo=None):
    if dynamo is None:
        dynamo = (sys.version_info.major, sys.version_info.minor) != (3, 14)
        if use_bn:
            dynamo = False

    onnx_file = name + ".onnx"
    from packaging.version import Version
    try:
        if Version(torch.__version__) >= Version("2.5.0"):
            torch.onnx.export(
                model,
                xinput,
                onnx_file,
                export_params=True,
                dynamo=dynamo,
                external_data=False,
            )
        else:
            torch.onnx.export(model, xinput, onnx_file, export_params=True)
    except Exception as e:
        logging.error("Failed to export ONNX model %s: %s", onnx_file, e)
        sys.exit(1)
    if not os.path.isfile(onnx_file) or os.path.getsize(onnx_file) == 0:
        logging.error("ONNX file %s was not created or is empty", onnx_file)
        sys.exit(1)
    logging.info("Exported %s", onnx_file)


def write_reference_output(model, xinput, name):
    model.eval()
    y = model.forward(xinput)

    print("output data : shape, ", y.shape)
    print(y)

    yvec = y.reshape([y.nelement()])
    with open(name + ".out", "w") as f:
        for i in range(0, y.nelement()):
            f.write(str(float(yvec[i].detach())) + " ")
    logging.info("Wrote %s.out", name)
