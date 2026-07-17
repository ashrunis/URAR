# -*- coding:utf-8 -*-

import argparse
import os
import sys

from train_ptv3_native_ood_fr_ddp import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--config_path", default="../config/semantickitti_ood_randla.yaml")
    arguments = parser.parse_args()
    if int(os.environ.get("RANK", 0)) == 0:
        print(" ".join(sys.argv))
        print(arguments)
    main(arguments)
