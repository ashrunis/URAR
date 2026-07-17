# -*- coding:utf-8 -*-

import argparse
import sys

from val_ptv3_native_ood import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--config_path", default="../config/semantickitti_ood_randla.yaml")
    parser.add_argument("--save_folder", default="../exp/semantic_kitti/backbone/randla")
    arguments = parser.parse_args()
    print(" ".join(sys.argv))
    print(arguments)
    main(arguments)
