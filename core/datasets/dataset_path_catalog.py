import os
from .cityscapes import cityscapesDataSet
from .cityscapes_self_distill import cityscapesSelfDistillDataSet
from .synthia import synthiaDataSet
from .gta5 import GTA5DataSet
from .a_datasets import ADataSet
from .b_datasets import BDataSet
from .c_datasets import CDataSet
from .ss_datasets import SSDataSet
from .t1_datasets import T1DataSet
from .t2_datasets import T2DataSet
from .c_datasets_self_distill import CDataSet1

class DatasetCatalog(object):
    DATASET_DIR = "datasets"
    DATASETS = {
        "a_datasets_train": {
            "data_dir": "a_datasets",
            "data_list": "a_datasets_train_list.txt"
        },
        "b_datasets_train": {
            "data_dir": "b_datasets",
            "data_list": "b_datasets_train_list.txt"
        },
        "c_datasets_train": {
            "data_dir": "c_datasets",
            "data_list": "c_datasets_train_list.txt"
        },
        "c_datasets_val": {
            "data_dir": "c_datasets",
            "data_list": "c_datasets_val_list.txt"
        },
        "b_datasets_val": {
            "data_dir": "b_datasets",
            "data_list": "b_datasets_val_list.txt"
        },
        "a_datasets_val": {
            "data_dir": "a_datasets",
            "data_list": "a_datasets_val_list.txt"
        },
        "d_datasets_train": {
            "data_dir": "d_datasets",
            "data_list": "d_datasets_train_list.txt"
        },
        "d_datasets_val": {
            "data_dir": "d_datasets",
            "data_list": "d_datasets_val_list.txt"
        },
    }

    @staticmethod
    def get(name, mode, num_classes, max_iters=None, transform=None):
        if "b_datasets" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            ddir = '/data/pth/lm-pth/lmdatasets'
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=ddir,
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return BDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        elif "c_datasets" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            ddir = '/data/pth/lm-pth/lmdatasets'
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=ddir,
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            if 'distill' in name:
                args['label_dir'] = os.path.join(data_dir, attrs["label_dir"])
                return CDataSet1(args["root"], args["data_list"], args['label_dir'], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
            return CDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        elif "a_datasets" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            ddir = '/data/pth/lm-pth/lmdatasets'
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=ddir,
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return ADataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        elif "d_datasets" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            ddir = '/data/pth/lm-pth/lmdatasets'
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=ddir,
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return ADataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes, split=mode, transform=transform)
        raise RuntimeError("Dataset not available: {}".format(name))