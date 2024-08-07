# -*- coding:utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  # sovle the multiple copies of libiomp5md.dll problem

import imageio
import json
import torch
from viz.plots import plot_point_cloud_batch, plot_voxels_batch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
# %matplotlib inline
import argparse
from data.conversion import GridDataConverter, PointCloudDataConverter, ERA5Converter
from models.function_representation import load_function_representation, FourierFeatures, FunctionRepresentation
from training.training_multi_task_INR import insert_adapter
from utils import store, init_weights, prune_and_store, generate_key, random_prune_and_store

device = torch.device('cpu')

if __name__ == '__main__':
    ######################################################
    # config file for training your secret INR model
    # config_path="configs\config_celebahq64INR.json"
    #config_path="configs\config_celebahq128INR.json"
    # config_path="configs\config_celebahq256INR.json"
    # config_path="configs\config_celebahq512INR.json"
    config_path="configs\config_celebahq128INR_multi_task_prune.json"
    #config_path="configs\config_celebahq128INR_17_task_prune.json"
    # config_path="configs\config_celebahq128INR_multi_task_init_prune.json"
    # config_path="configs\config_celebahq128INR_multi_task_random.json"

    # config_path="configs\config_era5INR.json"
    # config_path="configs\config_shapenet_point_clouds_INR.json"
    # config_path="configs\config_shapenet_voxels_INR.json"

    model_file = "2024-06-21_23-40-13_celebahq128_experiment_INR_prune\inr_model_50000.pt"
    sparse_masks = "2024-06-21_23-40-13_celebahq128_experiment_INR_prune\sparse_masks.pkl"
    key_random_seed = None # 1, 2, None
    
    datatype=["image","voxel","point_cloud","era5"]
    save_dir="."
    select_datatype =  datatype[0]
    ###########################################################################################################
    config_path = config_path.replace('\\', '/')

    #config.json should be consistent with config.json in experiment folder
    with open(config_path) as f:
        config = json.load(f)
    config["path_to_data"] = config["path_to_data"].replace('\\', '/')

    sr = config["sr"]
    startup_config = config["startup"]
    generate_key_method = startup_config.get("generate_key_method")
    init_method = startup_config.get("init_method")
    model_random_seed = startup_config.get("model_random_seed")

    flag = generate_key_method
    if flag is not None:
        assert flag in ['prune', 'random', 'init_prune']
        if flag != 'prune':
            assert sr is not None
        elif flag == 'init_prune':
            assert init_method == 'xavier'

    if key_random_seed is None:
        flag = None
    
    if flag is None:
        key_random_seed = None

    print("Model Config:")
    [print(k+":", v) for k, v in config.items() if not isinstance(v, dict)]
    print("\n")
    print("Sampling Config:")
    [print(k+":", v) for k, v in config['sampling'].items() if not isinstance(v, dict)]
    print("\n")

    # Create appropriate data converter based on config
    if config["dataset"] == 'mnist':
        data_shape = (1, config["resolution"], config["resolution"])
        data_converter = GridDataConverter(device, data_shape,
                                        normalize_features=True)
    elif config["dataset"] == 'celebahq':
        data_shape = (3, config["resolution"], config["resolution"])
        data_converter = GridDataConverter(device, data_shape,
                                        normalize_features=True)
    elif config["dataset"] == 'shapenet_voxels':
        data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
        data_converter = GridDataConverter(device, data_shape,
                                        normalize_features=True)
    elif config["dataset"] == 'shapenet_point_clouds':
        data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
        data_converter = PointCloudDataConverter(device, data_shape,
                                                normalize_features=True)
    elif config["dataset"] == "era5":
        data_shape = (46, 90)
        data_converter = ERA5Converter(device, data_shape,
                                                normalize_features=True)
    print("Load model:", model_file)

    function_representation_inr = load_function_representation(device, model_file) # one model at least
    if flag == 'prune':
        sparse_masks = torch.load(sparse_masks)
        print(sparse_masks)
        sparse_masks = [mask.to(device) for mask in sparse_masks]
        # 剪枝
        fixed = store(function_representation_inr, sparse_masks)
    elif flag == 'init_prune':
        if config["dataset"] == 'mnist':
            input_dim = 2
            output_dim = 1
            data_shape = (1, config["resolution"], config["resolution"])
        elif config["dataset"] == 'celebahq':
            input_dim = 2
            output_dim = 3
            data_shape = (3, config["resolution"], config["resolution"])
        elif config["dataset"] == 'shapenet_voxels':
            input_dim = 3
            output_dim = 1
            data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
        elif config["dataset"] == 'shapenet_point_clouds':
            input_dim = 3
            output_dim = 1
            data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
        elif config["dataset"] == 'era5':
            input_dim = 3
            output_dim = 1
            data_shape = (46, 90)
        print(sparse_masks)
        # Setup encoding for function distribution
        num_frequencies = config["generator"]["encoding"]["num_frequencies"]
        std_dev = config["generator"]["encoding"]["std_dev"]
        if num_frequencies:
            frequency_matrix = torch.normal(mean=torch.zeros(num_frequencies, input_dim),
                                            std=std_dev).to(device)
            encoding = FourierFeatures(frequency_matrix)
        else:
            encoding = torch.nn.Identity()

        # Setup generator models

        non_linearity = torch.nn.LeakyReLU(0.1)
        final_non_linearity = torch.nn.Tanh()  # output in (-1, 1)

        assert model_random_seed is not None
        assert init_method is not None
        init_function_representation_inr = FunctionRepresentation(input_dim, output_dim,
                                                config["generator"]["layer_sizes"],
                                                encoding, non_linearity,
                                                final_non_linearity).to(device)

        init_weights(init_function_representation_inr, model_random_seed, init_method)
        # 固定最大的5%，且替换key，剩下的共享权重
        sparse_masks, fixed = prune_and_store(init_function_representation_inr, sr=sr)
    elif flag == 'random':
        sparse_masks, fixed = random_prune_and_store(function_representation_inr, sr=sr, random_seed=model_random_seed)
    else:
        pass

    if key_random_seed is not None:
        key = generate_key(function_representation_inr, sparse_masks, random_seed=key_random_seed)

        insert_adapter(function_representation_inr, key)
        function_representation_inr.save_data_samples_from_representation("sample_from_inr_with_key.png", data_converter,
                                                                          select_datatype, save_dir)
        print("Sample from secret inr model complete ! " )
        print("filenane: sample_from_inr_with_key.png" )
        for name, param in function_representation_inr.named_parameters():
            print(name, param)
    else:
        function_representation_inr.save_data_samples_from_representation("sample_from_inr.png", data_converter,
                                                                          select_datatype, save_dir)


        print("Sample from secret inr model complete ! " )
        print("filenane: sample_from_inr.png" )
        for name, param in function_representation_inr.named_parameters():
            print(name, param)

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

def draw(function_representation_inr):
    model = function_representation_inr
    for k, m in list(model.named_modules()):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            weight = m.weight.data
            print(k)
            x, y = np.meshgrid(np.arange(weight.shape[1]), np.arange(weight.shape[0]))
            plt.pcolormesh(x, y, weight)
            plt.colorbar()
            plt.title(f'{k}_weight_mesh {weight.shape}')
            plt.savefig(f'{k}_weight_mesh.png')
            plt.close()
if __name__ == '__main__':
    draw(function_representation_inr)