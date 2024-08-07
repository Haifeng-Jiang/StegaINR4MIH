
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  # sovle the multiple copies of libiomp5md.dll problem
import json

import sys
import time
import torch
# from training.training import Trainer

# modify trainning.py for INR
from training.training_multi_task_INR import TrainerINR

from data.conversion import GridDataConverter, PointCloudDataConverter, ERA5Converter
from data.dataloaders import mnist, celebahq
from data.dataloaders_era5 import era5
from data.dataloaders3d import shapenet_voxels, shapenet_point_clouds
from models.discriminator import PointConvDiscriminator
from models.function_distribution import HyperNetwork, FunctionDistribution
from models.function_representation import FunctionRepresentation, FourierFeatures
from models.function_representation import load_function_representation
from utils import prune_and_store, generate_key, init_weights, random_prune_and_store, prune, store, reverse_mask


if __name__ == '__main__':
    ######################################################
    # config file for training your secret INR model
    #config_path="configs\config_celebahq128INR512_multi_task_prune.json"
    #config_path="configs\config_celebahq128INR_17_task_prune.json"
    #config_path="configs\config_celebahq128INR_multi_task_init_prune.json"
    #config_path="configs\config_celebahq128INR_multi_task_random.json"
    config_path="configs\config_celebahq128INR_multi_task_prune.json"
    # config_path="configs\config_celebahq512INR_multi_task_prune.json"

    # config_path="configs\config_era5INR.json"
    # config_path="configs\config_shapenet_point_clouds_INR.json"
    # config_path="configs\config_shapenet_voxels_INR.json"
    ####################################################################################################################################
    config_path = config_path.replace('\\', '/')
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # os.environ['CUDA_VISIBLE_DEVICES']='1' #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get config file from command line arguments
    # if len(sys.argv) != 2:
    #     raise(RuntimeError("Wrong arguments, use python main.py <config_path>"))
    # config_path = sys.argv[1]

    # Open config file
    with open(config_path) as f:
        config = json.load(f)
    config["path_to_data"] = config["path_to_data"].replace('\\', '/')
    if config["path_to_data"] == "":
        raise(RuntimeError("Path to data not specified. Modify path_to_data attribute in config to point to data."))
    sr = config["sr"]

    startup_config = config["startup"]
    generate_key_method = startup_config.get("generate_key_method")
    model_file = startup_config.get("model_file")
    init_method = startup_config.get("init_method")
    merge_secret_image = bool(startup_config.get("merge_secret_image", False))

    model_random_seed = startup_config.get("model_random_seed")
    key_random_seeds = startup_config["key_random_seeds"]

    print("Model Config:")
    [print(k, ":", v) for k, v in config.items() if not isinstance(v, dict)]
    print("\n")
    print("Startup Config:")
    [print(k, ":", v) for k, v in startup_config.items() if not isinstance(v, dict)]
    print("\n")

    flag = generate_key_method

    # Create a folder to store experiment results
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    directory = "{}_{}_{}".format(timestamp, config["id"], flag)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save config file in experiment directory
    with open(directory + '/config.json', 'w') as f:
        json.dump(config, f)

    # secret resolution
    secret_resolution = config.get("secret_resolution", config["resolution"])

    # Setup dataloader
    is_voxel = False
    is_point_cloud = False
    is_era5 = False
    if config["dataset"] == 'mnist':
        dataloader = mnist(path_to_data=config["path_to_data"],
                        batch_size=config["training"]["batch_size"],
                        size=config["resolution"],
                        train=True)
        input_dim = 2
        output_dim = 1
        data_shape = (1, config["resolution"], config["resolution"])
        secret_data_shape = (1, secret_resolution, secret_resolution)
    elif config["dataset"] == 'celebahq':
        dataloader, secret_dataloader = celebahq(path_to_data=config["path_to_data"],
                                                batch_size=config["training"]["batch_size"],
                                                size=config["resolution"], secret_size=secret_resolution)
        input_dim = 2
        output_dim = 3
        data_shape = (3, config["resolution"], config["resolution"])
        secret_data_shape = (3, secret_resolution, secret_resolution)
    elif config["dataset"] == 'shapenet_voxels':
        dataloader = shapenet_voxels(path_to_data=config["path_to_data"],
                                    batch_size=config["training"]["batch_size"],
                                    size=config["resolution"])
        input_dim = 3
        output_dim = 1
        data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
        secret_data_shape = (1, secret_resolution, secret_resolution, secret_resolution)
        is_voxel = True
    elif config["dataset"] == 'shapenet_point_clouds':
        dataloader = shapenet_point_clouds(path_to_data=config["path_to_data"],
                                        batch_size=config["training"]["batch_size"])
        input_dim = 3
        output_dim = 1
        data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
        secret_data_shape = (1, secret_resolution, secret_resolution, secret_resolution)
        is_point_cloud = True

    elif config["dataset"] == 'era5':
        dataloader = era5(path_to_data=config["path_to_data"],
                        batch_size=config["training"]["batch_size"])
        input_dim = 3
        output_dim = 1
        data_shape = (46, 90)
        secret_data_shape = (46, 90)
        is_era5 = True


    # Setup data converter : convert original data to coordinates and features.
    if is_point_cloud:
        data_converter = PointCloudDataConverter(device, data_shape, normalize_features=True)
        secret_data_converter = PointCloudDataConverter(device, secret_data_shape, normalize_features=True)
    elif is_era5:
        data_converter = ERA5Converter(device, data_shape, normalize_features=True)
        secret_data_converter = ERA5Converter(device, secret_data_shape, normalize_features=True)
    elif is_voxel:
        data_converter = GridDataConverter(device, data_shape, normalize_features=True)
        secret_data_converter = GridDataConverter(device, secret_data_shape, normalize_features=True)
    else:
        data_converter = GridDataConverter(device, data_shape, normalize_features=True)
        secret_data_converter = GridDataConverter(device, secret_data_shape, normalize_features=True)


    # Setup encoding for function distribution
    num_frequencies = config["generator"]["encoding"]["num_frequencies"]
    std_dev = config["generator"]["encoding"]["std_dev"]
    if num_frequencies:
        frequency_matrix = torch.normal(mean=torch.zeros(num_frequencies, input_dim),
                                        std=std_dev).to(device)
        torch.save(frequency_matrix, timestamp +'_frequency_matrix_for_inr_secret.pt')
        encoding = FourierFeatures(frequency_matrix)
    else:
        encoding = torch.nn.Identity()

    # Setup generator models

    non_linearity = torch.nn.LeakyReLU(0.1)
    final_non_linearity = torch.nn.Tanh()  # output in (-1, 1)

    # non_linearity = torch.nn.Sigmoid()
    # non_linearity = torch.nn.ReLU()
    #Add for INR: define a FunctionRepresentation class for representation
    
    n =  len(dataloader.dataset.samples) + len(secret_dataloader.dataset.samples)

    if model_file is not None:
        # 有预训练
        # 采用prune
        assert flag == 'prune'
        function_representation_inr = load_function_representation('cpu', model_file) # one model at least
        # 剪枝
        sparse_masks, fixed = prune_and_store(function_representation_inr, sr=sr)
        #loss_weight = [0.1]  + [0.9] * len(secret_dataloader.dataset.samples) # [1 /n ] * n
        loss_weight = [1 /n ] * n
        # 保存mask
        torch.save(sparse_masks, os.path.join(directory,'sparse_masks.pkl'))

    else:
        assert model_random_seed is not None

        if flag == 'init_prune':
            function_representation_inr = FunctionRepresentation(input_dim, output_dim,
                                                    config["generator"]["layer_sizes"],
                                                    encoding, non_linearity,
                                                    final_non_linearity)

            assert init_method is not None
            init_weights(function_representation_inr, model_random_seed, init_method)
            # 固定最大的5%，且替换key，剩下的共享权重
            sparse_masks, fixed = prune_and_store(function_representation_inr, sr=sr)

        elif flag == 'random':
            function_representation_inr = FunctionRepresentation(input_dim, output_dim,
                                                    config["generator"]["layer_sizes"],
                                                    encoding, non_linearity,
                                                    final_non_linearity)

            sparse_masks, fixed = random_prune_and_store(function_representation_inr, sr=sr, random_seed=model_random_seed)
        loss_weight = [1 /n ] * n

    keys = [generate_key(function_representation_inr, sparse_masks, random_seed=seed) for seed in key_random_seeds]
    ############################################
    # Add for INR:
    function_representation_inr.to(device)

    print("\nFunction distribution")
    print(function_representation_inr)
    print("Number of parameters: {}".format(count_parameters(function_representation_inr)))

    ###############################################
    # modify Setup: Trainer----->TrainerINR
    trainer = TrainerINR(device, function_representation_inr, data_converter, secret_data_converter,
                    lr=config["training"]["lr"], lr_disc=config["training"]["lr_disc"],
                    r1_weight=config["training"]["r1_weight"],
                    max_num_points=config["training"]["max_num_points"],
                    print_freq=config["training"]["print_freq"], save_dir=directory,
                    model_save_freq=config["training"]["model_save_freq"],
                    is_voxel=is_voxel, is_point_cloud=is_point_cloud,
                    is_era5=is_era5,
                    keys=keys,
                    fixed=fixed,
                    loss_weight =loss_weight,
                    merge_secret_image=merge_secret_image)
    trainer.train(dataloader, secret_dataloader, config["training"]["epochs"])
