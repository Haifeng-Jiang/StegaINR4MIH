
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  # sovle the multiple copies of libiomp5md.dll problem
import json

import sys
import time
import torch
# from training.training import Trainer

# modify trainning.py for INR
from training.training_INR import TrainerINR

from data.conversion import GridDataConverter, PointCloudDataConverter, ERA5Converter
from data.dataloaders import mnist, celebahq
from data.dataloaders_era5 import era5
from data.dataloaders3d import shapenet_voxels, shapenet_point_clouds
from models.discriminator import PointConvDiscriminator
from models.function_distribution import HyperNetwork, FunctionDistribution
from models.function_representation import FunctionRepresentation, FourierFeatures

if __name__ == '__main__':
    ######################################################
    # config file for training your secret INR model
    # config_path="configs\config_celebahq64INR.json"
    config_path="configs\config_celebahq128INR.json"
    #config_path="configs\config_celebahq512INR.json"
    # config_path="configs\config_celebahq256INR.json"
    # config_path="configs\config_celebahq512INR.json"

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

    # Create a folder to store experiment results
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    directory = "{}_{}".format(timestamp, config["id"])
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save config file in experiment directory
    with open(directory + '/config.json', 'w') as f:
        json.dump(config, f)

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
    elif config["dataset"] == 'celebahq':
        dataloader = celebahq(path_to_data=config["path_to_data"],
                            batch_size=config["training"]["batch_size"],
                            size=config["resolution"])[0]
        input_dim = 2
        output_dim = 3
        data_shape = (3, config["resolution"], config["resolution"])
    elif config["dataset"] == 'shapenet_voxels':
        dataloader = shapenet_voxels(path_to_data=config["path_to_data"],
                                    batch_size=config["training"]["batch_size"],
                                    size=config["resolution"])
        input_dim = 3
        output_dim = 1
        data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
        is_voxel = True
    elif config["dataset"] == 'shapenet_point_clouds':
        dataloader = shapenet_point_clouds(path_to_data=config["path_to_data"],
                                        batch_size=config["training"]["batch_size"])
        input_dim = 3
        output_dim = 1
        data_shape = (1, config["resolution"], config["resolution"], config["resolution"])
        is_point_cloud = True

    elif config["dataset"] == 'era5':
        dataloader = era5(path_to_data=config["path_to_data"],
                        batch_size=config["training"]["batch_size"])
        input_dim = 3
        output_dim = 1
        data_shape = (46, 90)
        is_era5 = True


    # Setup data converter : convert original data to coordinates and features.
    if is_point_cloud:
        data_converter = PointCloudDataConverter(device, data_shape, normalize_features=True)
    elif is_era5:
        data_converter = ERA5Converter(device, data_shape, normalize_features=True)
    elif is_voxel:
        data_converter = GridDataConverter(device, data_shape, normalize_features=True)
    else:
        data_converter = GridDataConverter(device, data_shape, normalize_features=True)


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
    function_representation_inr = FunctionRepresentation(input_dim, output_dim,
                                                    config["generator"]["layer_sizes"],
                                                    encoding, non_linearity,
                                                    final_non_linearity).to(device)

    ############################################
    # Add for INR:
    print("\nFunction distribution")
    print(function_representation_inr)
    print("Number of parameters: {}".format(count_parameters(function_representation_inr)))

    ###############################################

    # modify Setup: Trainer----->TrainerINR
    trainer = TrainerINR(device, function_representation_inr, data_converter,
                    lr=config["training"]["lr"], lr_disc=config["training"]["lr_disc"],
                    r1_weight=config["training"]["r1_weight"],
                    max_num_points=config["training"]["max_num_points"],
                    print_freq=config["training"]["print_freq"], save_dir=directory,
                    model_save_freq=config["training"]["model_save_freq"],
                    is_voxel=is_voxel, is_point_cloud=is_point_cloud,
                    is_era5=is_era5)
    trainer.train(dataloader, config["training"]["epochs"])
