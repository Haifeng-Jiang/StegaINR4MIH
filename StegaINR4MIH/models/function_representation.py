import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# ADD for INR
from torchvision.utils import save_image
from viz.plots import plot_voxels_batch, plot_point_cloud_batch, plot_point_cloud_batch_INR


class FunctionRepresentation(nn.Module):
    """Function to represent a single datapoint. For example this could be a
    function that takes pixel coordinates as input and returns RGB values, i.e.
    f(x, y) = (r, g, b).

    Args:
        coordinate_dim (int): Dimension of input (coordinates).
        feature_dim (int): Dimension of output (features).
        layer_sizes (tuple of ints): Specifies size of each hidden layer.
        encoding (torch.nn.Module): Encoding layer, usually one of
            Identity or FourierFeatures.
        final_non_linearity (torch.nn.Module): Final non linearity to use.
            Usually nn.Sigmoid() or nn.Tanh().
    """
    def __init__(self, coordinate_dim, feature_dim, layer_sizes, encoding,
                 non_linearity=nn.ReLU(), final_non_linearity=nn.Sigmoid()):
        super(FunctionRepresentation, self).__init__()
        self.coordinate_dim = coordinate_dim
        self.feature_dim = feature_dim
        self.layer_sizes = layer_sizes
        self.encoding = encoding
        self.non_linearity = non_linearity
        self.final_non_linearity = final_non_linearity

        self._init_neural_net()

    def _init_neural_net(self):
        """
        """
        # First layer transforms coordinates into a positional encoding
        # Check output dimension of positional encoding
        if isinstance(self.encoding, nn.Identity):
            prev_num_units = self.coordinate_dim  # No encoding, so same output dimension
        else:
            prev_num_units = self.encoding.feature_dim
        # Build MLP layers
        forward_layers = []
        for num_units in self.layer_sizes:
            forward_layers.append(nn.Linear(prev_num_units, num_units))
            forward_layers.append(self.non_linearity)
            prev_num_units = num_units
        forward_layers.append(nn.Linear(prev_num_units, self.feature_dim))
        forward_layers.append(self.final_non_linearity)
        self.forward_layers = nn.Sequential(*forward_layers)

    def forward(self, coordinates):
        """Forward pass. Given a set of coordinates, returns feature at every
        coordinate.

        Args:
            coordinates (torch.Tensor): Shape (batch_size, coordinate_dim)
        """
        encoded = self.encoding(coordinates)
        for l in self.forward_layers:
            x = l(encoded)
            if x.shape == encoded.shape and isinstance(l, nn.Linear):
                encoded = encoded + x
            else:
                encoded = x
        return encoded # self.forward_layers(encoded)

    def get_weight_shapes(self):
        """Returns lists of shapes of weights and biases in the network."""
        weight_shapes = []
        bias_shapes = []
        for param in self.forward_layers.parameters():
            if len(param.shape) == 1:
                bias_shapes.append(param.shape)
            if len(param.shape) == 2:
                weight_shapes.append(param.shape)
        return weight_shapes, bias_shapes

    def get_weights_and_biases(self):
        """Returns list of weights and biases in the network."""
        weights = []
        biases = []
        for param in self.forward_layers.parameters():
            if len(param.shape) == 1:
                biases.append(param)
            if len(param.shape) == 2:
                weights.append(param)
        return weights, biases

    def set_weights_and_biases(self, weights, biases):
        """Sets weights and biases in the network.

        Args:
            weights (list of torch.Tensor):
            biases (list of torch.Tensor):

        Notes:
            The inputs to this function should have the same form as the outputs
            of self.get_weights_and_biases.
        """
        weight_idx = 0
        bias_idx = 0
        with torch.no_grad():
            for param in self.forward_layers.parameters():
                if len(param.shape) == 1:
                    param.copy_(biases[bias_idx])
                    bias_idx += 1
                if len(param.shape) == 2:
                    param.copy_(weights[weight_idx])
                    weight_idx += 1

    def duplicate(self):
        """Returns a FunctionRepresentation instance with random weights."""
        # Extract device
        device = next(self.parameters()).device
        # Create new function representation and put it on same device
        return FunctionRepresentation(self.coordinate_dim, self.feature_dim,
                                      self.layer_sizes, self.encoding,
                                      self.non_linearity,
                                      self.final_non_linearity).to(device)

    def sample_grid(self, data_converter, resolution=None):
        """Returns function values evaluated on grid.

        Args:
            data_converter (data.conversion.DataConverter):
            resolution (tuple of ints): Resolution of grid on which to evaluate
                features. If None uses default resolution.
        """
        # Predict features at every coordinate in a grid
        if resolution is None:
            coordinates = data_converter.coordinates
        else:
            coordinates = data_converter.superresolve_coordinates(resolution)
        features = self(coordinates)

        positive_elements = torch.gt(features, 0)  # features in [-1,1],count

        # ʹ�� torch.sum ����С�����Ԫ�صĸ���
        count = torch.sum(positive_elements == True)
        # print("Number of positive elements:", count)

        # print(features)# [-1,1]
        # Convert features into appropriate data format (e.g. images)
        return data_converter.to_data(coordinates, features, resolution)

    # Add for INR: forword
    def sample_features(self, data_converter, resolution=None):
        """Returns function values evaluated on grid.

        Args:
            data_converter (data.conversion.DataConverter):
            resolution (tuple of ints): Resolution of grid on which to evaluate
                features. If None uses default resolution.
        """
        # Predict features at every coordinate in a grid
        if resolution is None:
            coordinates = data_converter.coordinates
        else:
            coordinates = data_converter.superresolve_coordinates(resolution)
        features = self(coordinates)

        positive_elements = torch.gt(features, 0)# features in [-1,1],count

        # ʹ�� torch.sum ����С�����Ԫ�صĸ���
        count = torch.sum(positive_elements)
        # print("Number of nonzero elements:",count)
        # Convert features into appropriate data format (e.g. images)

        return features


    def stateless_forward(self, coordinates, weights, biases):
        """Computes forward pass of function representation given a set of
        weights and biases without using the state of the PyTorch module.

        Args:
            coordinates (torch.Tensor): Tensor of shape (num_points, coordinate_dim).
            weights (list of torch.Tensor): List of tensors containing weights
                of linear layers of neural network.
            biases (list of torch.Tensor): List of tensors containing biases of
                linear layers of neural network.

        Notes:
            This is useful for computing forward pass for a specific function
            representation (i.e. for a given set of weights and biases). However,
            it might be easiest to just change the weights of the network directly
            and then perform forward pass.
            Doing the current way is definitely more error prone because we have
            to mimic the forward pass, instead of just directly using it.

        Return:
            Returns a tensor of shape (num_points, feature_dim)
        """
        # Positional encoding is first layer of function representation
        # model, so apply this transformation to coordinates
        hidden = self.encoding(coordinates)
        # Apply linear layers and non linearities
        for i in range(len(weights)):
            hidden = F.linear(hidden, weights[i], biases[i])
            if i == len(weights) - 1:
                hidden = self.final_non_linearity(hidden)
            else:
                hidden = self.non_linearity(hidden)
        return hidden

    def batch_stateless_forward(self, coordinates, weights, biases):
        """Stateless forward pass for multiple function representations.

        Args:
            coordinates (torch.Tensor): Batch of coordinates of shape
                (batch_size, num_points, coordinate_dim).
            weights (dict of list of torch.Tensor): Batch of list of tensors
                containing weights of linear layers for each neural network.
            biases (dict of list of torch.Tensor): Batch of list of tensors
                containing biases of linear layers for each neural network.

        Return:
            Returns a tensor of shape (batch_size, num_points, feature_dim).
        """
        features = []
        for i in range(coordinates.shape[0]):
            features.append(
                self.stateless_forward(coordinates[i], weights[i], biases[i]).unsqueeze(0)
            )
        return torch.cat(features, dim=0)


    def batch_stateless_forward_INR(self, coordinates, weights, biases):
        """Stateless forward pass for multiple function representations.

        Args:
            coordinates (torch.Tensor): Batch of coordinates of shape
                (batch_size, num_points, coordinate_dim).
            weights (dict of list of torch.Tensor): Batch of list of tensors
                containing weights of linear layers for each neural network.
            biases (dict of list of torch.Tensor): Batch of list of tensors
                containing biases of linear layers for each neural network.

        Return:
            Returns a tensor of shape (batch_size, num_points, feature_dim).
        """

        features = []
        for i in range(coordinates.shape[0]):
            features.append(
                self.stateless_forward(coordinates[i], weights[i], biases[i]).unsqueeze(0)
            )
        return torch.cat(features, dim=0)


    def _get_config(self):
        return {"coordinate_dim": self.coordinate_dim,
                "feature_dim": self.feature_dim,
                "layer_sizes": self.layer_sizes,
                "encoding": self.encoding,
                "non_linearity": self.non_linearity,
                "final_non_linearity": self.final_non_linearity}

    # Add for INR:save function representation f
    def save_model(self, path):
        """Saves model to given path.

        Args:
            path (string): File extension should be ".pt".
        """
        torch.save({'config': self._get_config(), 'state_dict': self.state_dict()}, path)

    # Add for INR��sample data from representation
    def save_data_samples_from_representation(self, filename, data_converter, datatype, save_dir, resolution = None ):
        """
        """
        with torch.no_grad():

            samples = []
            samples.append(self.sample_grid(data_converter, resolution))

            # Convert list of samples to a batch of tensors
            samples = torch.cat([sample.unsqueeze(0) for sample in samples], dim=0)

        # Save samples as grid
        num_samples_to_save = 1
        if datatype=="voxel" :#self.is_voxel:
            # Voxels lie in [0, 1], so use 0.5 as a threshold
            plot_voxels_batch(samples.detach().cpu() > .5, save_fig=save_dir + "/" + filename,
                              ncols=num_samples_to_save)# // 4)
        elif datatype=="point_cloud" :#self.is_point_cloud:
            # samples = samples.unsqueeze(0)
            plot_point_cloud_batch(samples.detach().cpu(), save_fig=save_dir + "/" + filename,
                              ncols=num_samples_to_save) #//4)
        elif datatype=="era5" :#self.is_era5:
            # ERA5 data has shape (batch_size, 3, num_lats, num_lons) where 3rd
            # dimension of 2nd axis corresponds to temperature, so extract this
            # (will correspond to grayscale image)
            save_image(samples[:, 2:3].detach().cpu(), save_dir + "/" + filename,
                       nrow=num_samples_to_save // 4)
        else:
            save_image(samples.detach().cpu(), save_dir + "/" + filename,
                       nrow=num_samples_to_save // 4)


    def hide_function_by_function(self,function_representation_secret):
        return

    def hide_function_by_model_config(self, model_config_file_dir):
        return

    def set_weights_and_biases_freeze(self, weights_mask, biases_mask):
        return

    def set_weights_and_biases_unfreeze(self, weights_mask, biases_mask):
        return

# Add for INR: global function : load function model
def load_function_representation(device, path):
    """
    """
    all_dicts = torch.load(path, map_location=lambda storage, loc: storage)
    config, state_dict = all_dicts["config"], all_dicts["state_dict"]
    # Initialize function representation
    # config_rep = config["function_representation"]
    config_rep = config#
    encoding = config_rep["encoding"].to(device)
    if hasattr(encoding, 'frequency_matrix'):
        encoding.frequency_matrix = encoding.frequency_matrix.to(device)
    function_representation = FunctionRepresentation(config_rep["coordinate_dim"],#input_dim
                                                     config_rep["feature_dim"], #outputdim
                                                     config_rep["layer_sizes"],
                                                     encoding,
                                                     config_rep["non_linearity"],
                                                     config_rep["final_non_linearity"]).to(device)
    # Initialize hypernetwork
    # config_hyp = config["hypernetwork"]
    # hypernetwork = HyperNetwork(function_representation, config_hyp["latent_dim"],
    #                             config_hyp["layer_sizes"], config_hyp["non_linearity"]).to(device)
    # # Initialize function distribution
    # function_distribution = FunctionDistribution(hypernetwork).to(device)
    # Load weights of function distribution
    function_representation.load_state_dict(state_dict)
    return function_representation

# def load_function_representation_by_config_for_extraction(device, model_path):
#     """
#     """
#     all_dicts = torch.load(model_path, map_location=lambda storage, loc: storage)
#     config, state_dict = all_dicts["config"], all_dicts["state_dict"]
#     # Initialize function representation
#     # config_rep = config["function_representation"]
#     config_rep = config#
#     encoding = config_rep["encoding"].to(device)
#     if hasattr(encoding, 'frequency_matrix'):
#         encoding.frequency_matrix = encoding.frequency_matrix.to(device)
#     function_representation = FunctionRepresentation(config_rep["coordinate_dim"],#input_dim
#                                                      config_rep["feature_dim"], #outputdim
#                                                      config_rep["layer_sizes"],
#                                                      encoding,
#                                                      config_rep["non_linearity"],
#                                                      config_rep["final_non_linearity"]).to(device)
#     # Initialize hypernetwork
#     # config_hyp = config["hypernetwork"]
#     # hypernetwork = HyperNetwork(function_representation, config_hyp["latent_dim"],
#     #                             config_hyp["layer_sizes"], config_hyp["non_linearity"]).to(device)
#     # # Initialize function distribution
#     # function_distribution = FunctionDistribution(hypernetwork).to(device)
#     # Load weights of function distribution
#     function_representation.load_state_dict(state_dict)
#     return function_representation

class FourierFeatures(nn.Module):
    """Random Fourier features.

    Args:
        frequency_matrix (torch.Tensor): Matrix of frequencies to use
            for Fourier features. Shape (num_frequencies, num_coordinates).
            This is referred to as B in the paper.
        learnable_features (bool): If True, fourier features are learnable,
            otherwise they are fixed.
    """
    def __init__(self, frequency_matrix, learnable_features=False):
        super(FourierFeatures, self).__init__()
        if learnable_features:
            self.frequency_matrix = nn.Parameter(frequency_matrix)
        else:
            # Register buffer adds a key to the state dict of the model. This will
            # track the attribute without registering it as a learnable parameter.
            # We require this so frequency matrix will also be moved to GPU when
            # we call .to(device) on the model
            self.register_buffer('frequency_matrix', frequency_matrix)
        self.learnable_features = learnable_features
        self.num_frequencies = frequency_matrix.shape[0]
        self.coordinate_dim = frequency_matrix.shape[1]
        # Factor of 2 since we consider both a sine and cosine encoding
        self.feature_dim = 2 * self.num_frequencies

    def forward(self, coordinates):
        """Creates Fourier features from coordinates.

        Args:
            coordinates (torch.Tensor): Shape (num_points, coordinate_dim)
        """
        # The coordinates variable contains a batch of vectors of dimension
        # coordinate_dim. We want to perform a matrix multiply of each of these
        # vectors with the frequency matrix. I.e. given coordinates of
        # shape (num_points, coordinate_dim) we perform a matrix multiply by
        # the transposed frequency matrix of shape (coordinate_dim, num_frequencies)
        # to obtain an output of shape (num_points, num_frequencies).
        prefeatures = torch.matmul(coordinates, self.frequency_matrix.T)
        # Calculate cosine and sine features
        cos_features = torch.cos(2 * math.pi * prefeatures)
        sin_features = torch.sin(2 * math.pi * prefeatures)
        # Concatenate sine and cosine features
        return torch.cat((cos_features, sin_features), dim=1)        


def create_function_representation(device, config_path):
    # Open config file
    with open(config_path) as f:
        config = json.load(f)
    config["path_to_data"] = config["path_to_data"].replace('\\', '/')
    if config["path_to_data"] == "":
        raise(RuntimeError("Path to data not specified. Modify path_to_data attribute in config to point to data."))

    # Setup dataloader
    if config["dataset"] == 'mnist':
        input_dim = 2
        output_dim = 1
    elif config["dataset"] == 'celebahq':
        input_dim = 2
        output_dim = 3
    elif config["dataset"] == 'shapenet_voxels':
        input_dim = 3
        output_dim = 1
    elif config["dataset"] == 'shapenet_point_clouds':
        input_dim = 3
        output_dim = 1
    elif config["dataset"] == 'era5':
        input_dim = 3
        output_dim = 1

    # Setup encoding for function distribution
    num_frequencies = config["generator"]["encoding"]["num_frequencies"]
    std_dev = config["generator"]["encoding"]["std_dev"]
    if num_frequencies:
        frequency_matrix = torch.normal(mean=torch.zeros(num_frequencies, input_dim),
                                        std=std_dev).to(device)
        encoding = FourierFeatures(frequency_matrix)
    else:
        encoding = torch.nn.Identity()


    non_linearity = torch.nn.LeakyReLU(0.1)
    final_non_linearity = torch.nn.Tanh()  # output in (-1, 1)

    function_representation_inr = FunctionRepresentation(input_dim, output_dim,
                                                    config["generator"]["layer_sizes"],
                                                    encoding, non_linearity,
                                                    final_non_linearity).to(device)
    return function_representation_inr