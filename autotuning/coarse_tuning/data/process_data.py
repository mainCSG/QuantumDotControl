# Import modules 
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from skimage import measure


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def filter(qflow_npy_filepath: dict) -> dict:
    """Function filters out the QFlow 2.0 Lite dataset and extracts only relevant parameters
    for training the CSD model.

    Args:
        qflow_npy_filepath (dict): See references/dataset_structure.pdf Fig. 4 for structure.

    Returns:
        dict: Returns the filtered dataset containing dIdV and the regimes.
    """
    qflow_data = np.load(qflow_npy_filepath, allow_pickle=True).item()

    voltages = {"P1": qflow_data['V_P1_vec'], "P2": qflow_data['V_P2_vec']}

    # Assume square image
    dV1 = voltages["P1"][1] - voltages["P1"][0]
    N = len(voltages["P1"])

    # Extract current, state regimes and gradient
    I = np.array([
        data['current'] for data in qflow_data['output']
    ]).reshape((N,N))

    regimes = np.array([
        data['state'] for data in qflow_data['output']
    ]).reshape((N,N))

    grad = np.gradient(I, dV1)
    dIdVP1, dIdVP2 = grad[0], grad[1]

    def normalize(matrix):
        mean = matrix.mean()
        std = matrix.std()
        return (matrix - mean) / std

    dIdVP1 = normalize(dIdVP1)
    dIdVP2 = normalize(dIdVP2)
    I = normalize(I)


    # Create dataset, use np.float32 for PyTorch
    dataset = {"dIdVP1": np.array(I,dtype=np.float32),"dIdVP2": np.array(I,dtype=np.float32), "labels": regimes, "voltages": voltages}
    
    return dataset

def tile_data(dataset: dict, tile_size: int) -> dict:
    """Function that tilizes a NxN image to [(N/tile_size)**2, tile_size, tile_size] image.

    Args:
        dataset (dict): Dictionary of the incoming dataset
        tile_size (int): Desired tile size

    Returns:
        dict: Tilized version of the inputted dataset.
    """

    data_P1 = dataset["dIdVP1"]
    data_P2 = dataset["dIdVP2"]
    regimes = dataset["labels"]
    voltages = dataset["voltages"]

    data_P1_tiled = np.array([
        data_P1[x:x + tile_size, y:y+tile_size] for x in range(0, data_P1.shape[0], tile_size) for y in range(0, data_P1.shape[1], tile_size)
    ])

    data_P2_tiled = np.array([
        data_P2[x:x + tile_size, y:y+tile_size] for x in range(0, data_P2.shape[0], tile_size) for y in range(0, data_P2.shape[1], tile_size)
    ])

    regimes_tiled = np.array([
        regimes[x:x + tile_size, y:y+tile_size] for x in range(0, regimes.shape[0], tile_size) for y in range(0, regimes.shape[1], tile_size)
    ])
    
    probability_tiled = []
    for tile in regimes_tiled:
        prob_0QD = np.count_nonzero(tile == 0)/tile_size
        prob_1QD = np.count_nonzero(tile == 1)/tile_size
        prob_2QD = np.count_nonzero(tile == 2)/tile_size
        prob_neg1 = 1- (prob_0QD + prob_1QD + prob_2QD)
        probability_tiled.append([prob_0QD, prob_1QD, prob_2QD, prob_neg1])
    probability_tiled = np.array(probability_tiled)

    dataset_tiled = {"dIdVP1": data_P1_tiled,"dIdVP2": data_P2_tiled, "labels": probability_tiled, "voltages": voltages}

    return dataset_tiled

def process_data(qflow_raw_data_path: str, save_dir: str, tile_size: int, train_val_split: float, dataset_size: int) -> None:
    """Function that processes qflow's dataset and organizes it into training and validation data.

    Args:
        qflow_raw_data_path (str): Relative or absolute path to qflow's raw data.
        save_dir (str): Relative or absolute path to the desired save directory.
        tile_size (int): Desired tile size for the training data.
        train_val_split (float): Training validation split.
        display_epoch (int, optional): Gives the user some idea of where the program is. Defaults to 50.
    """
    num_of_train = int(np.floor(train_val_split * dataset_size))
    display_epoch = dataset_size // 2

    temp_tiled = []
    temp = []
    counter = 0
    for filename in os.listdir(qflow_raw_data_path):
        if filename.endswith(".npy"):
            if counter % display_epoch == 0:
                print("File {}/{} completed ... \n".format(counter, dataset_size))
            
            if counter == dataset_size: 
                break

            qflow_filepath = os.path.join(qflow_raw_data_path, filename)
            
            dataset = filter(qflow_filepath)
            dataset_tiled = tile_data(dataset, tile_size)

            temp_tiled.append(dataset_tiled)
            temp.append(dataset)

            counter += 1

    train_data_tiled = temp_tiled[:num_of_train]
    val_data_tiled = temp_tiled[num_of_train:]

    train_data = temp[:num_of_train]
    val_data = temp[num_of_train:]

    with open(save_dir+"/train/train.json", 'w') as fout:
        json.dump(train_data, fout, cls=NumpyEncoder)

    with open(save_dir+"/val/val.json", 'w') as fout:
        json.dump(val_data, fout, cls=NumpyEncoder)

    with open(save_dir+"/train/train_tiled.json", 'w') as fout:
        json.dump(train_data_tiled, fout, cls=NumpyEncoder)

    with open(save_dir+"/val/val_tiled.json", 'w') as fout:
        json.dump(val_data_tiled, fout, cls=NumpyEncoder)


def prepare_train_data_for_torch(path_to_data: str, batch_size: int, tiled: bool) -> list:
    """Functions takes in the filtered dataset and creates a PyTorch DataLoader object to use for training.

    Args:
        dataset (dict): Filtered and tiled QFlow dataset
        batch_size (int): Batch size desired for training

    Returns:
        List of dataLoader objects for both plunger gates.
    """

    suffix = "_tiled" if tiled else ""

    with open(path_to_data+'/train'+"/train"+suffix+".json", 'r') as f:
        dataset_train = json.load(f)

    torch_data_P1 = []
    torch_data_P2 = []
    torch_data_labels = []
    voltages = []

    for data in dataset_train:
        voltages.append(data["voltages"])

        torch_data_P1.append(data["dIdVP1"])
        torch_data_P2.append(data["dIdVP2"])
        torch_data_labels.append(data["labels"])

    data_P1 = torch.tensor(
        torch_data_P1, dtype=torch.float32
    )

    data_P2 = torch.tensor(
        torch_data_P2, dtype=torch.float32
    )

    labels = torch.tensor(
        torch_data_labels, dtype=torch.float32
    )

    torch_dataset_P1 = torch.utils.data.TensorDataset(data_P1, labels)
    torch_dataset_P2 = torch.utils.data.TensorDataset(data_P2, labels)

    dataloader_P1 = torch.utils.data.DataLoader(torch_dataset_P1, batch_size=batch_size)
    dataloader_P2 = torch.utils.data.DataLoader(torch_dataset_P2, batch_size=batch_size)

    return [dataloader_P1, dataloader_P2, voltages]

def prepare_val_data_for_torch(path_to_data: str, tiled: bool) -> list:
    """Functions takes in the filtered dataset and creates a PyTorch DataLoader object to use for validation.

    Args:
        dataset (dict): Filtered and tiled QFlow dataset
        batch_size (int): Batch size desired for validation

    Returns:
        List of dataLoader objects for both plunger gates.
    """

    suffix = "_tiled" if tiled else ""

    with open(path_to_data+'/val'+"/val"+suffix+".json", 'r') as f:
        dataset_train = json.load(f)

    torch_data_P1 = []
    torch_data_P2 = []
    torch_data_labels = []
    voltages = []
    for data in dataset_train:
        voltages.append(data["voltages"])
        torch_data_P1.append(data["dIdVP1"])
        torch_data_P2.append(data["dIdVP2"])
        torch_data_labels.append(data["labels"])

    data_P1 = torch.tensor(
        torch_data_P1, dtype=torch.float32
    )

    data_P2 = torch.tensor(
        torch_data_P2, dtype=torch.float32
    )

    labels = torch.tensor(
        torch_data_labels, dtype=torch.float32
    )

    torch_dataset_P1 = torch.utils.data.TensorDataset(data_P1, labels)
    torch_dataset_P2 = torch.utils.data.TensorDataset(data_P2, labels)

    dataloader_P1 = torch.utils.data.DataLoader(torch_dataset_P1)
    dataloader_P2 = torch.utils.data.DataLoader(torch_dataset_P2)

    return [dataloader_P1, dataloader_P2, voltages]

def gen_vgg_annotation_json(dataset_folder: str, save_folder: str, train_val_split: float, dataset_size: int) -> None:

    objects_list = []
    
    image_size = 100
    num_of_train = int(np.floor(train_val_split * dataset_size))

    counter = 0

    for filename in os.listdir(dataset_folder):
        if filename.endswith('.npy'):
            if counter == dataset_size:
                break

            filepath = os.path.join(dataset_folder, filename)
            data_dict = np.load(filepath, allow_pickle=True).item()

            labels_raw = np.array([d["state"] for d in data_dict['output']]).reshape((image_size, image_size))        

            labels = measure.label(
                labels_raw, background=-1, connectivity=1
                )

            regions = measure.regionprops(labels)

            regions_list = []
            for index in range(1, labels.max()):
                region_dict = {}
                vertices = regions[index].coords
                y, x = vertices.T

                regime = labels_raw[int(np.average(y)), int(np.average(x))]
                region_dict["shape_name"] = "polygon"
                region_dict["all_points_x"] = x.tolist()
                region_dict["all_points_y"] = y.tolist()
                region_dict["class"] = regime

                regions_list.append(region_dict)

            object = {}
            object["filename"] = filename
            object["size"] = image_size ** 2
            object["regions"] = regions_list
            
            objects_list.append(object)

            counter += 1

    vgg_format = {}
    for object in objects_list:
        
        filename = object["filename"]
        size = object["size"]
        regions = object["regions"]

        file_id = f"{filename}{size}"
        vgg_format[file_id] = {}
        vgg_format[file_id]["filename"] = filename
        vgg_format[file_id]["size"] = size
        vgg_format[file_id]["regions"] = {}

        index = 0

        for region in regions:
            region_dict = {}
            shape_attributes = {}

            shape_attributes["name"] = region["shape_name"]
            shape_attributes["all_points_x"] = region["all_points_x"]
            shape_attributes["all_points_y"] = region["all_points_y"]

            region_dict["shape_attributes"] = shape_attributes
            region_dict["region_attributes"] = {"label": region["class"]}

            vgg_format[file_id]["regions"][str(index)] = region_dict

            index += 1

        vgg_format[file_id]["file_attributes"] = {}


    keys, values = zip(*vgg_format.items())
    vgg_format_train = dict(zip(keys[:num_of_train], values[:num_of_train]))
    vgg_format_val = dict(zip(keys[num_of_train:], values[num_of_train:]))

    with open(save_folder+"/train/annotations_vgg.json", "w") as f:
        json.dump(vgg_format_train, f, cls=NumpyEncoder)
    with open(save_folder+"/val/annotations_vgg.json", "w") as f:
        json.dump(vgg_format_val, f, cls=NumpyEncoder)

# raw = "/Users/andrijapaurevic/Documents/UWaterloo/Research/mainCSG/QuantumDotControl/data/temp"
# save = "/Users/andrijapaurevic/Documents/UWaterloo/Research/mainCSG/QuantumDotControl/data/temp"
# tile_size = 10
# train_val_split = 0.5
# dataset_size = 2
# # process_data(raw,save,tile_size,train_val_split, dataset_size)
# # dataloader_P1, dataloader_P2 = prepare_for_torch(save, batch_size=10, tiled=True)
# gen_vgg_annotation_json(raw+"/train", save, train_val_split, dataset_size)
# gen_vgg_annotation_json(raw+"/val", save, train_val_split, dataset_size)