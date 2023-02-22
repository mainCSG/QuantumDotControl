import os
import numpy as np
import matplotlib.pyplot as plt

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

    # Normalize dIdV to feed into model
    mean = dIdVP1.mean()
    std = dIdVP1.std()
    dIdVP1 = (dIdVP1 - mean) / std

    # Create dataset, use np.float64 for PyTorch
    dataset = {"dIdV": np.array(dIdVP1,dtype=np.float64), "regimes": regimes}
    
    return dataset
