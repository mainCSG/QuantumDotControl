import sys, os
import numpy as np
from PIL import Image
import h5py

class DataConverter():
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.raw_folder = os.path.join(data_folder, "raw")
        self.raw_files = os.listdir(self.raw_folder)

    def convert_files(self):
        # Process files in the raw folder
        for file in self.raw_files:
            if file.endswith(".hdf5"):
                raw_file_path = os.path.join(self.raw_folder, file)
                # Save the processed results to the processed folder
                self.save_as_npy(raw_file_path)
                
    def save_as_npy(self, hdf5_file_path):

        fileID = 0
        with h5py.File(hdf5_file_path, "r") as f:
            
            d = [n for n in f.keys()]
            npy_dict = {}
            npy_dict['output'] = {}
            for data in d:
                
                qflow_data = f[data]

                npy_dict['V_P1_vec'] = np.array(qflow_data["V_P1_vec"])
                npy_dict['V_P2_vec'] = np.array(qflow_data["V_P2_vec"])
                npy_dict['output']['sensor'] = np.array(qflow_data['output']['sensor'])
                npy_dict['output']['state'] = np.array(qflow_data['output']['state'])

                np.save(os.path.join(self.raw_folder, str(data)+str(fileID)+".npy"), npy_dict)

                fileID += 1

data_dir = sys.argv[1]
data = DataConverter(data_dir)
data.convert_files()
