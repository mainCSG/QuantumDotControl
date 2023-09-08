import sys, os
import numpy as np
from PIL import Image

import yaml
import re

class DataProcessor():
    def __init__(self, data_folder, model_name, config_path):

        with open(config_path, 'r') as config_yaml:
            model_config = yaml.load(config_yaml, Loader=yaml.FullLoader)
            model_config = model_config[model_name]
            self.model_info = model_config['info']
            self.model_hyperparams = model_config['hyperparameters']

        self.num_of_train = self.model_hyperparams["dataset_size"] * self.model_hyperparams["train_val_split"]
        self.num_of_val = self.model_hyperparams["dataset_size"] * (1 - self.model_hyperparams["train_val_split"])

        self.data_folder = data_folder

        self.raw_folder = os.path.join(data_folder, "raw")
        self.raw_files = os.listdir(self.raw_folder)

    def process_files(self):
        # Process files in the raw folder

        counter = 0

        for file in self.raw_files:

            if file.endswith('npy'):

                raw_file_path = os.path.join(self.raw_folder, file)
                
                filename, ext = os.path.splitext(file)

                files_to_ignore = "|".join(self.model_info['files_to_ignore'])

                ignoreFile = bool(re.match(files_to_ignore, filename))

                if "exp" in filename and ignoreFile:
                        self.processed_folder = os.path.join(self.data_folder, "processed/test")
                        os.makedirs(self.processed_folder, exist_ok = True)

                        csd_data = self.process_npy_file(raw_file_path)
                        self.save_processed_file(csd_data, filename)

                if not(ignoreFile):

                    if "exp" in filename:
                        self.processed_folder = os.path.join(self.data_folder, "processed/train")
                        os.makedirs(self.processed_folder, exist_ok = True)
                    elif counter < self.num_of_train:
                        self.processed_folder = os.path.join(self.data_folder, "processed/train")
                        os.makedirs(self.processed_folder, exist_ok = True)
                    else:
                        self.processed_folder = os.path.join(self.data_folder, "processed/val")
                        os.makedirs(self.processed_folder, exist_ok = True)

                    if counter >= self.model_hyperparams["dataset_size"]:
                        break 

                    csd_data = self.process_npy_file(raw_file_path)

                    # Save the processed results to the processed folder
                    self.save_processed_file(csd_data, filename)

                    counter += 1

                    if counter % 100 == 0:
                        print(f"Processed #{counter}.")

    def process_npy_file(self, npy_file):

        # Loads, *.npy file, extracts CSD
        qflow_data = np.load(npy_file, allow_pickle=True).item()
        
        voltage_P1_key = "x" if "exp" in npy_file else "V_P1_vec"
        voltage_P2_key = "y" if "exp" in npy_file else "V_P2_vec"

        N = len(qflow_data[voltage_P1_key])
        M = len(qflow_data[voltage_P2_key])

        try:
            charge_sensor_data = np.array(
                [
                    data['sensor'][1] for data in qflow_data['output']
                ]
            ).reshape((N,M))

        except TypeError:
            charge_sensor_data = qflow_data['output']['sensor']

        except KeyError:
            charge_sensor_data = qflow_data['sensor']

        charge_sensor_data_norm = charge_sensor_data / np.amax(charge_sensor_data)

        return charge_sensor_data_norm
    
    def save_processed_file(self, csd_data, filename):
        # save CSD as image
        csd_image = Image.fromarray((255 * csd_data).astype(np.uint8))
        csd_image.save(os.path.join(self.processed_folder, '{}.jpg'.format(filename)))



data_dir = sys.argv[1]
model_name = sys.argv[2]
config_path = sys.argv[3]
data = DataProcessor(data_dir, model_name, config_path)
data.process_files()
