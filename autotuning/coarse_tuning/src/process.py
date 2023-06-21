import sys, os
import numpy as np
from PIL import Image
import config as custom_cfg
import h5py

class DataProcessor():
    def __init__(self, data_folder):
        self.num_of_train = custom_cfg.dataset_size * custom_cfg.train_val_split
        self.num_of_val = custom_cfg.dataset_size - self.num_of_train
        self.data_folder = data_folder
        self.raw_folder = os.path.join(data_folder, "raw")
        self.raw_files = os.listdir(self.raw_folder)
        self.num_of_raw_files = len(self.raw_files)

    def process_files(self):
        # Process files in the raw folder

        counter = 0

        for file in self.raw_files:
            if file.endswith('npy'):

                raw_file_path = os.path.join(self.raw_folder, file)
                
                filename, ext = os.path.splitext(file)

                if "exp" in filename or "d_" in filename:
                        self.processed_folder = os.path.join(self.data_folder, "processed/test")
                        # Create the processed folder if it doesn't exist
                        os.makedirs(self.processed_folder, exist_ok = True)
                else:
                    if counter < self.num_of_train:
                        self.processed_folder = os.path.join(self.data_folder, "processed/train")
                        # Create the processed folder if it doesn't exist
                        os.makedirs(self.processed_folder, exist_ok = True)
                    else:
                        self.processed_folder = os.path.join(self.data_folder, "processed/val")
                        # Create the processed folder if it doesn't exist
                        os.makedirs(self.processed_folder, exist_ok = True)

                if counter >= custom_cfg.dataset_size:
                    break 

                csd_data = self.process_npy_file(raw_file_path)

                # Save the processed results to the processed folder
                self.save_processed_file(csd_data, filename)
                counter += 1
                if counter % 100 == 0:
                    print("Processed: ", counter)

    def process_npy_file(self, npy_file):
        # Loads, *.npy file, extracts CSD
        qflow_data = np.load(npy_file, allow_pickle=True).item()
        
        voltage_P1_key = "x" if "exp" in npy_file or "d_" in npy_file else "V_P1_vec"
        voltage_P2_key = "y" if "exp" in npy_file or "d_" in npy_file else "V_P2_vec"

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
   
        csd_image = Image.fromarray((255 * csd_data).astype(np.uint8))
        csd_image.save(os.path.join(self.processed_folder, '{}.jpg'.format(filename)))



data_dir = sys.argv[1]
data = DataProcessor(data_dir)
data.process_files()
