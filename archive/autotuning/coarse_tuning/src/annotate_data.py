import os, sys
import skimage as sk
import numpy as np
import json 
import cv2
from imantics import Mask
import albumentations as A
import yaml
import re

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

class AnnotateData():
    def __init__(self, data_folder, model_name, config_path):

        with open(config_path, 'r') as config_yaml:
            model_config = yaml.load(config_yaml, Loader=yaml.FullLoader)
            model_config = model_config[model_name]
            self.model_info = model_config['info']
            self.model_hyperparams = model_config['hyperparameters']

        self.model_name = model_name
        self.data_folder = data_folder
        self.raw_folder = os.path.join(data_folder, "raw")
        self.custom_folder = os.path.join(data_folder, "custom")
        
    def process_files(self):
        # Process files in the raw folder
        sub_folders = os.listdir(os.path.join(self.data_folder, "processed"))
        
        custom_json_training_file = os.path.join(self.custom_folder, self.model_info['custom_annotations_file'])

        with open(custom_json_training_file,'r') as f:
            custom_train_json = json.load(f)
            file_name_regexp = r".*\.jpg" 
            self.custom_filenames = [re.search(file_name_regexp, key).group() for key in custom_train_json.keys()]

        for folder in sub_folders:
            self.json_file = {}

            if folder == 'train':
                self.json_file = custom_train_json
                self.processed_folder = os.path.join(self.data_folder, "processed/train")

            elif folder == 'val':
                self.processed_folder = os.path.join(self.data_folder, "processed/val")

            elif folder == 'test':
                self.processed_folder = os.path.join(self.data_folder, "processed/test")

            else:
                continue
            
            for file in os.listdir(self.processed_folder):
                if file.endswith('.jpg') and folder != "test" and "augment" not in file:
                    filename, ext = os.path.splitext(file)
                    self.process_npy_file(filename)

            self.dump_json()

    def tuple_to_unique_number(self, charge_state):
        prime_numbers = [3, 5]  # List of prime numbers, should be equal to number of charge states
        unique_number = sum(element * prime_numbers[index] for index, element in enumerate(charge_state))
        return int(unique_number)

    def process_npy_file(self, npy_file):

        file_name_jpg = npy_file + ".jpg"

        if file_name_jpg not in self.custom_filenames: # no need to do polygon stuff with custom jsons

            # Loads, *.npy file, extracts states
            qflow_data = np.load(os.path.join(self.raw_folder,npy_file+".npy"), allow_pickle=True).item()

            N = len(qflow_data["V_P1_vec"])
            M = len(qflow_data["V_P2_vec"])

            if model_name == "dot_num":

                # Data is the QD regions
                csd_data = np.array(qflow_data['output'][self.model_info['data_key']])
                background = -1
                correction = 2 # because the numbers as decimals
                
                csd_data_labelled_regions = sk.measure.label((correction * csd_data).astype(np.uint8), background=background, connectivity=1)

                csd_data_regions = sk.measure.regionprops(csd_data_labelled_regions)

                class_dict = {0.0: "ND", -1.0: "ND", 0.5: "LD", 1.0: "CD", 1.5: "RD", 2.0: "DD"}


            elif model_name == "charge_state":

                # Data is the charge state regions
                csd_data = np.array([self.tuple_to_unique_number(data['charge']) for data in qflow_data['output']]).reshape((N,M) if N > M else (M,N))
                qd_regimes = np.array([data['state']  for data in qflow_data['output']]).reshape((N,M) if N > M else (M,N))
                
                important_charge_states_mask = np.zeros(qd_regimes.shape, dtype=bool)
                # charge states that matter
                min_charge = 0
                max_charge = 2
                spin_qubit_charge_states = [(i,j) for i in range(min_charge,max_charge + 1) for j in range(min_charge,max_charge + 1)]
                important_charge_states = [self.tuple_to_unique_number(x) for x in spin_qubit_charge_states]
                for x in important_charge_states:
                    important_charge_states_mask |= (csd_data == x)
                DD_mask = (qd_regimes == 2)
                
                csd_data *= DD_mask
                csd_data *= important_charge_states_mask
                # this will mask out all of the unneccesary charge states
                
                csd_data_labelled_regions = sk.measure.label(csd_data, connectivity=1)

                csd_data_regions = sk.measure.regionprops(csd_data_labelled_regions)

            csd_object_list = []
            regions_list = []
            image_polygon_info_dict = {}
            image_polygon_info_dict["filename"] = file_name_jpg 
            image_polygon_info_dict["regions"] = []
            
            for index in range(len(csd_data_regions)):
                region_info = {}

                region_coords = csd_data_regions[index].coords

                # Get boundaries of coordinates
                temp = {}
                for row in region_coords:
                    key = row[0]
                    value = row[1]
                    if key not in temp:
                        temp[key] = [value, value]  # Initialize with the current value
                    else:
                        temp[key][0] = min(temp[key][0], value)  # Update minimum value
                        temp[key][1] = max(temp[key][1], value)  # Update maximum value
                region_coords = np.array([[key, minmax[0]] for key, minmax in temp.items()] + [[key, minmax[1]] for key, minmax in temp.items()])

                y,x = region_coords.T

                px = x.tolist()
                py = y.tolist()
                
                if len(px) <= 10 or len(py) <= 10:
                    # print("Ignoring polygon from ", npy_file, "because a polygon was too small for detectron2.")
                    continue

                import matplotlib.pyplot as plt

                poly = [(x, y) for x, y in zip(px, py)]
                poly = np.array([p for x in poly for p in x]).reshape(-1,2)
                

                poly_clockwise = self.organize_array_clockwise(poly)

                x0, y0 = self.find_polygon_centroid(poly_clockwise)
                x0_, y0_ = self.flip_coordinates_horizontal_axis([x0], [y0], axis=csd_data.shape[0]/2)
                x0_, y0_ = self.flip_coordinates_horizontal_axis(y0_, x0_, axis=csd_data.shape[1]/2)
                x0_val, y0_val = x0_[0], y0_[0]

                data_value = float(csd_data[csd_data.shape[0] - int(x0_val), csd_data.shape[1] - int(y0_val)])
                class_value = class_dict[data_value] if self.model_name == "dot_num" else int(data_value)
                
                region_info['shape_name'] = 'polygon'
                region_info["all_points_x"] = [coord[0] for coord in poly_clockwise]
                region_info["all_points_y"] = [coord[1] for coord in poly_clockwise]
                region_info["class"] = class_value

                image_polygon_info_dict["regions"].append((poly_clockwise, class_value))

                regions_list.append(region_info)


            csd_object = {}
            csd_object["filename"] = file_name_jpg
            csd_object["size"] = N * M
            csd_object["height"] = N 
            csd_object["width"] = M
            csd_object["regions"] = regions_list

            csd_object_list.append(csd_object)

            for object in csd_object_list:
                
                filename = object["filename"]
                size = object["size"]
                regions = object["regions"]
                height = csd_object["height"]
                width =  csd_object["width"]  

                self.json_file[file_name_jpg] = {}
                self.json_file[file_name_jpg]["filename"] = filename
                self.json_file[file_name_jpg]["size"] = size
                self.json_file[file_name_jpg]["height"] = height
                self.json_file[file_name_jpg]["width"] = width
                self.json_file[file_name_jpg]["regions"] = {}

                index = 0 

                for region in regions:
                    region_info = {}
                    shape_info = {}

                    shape_info["name"] = region["shape_name"]
                    shape_info["all_points_x"] = region["all_points_x"]
                    shape_info["all_points_y"] = region["all_points_y"]

                    region_info["shape_attributes"] = shape_info
                    region_info["region_attributes"] = {"label": region["class"]}

                    self.json_file[file_name_jpg]["regions"][str(index)] = region_info

                    index += 1
                
            self.json_file[file_name_jpg]["file_attributes"] = {}

        else: 
            image_polygon_info_dict = {}
            # one of the custom ones, need to create custom image dict
            image_polygon_info_dict["filename"] = file_name_jpg
            image_polygon_info_dict["regions"] = []
            for idx, v in enumerate(self.json_file.values()):
                if v["filename"] == file_name_jpg:
                    regions = v["regions"]
                    for region in regions:
                        px = region["shape_attributes"]["all_points_x"]
                        py = region["shape_attributes"]["all_points_y"]
                        poly = [(x, y) for x, y in zip(px, py)]
                        poly = np.array([p for x in poly for p in x]).reshape(-1,2)
                        # FIX
                        if self.model_name == "charge_state":
                            charge_state_string = region["region_attributes"]["label"]
                            charge_state_tuple = (int(charge_state_string[1]), int(charge_state_string[3]))
                            class_value = self.tuple_to_unique_number(charge_state_tuple)
                        else:
                            class_value = region["region_attributes"]["label"]
                        image_polygon_info_dict["regions"].append((poly, class_value))

        self.augment_image(image_polygon_info_dict)

    def augment_image(self, image_polygon_info_dict):
        original_image_path = image_polygon_info_dict["filename"]

        image = cv2.imread(os.path.join(self.processed_folder,original_image_path))
        csd_object_list = []

        N, M = image.shape[0], image.shape[1]
        augmented_image_polygon_info_dict = {}
        
        def get_exp_data_augmentation():
            return A.Compose([
                A.VerticalFlip(p=0.7),
                A.RandomBrightnessContrast(brightness_limit=(0.05,0.075), contrast_limit=(0.05, 0.1), p=0.8),
                A.GaussNoise(var_limit=(0.0,70.0), p=0.8),
                A.Affine(scale=(1,1.4), p=0.8),
                A.RandomToneCurve(p=0.8),
                A.AdvancedBlur(blur_limit=(9,11),p=0.8),
                A.RingingOvershoot(p=0.5),
            ], is_check_shapes=False)

        def get_sim_data_augmentation():
            return A.Compose([
                A.VerticalFlip(p=0.7),
                A.RandomBrightnessContrast(brightness_limit=(0.05,0.1), contrast_limit=(0.05, 0.1), p=1),
                A.GaussNoise(var_limit=(50.0,120.0), p=0.8),
                A.GridDistortion(distort_limit=0.2, p=0.8),
                A.Affine(scale=(1,1.3), p=0.8),
                A.RandomToneCurve(p=1),
                A.AdvancedBlur(blur_limit=(9,11),p=1),
                A.RingingOvershoot(p=0.5)
            ], is_check_shapes=False)

        def polygon_to_mask(polygon_coords, image_shape):
            mask = np.zeros(image_shape[:2])
            cv2.fillPoly(mask, [polygon_coords], 255)
            return np.array(mask)

        def PolyArea(x,y):
            return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

        def convert_mask_to_polygon_coordinates(mask):
            polygons = Mask(mask).polygons()
            if len(polygons.points) != 0:
                polygon_area = []
                for i in range(len(polygons.points)):
                    x,y = zip(*polygons.points[i])
                    polygon_area.append(PolyArea(x,y))
                idx = np.argmax(polygon_area)

                return polygons.points[idx]
            else: 
                return []
            
        if "exp_large" in original_image_path:
            num_of_augmented = self.model_hyperparams['augments_per_exp_img']
            augmentation = get_exp_data_augmentation()
        else:
            num_of_augmented = self.model_hyperparams['augments_per_sim_img']
            augmentation = get_sim_data_augmentation()

        for i in range(num_of_augmented):

            # Extract the directory and filename parts from the original file path
            # directory, filename_extension = original_image_path.rsplit('/', 1)
            filename, extension = original_image_path.split('.')

            # Append the desired suffix to the filename
            new_filename = filename + f'_augment{i}'

            # Create the new file path by concatenating the directory, new filename, and extension
            new_file_path = f"{new_filename}.{extension}"

            augmented_image_polygon_info_dict["filename"] = new_file_path

            regions = image_polygon_info_dict["regions"]
            # print(regions)

            masks_list = []
            for region in regions:
                masks_list.append(polygon_to_mask(region[0], image.shape))

            augmented = augmentation(image=image, masks=masks_list)
            augmented_image = augmented['image']
            augmented_masks = augmented['masks']

            cv2.imwrite(os.path.join(self.processed_folder, new_file_path), augmented_image)

            regions_list = []
            counter = 0
            for region in regions:

                augmented_polygon = convert_mask_to_polygon_coordinates(augmented_masks[counter])

                if len(augmented_polygon) == 0:
                    continue
                
                region_info = {}
                region_info['shape_name'] = 'polygon'
                region_info["all_points_x"] = [coord[0] for coord in augmented_polygon]
                region_info["all_points_y"] = [coord[1] for coord in augmented_polygon]
                region_info["class"] = region[1]
            
                regions_list.append(region_info)
                counter += 1

            csd_object = {}
            csd_object["filename"] = new_file_path
            csd_object["size"] = N * M
            csd_object["height"] = N 
            csd_object["width"] = M
            csd_object["regions"] = regions_list

            csd_object_list.append(csd_object)

            for object in csd_object_list:
                
                filename = object["filename"]
                size = object["size"]
                regions = object["regions"]
                height = csd_object["height"]
                width =  csd_object["width"]  

                self.json_file[new_file_path] = {}
                self.json_file[new_file_path]["filename"] = filename
                self.json_file[new_file_path]["size"] = size
                self.json_file[new_file_path]["height"] = height
                self.json_file[new_file_path]["width"] = width
                self.json_file[new_file_path]["regions"] = {}

                index = 0 

                for region in regions:
                    region_info = {}
                    shape_info = {}

                    shape_info["name"] = region["shape_name"]
                    shape_info["all_points_x"] = region["all_points_x"]
                    shape_info["all_points_y"] = region["all_points_y"]

                    region_info["shape_attributes"] = shape_info
                    region_info["region_attributes"] = {"label": region["class"]}

                    self.json_file[new_file_path]["regions"][str(index)] = region_info

                    index += 1
                
            self.json_file[new_file_path]["file_attributes"] = {}

    def organize_array_clockwise(self, arr):
                
        # Calculate the centroid of the points
        centroid = np.mean(arr, axis=0)

        # Calculate the angle of each point with respect to the centroid
        angles = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])

        # Sort the points based on the angles in clockwise order
        indices = np.argsort(angles)
        sorted_arr = arr[indices]

        return sorted_arr        
    
    def flip_coordinates_horizontal_axis(self, x_coordinates, y_coordinates, axis):
        flipped_x_coordinates = []
        flipped_y_coordinates = []
        
        for x, y in zip(x_coordinates, y_coordinates):
            distance = (axis - y)
            y_flipped = axis + distance
            flipped_x_coordinates.append(x)
            flipped_y_coordinates.append(y_flipped)
        
        return flipped_x_coordinates, flipped_y_coordinates

    def find_polygon_centroid(self, coordinates):
        n = len(coordinates)
        
        # Check if all x-values are the same
        x_values = [x for x, _ in coordinates]
        if len(set(x_values)) == 1:
            centroid_x = x_values[0]
            
            # Calculate the average of the y-values
            y_values = [y for _, y in coordinates]
            centroid_y = sum(y_values) / n
        else:
            # Calculate the signed area of the polygon
            signed_area = 0
            for i in range(n):
                x_i, y_i = coordinates[i]
                x_j, y_j = coordinates[(i + 1) % n]
                signed_area += (x_i * y_j - x_j * y_i)
            signed_area *= 0.5
            
            # Calculate the coordinates of the centroid
            centroid_x = 0
            centroid_y = 0
            for i in range(n):
                x_i, y_i = coordinates[i]
                x_j, y_j = coordinates[(i + 1) % n]
                factor = x_i * y_j - x_j * y_i
                centroid_x += (x_i + x_j) * factor
                centroid_y += (y_i + y_j) * factor
            centroid_x /= (6 * signed_area)
            centroid_y /= (6 * signed_area)
        
        return centroid_x, centroid_y
    
    def convert_label_to_region_mask(self, data, label):

        label_index_to_region_dict = {0: 0.0, 1: 0.5, 2:1 , 3: 1.5, 4: 2}
        region_num = label_index_to_region_dict[np.argmax(label)]
        return region_num * np.ones_like(data)

    def dump_json(self):
         json_file_path = os.path.join(self.processed_folder, "via_region_data.json")

         with open(json_file_path, 'w') as f:
             json.dump(self.json_file, f, cls=NumpyEncoder)

data_dir = sys.argv[1]
model_name = sys.argv[2]
config_path = sys.argv[3]
annotate = AnnotateData(data_dir, model_name, config_path)
annotate.process_files()    