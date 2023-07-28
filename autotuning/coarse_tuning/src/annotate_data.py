import os, sys
import skimage as sk
import numpy as np
import json 
import cv2
from imantics import Mask
import albumentations as A

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
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.raw_folder = os.path.join(data_folder, "raw")
        
    def process_files(self):
        # Process files in the raw folder
        sub_folders = os.listdir(os.path.join(self.data_folder, "processed"))

        for folder in sub_folders:
            self.json_file = {}
            if folder == '.DS_Store':
                continue
            elif folder == 'train':
                self.processed_folder = os.path.join(self.data_folder, "processed/train")
            elif folder == 'val':
                self.processed_folder = os.path.join(self.data_folder, "processed/val")
            elif folder == 'test':
                self.processed_folder = os.path.join(self.data_folder, "processed/test")

            for file in os.listdir(self.processed_folder):
                if file.endswith('.jpg') and folder != "test" and "augment" not in file:
                    filename, ext = os.path.splitext(file)
                    self.process_npy_file(filename)

            self.dump_json()

    def process_npy_file(self, npy_file):

        file_path =os.path.abspath(os.path.join(self.processed_folder, npy_file + ".jpg"))
        # Loads, *.npy file, extracts states
        qflow_data = np.load(os.path.join(self.raw_folder,npy_file+".npy"), allow_pickle=True).item()

        voltage_P1_key = "x" if "d_" in npy_file else "V_P1_vec"
        voltage_P2_key = "y" if "d_" in npy_file else "V_P2_vec"
        N = len(qflow_data[voltage_P1_key])
        M = len(qflow_data[voltage_P2_key])

        if "d_" in npy_file:

            # exp small dataset that is labelled
            csd_qd_states = self.convert_label_to_region_mask(qflow_data['sensor'], qflow_data['label'])

            background = -1
            correction = 1
        else:
            try:
                csd_qd_states = np.array([
                    data['state'] for data in qflow_data['output']
                ]).reshape((N,M) if N > M else (M,N))
                background = -1
                correction = 1

            except TypeError:
        
                csd_qd_states = np.array(qflow_data['output']['state']).reshape((N,M) if N > M else (M,N))
                background = -1
                correction = 2
        
        csd_qd_labelled_regions = sk.measure.label((correction * csd_qd_states).astype(np.uint8), background=background, connectivity=1)

        csd_qd_regions = sk.measure.regionprops(csd_qd_labelled_regions)

        num_of_predicted_regions = len(csd_qd_regions)

        if num_of_predicted_regions > 7:
            try:
                csd_qd_states = np.array([
                    data['state'] for data in qflow_data['output']
                ])
                background = -1
                correction = 1

            except TypeError:
                csd_qd_states = np.array(qflow_data['output']['state'])
                background = -1
                correction = 2
                
            csd_qd_states = csd_qd_states.reshape(M,N)

            csd_qd_labelled_regions = sk.measure.label((correction * csd_qd_states).astype(np.uint8), background=background, connectivity=1)

            csd_qd_regions = sk.measure.regionprops(csd_qd_labelled_regions)

        csd_object_list = []
        regions_list = []
        image_polygon_info_dict = {}
        image_polygon_info_dict["filename"] = file_path 
        image_polygon_info_dict["regions"] = []
        for index in range(len(csd_qd_regions)):
            region_info = {}

            region_coords = csd_qd_regions[index].coords

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
    
            poly = [(x, y) for x, y in zip(px, py)]
            poly = np.array([p for x in poly for p in x]).reshape(-1,2)
            if len(px) <= 70 or len(py) <= 70:
                # print("Ignoring polygon from ", npy_file, "because a polygon was too small for detectron2.")
                continue

            poly_clockwise = self.organize_array_clockwise(poly)

            x0, y0 = self.find_polygon_centroid(poly_clockwise)
            x0_, y0_ = self.flip_coordinates_horizontal_axis([x0], [y0], axis=csd_qd_states.shape[0]/2)
            x0_, y0_ = self.flip_coordinates_horizontal_axis(y0_, x0_, axis=csd_qd_states.shape[1]/2)
            x0_val, y0_val = x0_[0], y0_[0]

            class_dict = {0.0: "ND", -1.0: "ND", 0.5: "LD", 1.0: "CD", 1.5: "RD", 2.0: "DD"}
            num_of_dots = float(csd_qd_states[csd_qd_states.shape[0] - int(x0_val), csd_qd_states.shape[1] - int(y0_val)])
            qd_num = class_dict[num_of_dots]
            
            region_info['shape_name'] = 'polygon'
            region_info["all_points_x"] = [coord[0] for coord in poly_clockwise]
            region_info["all_points_y"] = [coord[1] for coord in poly_clockwise]
            region_info["class"] = qd_num

            image_polygon_info_dict["regions"].append((poly_clockwise, qd_num))

            regions_list.append(region_info)


        csd_object = {}
        csd_object["filename"] = file_path
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

            self.json_file[file_path] = {}
            self.json_file[file_path]["filename"] = filename
            self.json_file[file_path]["size"] = size
            self.json_file[file_path]["height"] = height
            self.json_file[file_path]["width"] = width
            self.json_file[file_path]["regions"] = {}

            index = 0 

            for region in regions:
                region_info = {}
                shape_info = {}

                shape_info["name"] = region["shape_name"]
                shape_info["all_points_x"] = region["all_points_x"]
                shape_info["all_points_y"] = region["all_points_y"]

                region_info["shape_attributes"] = shape_info
                region_info["region_attributes"] = {"label": region["class"]}

                self.json_file[file_path]["regions"][str(index)] = region_info

                index += 1
            
        self.json_file[file_path]["file_attributes"] = {}

        if "d_" not in file_path:    
            self.augment_image(image_polygon_info_dict)

    def augment_image(self, image_polygon_info_dict, num_of_augmented = 5):
        original_image_path = image_polygon_info_dict["filename"]
        image = cv2.imread(original_image_path)
        csd_object_list = []

        N, M = image.shape[0], image.shape[1]
        augmented_image_polygon_info_dict = {}
        
        def get_augmentation():
            return A.Compose([
                A.VerticalFlip(p=0.7),
                A.Rotate(limit=5, p=0.8),
                A.RandomBrightnessContrast(brightness_limit=(0.05,0.25), contrast_limit=(0, 0.1), p=0.8),
                A.GaussNoise(var_limit=(30.0,110.0), p=0.8),
                A.GridDistortion(distort_limit=0.2, p=0.8),
                A.Affine(scale=(1,1.3), p=0.9),
                A.RandomToneCurve(p=0.5),
                A.AdvancedBlur(p=0.8),
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


        for i in range(num_of_augmented):

            # Extract the directory and filename parts from the original file path
            directory, filename_extension = original_image_path.rsplit('/', 1)
            filename, extension = filename_extension.split('.')

            # Append the desired suffix to the filename
            new_filename = filename + f'_augment{i}'

            # Create the new file path by concatenating the directory, new filename, and extension
            new_file_path = f"{directory}/{new_filename}.{extension}"

            augmented_image_polygon_info_dict["filename"] = new_file_path

            regions = image_polygon_info_dict["regions"]

            masks_list = []
            for region in regions:
                masks_list.append(polygon_to_mask(region[0], image.shape))

            augmentation = get_augmentation()
            augmented = augmentation(image=image, masks=masks_list)
            augmented_image = augmented['image']
            augmented_masks = augmented['masks']

            cv2.imwrite(new_file_path, augmented_image)

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
annotate = AnnotateData(data_dir)
annotate.process_files()    