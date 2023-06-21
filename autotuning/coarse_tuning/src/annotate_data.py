import os, sys
import skimage as sk
import numpy as np
import json 

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
        self.json_file = {}

    def process_files(self):
        # Process files in the raw folder
        sub_folders = os.listdir(os.path.join(self.data_folder, "processed"))

        for folder in sub_folders:
            if folder == '.DS_Store':
                continue
            elif folder == 'train':
                self.processed_folder = os.path.join(self.data_folder, "processed/train")
            elif folder == 'val':
                self.processed_folder = os.path.join(self.data_folder, "processed/val")
            elif folder == 'test':
                self.processed_folder = os.path.join(self.data_folder, "processed/test")

            for file in os.listdir(self.processed_folder):
                if file.endswith('.jpg') and folder != "test":
                    filename, ext = os.path.splitext(file)
                    self.process_npy_file(filename)

            self.dump_json()

    def process_npy_file(self, npy_file):
        # print(npy_file)
        file_path =os.path.abspath(os.path.join(self.processed_folder, npy_file + ".jpg"))
        # Loads, *.npy file, extracts CSD
        qflow_data = np.load(os.path.join(self.raw_folder,npy_file+".npy"), allow_pickle=True).item()

        voltage_P1_key = "x" if "d_" in npy_file else "V_P1_vec"
        voltage_P2_key = "y" if "d_" in npy_file else "V_P2_vec"
        N = len(qflow_data[voltage_P1_key])
        M = len(qflow_data[voltage_P2_key])

        try:
            csd_qd_states = np.array([
                data['state'] for data in qflow_data['output']
            ]).reshape((N,M) if N > M else (M,N))
            background = 0
            correction = 1

        except TypeError:
            csd_qd_states = np.array(qflow_data['output']['state']).reshape((N,M) if N > M else (M,N))
            background = -1
            correction = 2

        csd_qd_labelled_regions = sk.measure.label((correction * csd_qd_states).astype(np.uint8), background=background, connectivity=1)

        csd_qd_regions = sk.measure.regionprops(csd_qd_labelled_regions)

        num_of_predicted_regions = len(csd_qd_regions)

        if num_of_predicted_regions > 10:
            try:
                csd_qd_states = np.array([
                    data['state'] for data in qflow_data['output']
                ])
                background = -1
                correction = 1

            except TypeError:
                csd_qd_states = np.array(qflow_data['output']['state'])
                background = 0
                correction = 2
                
            csd_qd_states = csd_qd_states.reshape(M,N)

            csd_qd_labelled_regions = sk.measure.label((correction * csd_qd_states).astype(np.uint8), background=background, connectivity=1)

            csd_qd_regions = sk.measure.regionprops(csd_qd_labelled_regions)

        csd_object_list = []
        regions_list = []

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
            if len(px) <= 20 or len(py) <= 20:
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
    
    def dump_json(self):
         json_file_path = os.path.join(self.processed_folder, "via_region_data.json")

         with open(json_file_path, 'w') as f:
             json.dump(self.json_file, f, cls=NumpyEncoder)

data_dir = sys.argv[1]
annotate = AnnotateData(data_dir)
annotate.process_files()    