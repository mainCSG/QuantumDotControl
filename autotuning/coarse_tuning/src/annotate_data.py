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
            ]).reshape((N,M))

        except TypeError:
            csd_qd_states = np.array(qflow_data['output']['state']).reshape((N,M))

        csd_qd_labelled_regions = sk.measure.label(csd_qd_states, background=-1, connectivity=1)
        csd_qd_regions = sk.measure.regionprops(csd_qd_labelled_regions)

        csd_object_list = []
        regions_list = []

        for index in range(len(csd_qd_regions)):
            region_info = {}

            region_coords = csd_qd_regions[index].coords

            # Get boundaries of coordiantes
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

            x,y = region_coords.T

            px = x.tolist()
            py = y.tolist()

            qd_num = csd_qd_states[int(np.average(x)), int(np.average(y))]

            poly = [(x, y) for x, y in zip(px, py)]
            poly = np.array([p for x in poly for p in x]).reshape(-1,2)

            poly_clockwise = self.organize_array_clockwise(poly)

            if len(px) <= 2 or len(py) <= 2:
                print("Ignoring polygon from ", npy_file, "because a polygon was too small for detectron2.")
                continue

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
    
    def dump_json(self):
         json_file_path = os.path.join(self.processed_folder, "via_region_data.json")

         with open(json_file_path, 'w') as f:
             json.dump(self.json_file, f, cls=NumpyEncoder)

data_dir = sys.argv[1]
annotate = AnnotateData(data_dir)
annotate.process_files()