import pandas as pd
import numpy as np
import yaml, os
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer

import torch
from PIL import Image, ImageDraw



from skimage import measure

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def mask_to_polygon(binary_mask, tolerance):
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

def setup_predictor(model_path: str, 
              config_path: str,
              model_name: str,
              processor: str,
              confidence_threshold: float):
    
    with open(config_path, 'r') as config_yaml:
            model_yaml = yaml.load(config_yaml, Loader=yaml.FullLoader)
            model_config = model_yaml[model_name]
            model_info = model_config['info']
            model_hyperparams = model_config['hyperparameters']
            model_device = model_yaml['device']

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.MODEL.DEVICE = processor 
    cfg.DATALOADER.NUM_WORKERS = model_device[processor]['num_workers']

    cfg.SOLVER.IMS_PER_BATCH = model_hyperparams['batch_num']
    cfg.SOLVER.BASE_LR = model_hyperparams['learning_rate']
    cfg.SOLVER.MAX_ITER = model_hyperparams['num_epochs']     
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = model_hyperparams['batch_size_per_img']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = model_info['num_of_classes'] 

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    MetadataCatalog.clear()
    MetadataCatalog.get("model").set(thing_classes=list(model_info['class_dict'].keys()))
    inference_metadata = MetadataCatalog.get('model')

    return predictor, inference_metadata

def convert_data_to_image(data: pd.DataFrame):
    X, Y, Z = data.columns[:3]
    raw_numpy_data = data.pivot_table(values=Z, index=[X], columns=[Y])
    Xdata, Ydata = raw_numpy_data.columns, raw_numpy_data.index

    raw_numpy_data = raw_numpy_data.values

    image_data = raw_numpy_data.copy()
    image_data /= raw_numpy_data.max()
    image_data *= 255
    image = image_data.astype(np.uint8)
    image = np.stack((image,image,image),axis=2)
    return image, Xdata, Ydata

def pixel_mask_to_polygon_units(mask, data, plot=False):
    image, Xdata, Ydata = convert_data_to_image(data)
    polygon_pixels = np.array(mask_to_polygon(mask.numpy().astype(int))).astype(int).reshape(-1,2)
    xs, ys = polygon_pixels[:,0], polygon_pixels[:,1]

    polygon_units = []
    for coordinate in polygon_pixels:
        polygon_units.append([Xdata[coordinate[0]], Ydata[coordinate[1]]])
    polygon_units = np.array(polygon_units)

    if plot:
        plt.imshow(image, extent=[Xdata.min(), Xdata.max(), Ydata.min(), Ydata.max()])
        plt.plot(polygon_units[:,0], polygon_units[:,1])
        plt.show()

    return polygon_units


def inference_data(data: pd.DataFrame, 
              model_path: str, 
              config_path: str,
              model_name: str,
              processor: str,
              polygon_threshold: float,
              confidence_threshold: float,
              plot_predictions: bool):

    predictor, metadata = setup_predictor(model_path, config_path, model_name, processor, confidence_threshold)
    image, Xdata, Ydata = convert_data_to_image(data)
    print(image,Xdata,Ydata)
    plt.imshow(image)
    plt.show()
    outputs = predictor(image) 
    print(outputs)
    filtered_masks = []
    for i, mask in enumerate(outputs['instances'].pred_masks.to('cpu')):
        polygon = np.array(mask_to_polygon(mask.numpy().astype(int), polygon_threshold)).astype(int)
        img = Image.new('L', (image.shape[1], image.shape[0]), 0)
        ImageDraw.Draw(img).polygon(polygon[0].tolist(), outline=1, fill=1)
        new_binary_mask = np.array(img)
        filtered_masks.append(torch.from_numpy(new_binary_mask))

    filtered_masks = torch.stack(filtered_masks)
    outputs['instances'].set('pred_masks', filtered_masks)

    if plot_predictions:
        v = Visualizer(
            image,
            metadata, 
            scale=3,
        )

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.title("Data")
        plt.imshow(out.get_image(), extent=[Xdata.min(), Xdata.max(), Ydata.min(), Ydata.max()])
        plt.xlabel(r'$V_X$ (mV)')
        plt.ylabel(r'$V_Y$ (mV)')
        plt.show()

    return outputs, metadata, image

def inference_image(image: np.array, 
              model_path: str, 
              config_path: str,
              model_name: str,
              processor: str,
              polygon_threshold: float,
              confidence_threshold: float,
              plot_predictions: bool):

    predictor, metadata = setup_predictor(model_path, config_path, model_name, processor, confidence_threshold)
    outputs = predictor(image) 

    filtered_masks = []
    for i, mask in enumerate(outputs['instances'].pred_masks.to('cpu')):
        polygon = np.array(mask_to_polygon(mask.numpy().astype(int), polygon_threshold)).astype(int)
        img = Image.new('L', (image.shape[1], image.shape[0]), 0)
        ImageDraw.Draw(img).polygon(polygon[0].tolist(), outline=1, fill=1)
        new_binary_mask = np.array(img)
        filtered_masks.append(torch.from_numpy(new_binary_mask))

    filtered_masks = torch.stack(filtered_masks)
    outputs['instances'].set('pred_masks', filtered_masks)

    if plot_predictions:
        v = Visualizer(
            image,
            metadata, 
            scale=3,
        )

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.title("Data")
        plt.imshow(out.get_image())
        plt.xlabel(r'$X$ (mV)')
        plt.ylabel(r'$Y$ (mV)')
        plt.show()

    return outputs, metadata, image