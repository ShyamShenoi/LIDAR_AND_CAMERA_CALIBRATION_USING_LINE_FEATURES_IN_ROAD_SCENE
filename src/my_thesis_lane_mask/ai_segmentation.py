import os
import sys
import tarfile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Setup TF GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU Setup Error: {e}")

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


class CityscapesSegmenter:
    """Class to load inference model from tarball and run segmentation."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'ResizeBilinear_3:0'
    INPUT_SIZE = 769
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        self.graph = tf.Graph()
        graph_def = None

        with tarfile.open(tarball_path) as tar_file:
            for tar_info in tar_file.getmembers():
                if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                    file_handle = tar_file.extractfile(tar_info)
                    graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                    break

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
            self.sess = tf.compat.v1.Session(graph=self.graph, config=config)

    def process_image(self, image):
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        
        # ANTIALIAS is deprecated in newer PIL versions, replaced with LANCZOS or equivalent. 
        # Using Resampling.LANCZOS for modern PIL compatibility if supported, else fallback
        try:
            resample_mode = Image.Resampling.LANCZOS
        except AttributeError:
            resample_mode = Image.ANTIALIAS
            
        resized_image = image.convert('RGB').resize(target_size, resample_mode)

        with self.sess.as_default():
            batch_seg_map = self.sess.run(
                self.OUTPUT_TENSOR_NAME,
                feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]}
            )

            seg_map = batch_seg_map[:, :int(769 * height / width), :769, :]
            seg_map = tf.image.resize(seg_map, (height, width), method=tf.image.ResizeMethod.BILINEAR)
            seg_map = tf.nn.softmax(seg_map).eval()

        return resized_image, seg_map


def build_colormap():
    """Builds a Cityscapes colormap."""
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap


def colorize_label_mask(label):
    if label.ndim != 2:
        raise ValueError('Expected 2-D input label')
    colormap = build_colormap()
    if np.max(label) >= len(colormap):
        raise ValueError('Label value exceeds colormap capacity.')
    return colormap[label]


def process_image_with_ai(image_path, model_path="../../deeplabv3_cityscapes_train_2018_02_06.tar.gz"):
    """
    Runs segmentation on an image. 
    By default it points slightly up the folder tree to where the model tarball is stored.
    """
    model = CityscapesSegmenter(model_path)
    original_im = Image.open(image_path)
    resized_im, seg_map = model.process_image(original_im)
    
    seg_map = np.squeeze(seg_map)
    seg_map_result = np.squeeze(np.argmax(seg_map, 2))
    
    # We do not strictly need to visualize internally here, but returning ensures we have data
    return seg_map, seg_map_result
