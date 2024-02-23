import os
from typing import Dict, List, Optional

import cv2
import imageio
import numpy as np
import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.patches import FancyArrowPatch

def put_text_on_image(image: np.ndarray, lines: List[str]):
    assert image.dtype == np.uint8, image.dtype
    image = image.copy()

    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    for line in lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (0, 255, 0),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    return image


def append_text_to_image(image: np.ndarray, lines: List[str]):
    r"""Appends text left to an image of size (height, width, channels).
    The returned image has white text on a black background.
    Args:
        image: the image to put text
        text: a string to display
    Returns:
        A new image with text inserted left to the input image
    See also:
        habitat.utils.visualization.utils
    """
    # h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    y = 0
    for line in lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    # text_image = blank_image[0 : y + 10, 0:w]
    # final = np.concatenate((image, text_image), axis=0)
    final = np.concatenate((blank_image, image), axis=1)
    return final

def crop_image(image, crop_size=0.6):
    assert 0 < crop_size <= 1
    # Cropping from the center
    w, h = image.shape[:2]
    crop_w, crop_h = int(w * crop_size), int(h * crop_size)
    start_w, start_h = (w - crop_w) // 2, (h - crop_h) // 2
    cropped_img = image[start_w:start_w+crop_w, start_h:start_h+crop_h]

    return cropped_img

def put_info_on_image(image, info: Dict[str, float], extras=None, overlay=True):
    lines = [f"{k}: {v:.3f}" for k, v in info.items()]
    if extras is not None:
        lines.extend(extras)
    if overlay:
        return put_text_on_image(image, lines)
    else:
        return append_text_to_image(image, lines)
    
def overlay_image(image, overlay_image, resize=0.3, corner="right bottom"):
    assert corner in ["right bottom", "left bottom"]
    assert image.dtype == np.uint8, image.dtype
    assert overlay_image.dtype == np.uint8, overlay_image.dtype
    image = image.copy()
    overlay_image = overlay_image.copy()
    
    # Resize the overlay image
    if resize != 1.0:
        overlay_image = cv2.resize(overlay_image, (0, 0), fx=resize, fy=resize)
    
    # Get the image and overlay image sizes
    h, w, _ = image.shape
    h_, w_, _ = overlay_image.shape
    
    # Calculate the corner
    if corner == "right bottom":
        x = w - w_
        y = h - h_
    elif corner == "left bottom":
        x = 0
        y = h - h_
    
    # Overlay the image
    image[y:y+h_, x:x+w_, :3] = overlay_image
    return image

def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    verbose: bool = True,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    References:
        https://github.com/facebookresearch/habitat-lab/blob/main/habitat/utils/visualizations/utils.py
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    output_path = os.path.join(output_dir, video_name)
    writer = imageio.get_writer(output_path, fps=fps, quality=quality, **kwargs)
    if verbose:
        print(f"Video created: {output_path}")
        images_iter = tqdm.tqdm(images)
    else:
        images_iter = images
    for im in images_iter:
        # Convert to RGB
        writer.append_data(im)
    writer.close()


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)
    
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)

setattr(Axes3D, 'arrow3D', _arrow3D)


def transform_points(points, T):
    points = np.asarray(points)
    assert points.shape[1] == 3
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = points @ T.T
    return points[:, :3]

def transform_point(point, T):
    point = np.asarray(point)
    assert point.shape == (3,)
    point = np.hstack((point, np.ones(1)))
    point = point @ T.T
    return point[:3]