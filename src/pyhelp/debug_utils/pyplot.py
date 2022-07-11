"""
    Wrapper for matplotlib.pyplot for better use in vscode debugging.
"""
import matplotlib.pyplot as matplotlibplt
import numpy as np
import torch

image_std = np.array([0.229, 0.224, 0.225])
rgb_mean = np.array([0.485, 0.456, 0.406]),

def tensor2numpy(image):
    if isinstance(image, torch.Tensor):
        return image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise NotImplementedError

def deal_axis(image:np.ndarray):
    # if [H, W] just passed through
    if len(image.shape) == 2:
        return image.copy()

    # image_should_be batch == 0
    if len(image.shape) == 4:
        _image = image[0].copy()
    else:
        _image = image.copy()
    
    if _image.shape[0] == 1: #[1, H, W]
        return _image[0] #[H, W]
    
    if _image.shape[-1] == 1: #[H, W, 1]
        return _image[..., 0] #[H,W]
    
    if _image.shape[0] == 3: #[3, H, W]
        return _image.transpose(1, 2, 0) #[H, W, 3]
    
    if _image.shape[-1] == 3: #[H, W, 3]
        return _image
    
    # [C, H, W] by default
    return _image.transpose(1, 2, 0) #[H, W, C]
    
    
def type_agnosis(image:np.ndarray):
    # commonly used visualize image:
    # [H, W], single heatmap or depth;
    # [H, W, 3], rgb image;
    # [H, W, C], feature maps
    if len(image.shape) == 2:
        return "single"
    if len(image.shape) == 3 and image.shape[-1] == 3:
        return "rgb"
    else:
        return "feature"
    
def show_single(image, *args, **kwargs):
    matplotlibplt.imshow(image, *args, **kwargs)

def show_rgb(image:np.ndarray, **kwargs):
    if image.dtype == np.uint8:
        matplotlibplt.imshow(image, **kwargs)
        return
    
    denorm = kwargs.get('denorm', False)
    if denorm:
        image = image * image_std + rgb_mean
    image = image * 255
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    matplotlibplt.imshow(image, **kwargs)
    return

def show_feature(image:np.ndarray, **kwargs):
    normed_image = np.linalg.norm(image, axis=-1)
    matplotlibplt.imshow(normed_image, **kwargs)


show_dict={
    "single": show_single,
    "rgb": show_rgb,
    "feature": show_feature
}

def imshow(images, *args, **kwargs):
    if isinstance(images, list):
        num_images = len(images)
        num_cols = int(np.sqrt(num_images + 1))
        num_rows = int(np.ceil(num_images / num_cols))
    else:
        images = [images]
        num_rows = 1
        num_cols = 1
    
    for i, image in enumerate(images):
        matplotlibplt.subplot(num_rows, num_cols, i + 1)
        image = tensor2numpy(image)
        image = deal_axis(image)
        image_type = type_agnosis(image)
        print(f"Debugging image type: {image_type}")
        show_dict[image_type](image, *args, **kwargs)
    
    matplotlibplt.show()
    matplotlibplt.savefig("debug.png")
