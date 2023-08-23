import numpy as np

def _normalize(image):
    for channel_id in range(image.shape[0]):
        image_channel = image[channel_id, :]
        image_channel = image_channel / 255
        image[channel_id,:] = image_channel
    return image    

def _scale(image):
    for channel_id in range(image.shape[0]):
        image_channel = image[channel_id, :]
        scale_factor = np.median(image_channel[image_channel>0])
        image_channel = image_channel.astype(np.float32) / scale_factor
        image[channel_id, :] = image_channel
    return image