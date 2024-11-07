import torch

def histogram_equalization(image):

    image = (image - image.min()) * 255 / (image.max() - image.min())
    image_tensor = 0.299*image[:,0,:,:] + 0.587*image[:,1,:,:] + 0.114*image[:,2,:,:]
    hist = torch.histc(image_tensor, bins=256, min=0, max=255)
    cdf = hist.cumsum(dim=0)
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    equalized_image = torch.gather(cdf.to(torch.int64), 0, image_tensor.to(torch.int64).reshape(-1)).float()
    equalized_image = torch.stack([equalized_image, equalized_image, equalized_image], dim=1)
    equalized_image = equalized_image
    
    return equalized_image.reshape(image.shape)