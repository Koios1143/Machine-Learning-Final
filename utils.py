import numpy as np
import torch
import PIL
import torchvision
from torchvision import transforms
from PIL import Image

def load_image(path):
    fp = open(path, 'rb')
    image = Image.open(fp).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0   #(425, 425, 3)
    fp.close()
    image = image[None].transpose(0, 3, 1, 2)           # (1, 3, 425, 425)
    image = torch.from_numpy(image)
    return image

def save_image(samples, path):     
    samples = 255 * samples.clamp(0,1)    # (1, 3, 425, 425)
    samples = samples.detach().numpy()
    samples = samples.transpose(0, 2, 3, 1)       #(1, 425, 425, 3)
    image = samples[0]                            #(425, 425, 3)
    image = Image.fromarray(image.astype(np.uint8))
    image.save(path)

def encode_img(input_img, vae):
    # Single image -> single latent in a batch (so size 1, 4, 53, 53)
    if len(input_img.shape)<4:
        input_img = input_img.unsqueeze(0)
    with torch.no_grad():
        latent = vae.encode(input_img*2 - 1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def decode_img(latents, vae):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach()
    return image

def transform(image):
    return transforms.Resize([512, 512])(image)

def to_PIL(tensor):
    return torchvision.transforms.ToPILImage()(tensor)