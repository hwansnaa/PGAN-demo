import torch
import numpy as np
import os
import cv2
import uuid

from configure import OUTPUT_DIR


class Inference:
    def __init__(self):
        self.model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                               'PGAN', model_name='celebAHQ-512',
                               pretrained=True, useGPU=False)

    def run(self, num_imgs):
        noise, _ = self.model.buildNoiseData(num_imgs)
        with torch.no_grad():
            generated_images = self.test(noise)
        output_name = uuid.uuid4()
        postprocessed_images = self.post_process(generated_images)
        output_file = os.path.join(OUTPUT_DIR, output_name)
        self.save_imgs(postprocessed_images, output_file)

    def post_process(self, images):
        images = np.array(images)
        images = np.transpose(images, (0, 2, 3, 1))
        images = ((images + 1) / 255).astype(np.int)
        return images

    @staticmethod
    def save_imgs(generated_images, output_path):
        cv2.imwrite(generated_images[0], output_path)
