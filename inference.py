import torch
import numpy as np
import os
import cv2
import uuid
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from configure import OUTPUT_DIR


class Inference:
    def __init__(self):
        self.model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                                    'PGAN', model_name='celebAHQ-512',
                                    pretrained=True, useGPU=False)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def run(self, num_imgs):
        num_imgs = int(num_imgs)
        noise, _ = self.model.buildNoiseData(num_imgs)
        with torch.no_grad():
            generated_images = self.model.test(noise)
        output_name = uuid.uuid4()
        postprocessed_images = self.postprocess(generated_images)
        for num in range(num_imgs):
            output_file_name = os.path.join(OUTPUT_DIR, f'{output_name}_{num}.png')
            self.save_img(output_file_name, postprocessed_images[num])
        return output_name, num_imgs

    @staticmethod
    def postprocess(images):
        images = np.array(images)
        images = np.clip(images, -1, 1)
        images = np.transpose(images, (0, 2, 3, 1))
        images = np.stack([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images], 0)
        images = ((images + 1) * 127.5).astype(np.int)
        return images

    @staticmethod
    def save_img(output_path, generated_image):
        cv2.imwrite(output_path, generated_image)


if __name__ == '__main__':
    infer = Inference()
    infer.run(4)
