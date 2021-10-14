import torch
import numpy as np
import os
import pickle
from configure import ROOT_DIR, MODEL_DIR, PKL_NAME


class Inference:
    def __init__(self):
        model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                               'PGAN', model_name='celebAHQ-512',
                               pretrained=True, useGPU=False)

    def run(self, num_imgs):
        noise, _ = self.model.buildNoiseData(num_imgs)
        with torch.no_grad():
            generated_images = self.test(noise)
