import sys
sys.path.append('/home/maria/Documents/projects/anomaly_detection_study')

import numpy as np
import cv2
from dataau.realsyn.placer import PlaceFinder
from dataau.realsyn.direct_paste import DirectPaste
from dataau.realsyn.poisson_blending import PoissonBlending
from dataau.realsyn.alpha_blending import AlphaBlending

class RealSyn:
    def __init__(self, source, target, mask, method='poisson', resize_shape=None, verbose=False):
        self.source = source
        self.target = target
        self.mask = mask
        self.resize_shape = resize_shape
        self.method = method
        self.verbose = verbose

    def run(self):

        placer = PlaceFinder(dest_img=self.target, src_img=self.source, src_mask=self.mask, verbose=self.verbose)
        resized_image, cropped_fit_source, cropped_fit_mask, placement_mask, placement_image, centre_x, centre_y = placer.run()

        if self.method == 'direct':
            composer = DirectPaste(dest_img=self.target, src_img=resized_image, cropped_source=cropped_fit_source, cropped_mask=cropped_fit_mask, centre_x=centre_x, centre_y=centre_y)
            result = composer.apply()

        elif self.method == 'alpha':
            composer = AlphaBlending(dest_img=self.target, cropped_source=cropped_fit_source, cropped_mask=cropped_fit_mask, centre_x=centre_x, centre_y=centre_y, feather_radius=15)
            result = composer.apply()

        elif self.method == 'poisson':
            composer = PoissonBlending(placement_image=placement_image, dest_img=self.target, placement_mask=placement_mask)
            result, _ = composer.apply()

        if self.resize_shape:
            result = cv2.resize(result, (self.resize_shape[1], self.resize_shape[0]))
            placement_mask = cv2.resize(placement_mask, (self.resize_shape[1], self.resize_shape[0]))

        return result, np.expand_dims(placement_mask, axis=2)