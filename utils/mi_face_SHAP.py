# Modified version of MIFace attack with SHAP-guided gradients and initialization.
# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements model inversion attacks.

| Paper link: https://dl.acm.org/doi/10.1145/2810103.2813677
"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.estimators.classification.classifier import ClassifierMixin, ClassGradientsMixin
from art.estimators.estimator import BaseEstimator
from art.attacks.attack import InferenceAttack
from art.utils import get_labels_np_array, check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class MIFace_SHAP(InferenceAttack):
    """
    Implementation of the MIFace algorithm from Fredrikson et al. (2015). While in that paper the attack is demonstrated
    specifically against face recognition models, it is applicable more broadly to classifiers with continuous features
    which expose class gradients.

    | Paper link: https://dl.acm.org/doi/10.1145/2810103.2813677
    """

    attack_params = InferenceAttack.attack_params + [
        "max_iter",
        "window_length",
        "threshold",
        "learning_rate",
        "batch_size",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin, ClassGradientsMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        max_iter: int = 10000,
        window_length: int = 100,
        threshold: float = 0.99,
        learning_rate: float = 0.1,
        batch_size: int = 1,
        verbose: bool = True,
        shap_importance: np.ndarray | None = None,  # Shape: (n_features,) or (timesteps, features)
        shap_weight: float = 0.5,                    # 0.0 = no SHAP, 1.0 = full SHAP weighting
        shap_init_ratio: float = 0.3, 
    ):
        """
        Create an MIFace attack instance.

        :param classifier: Target classifier.
        :param max_iter: Maximum number of gradient descent iterations for the model inversion.
        :param window_length: Length of window for checking whether descent should be aborted.
        :param threshold: Threshold for descent stopping criterion.
        :param batch_size: Size of internal batches.
        :param verbose: Show progress bars.
        :param shap_importance: Global SHAP importance scores (mean |SHAP| per feature).
        :param shap_weight: Weight for SHAP-guided gradients (0.0-1.0).
        :param shap_init_ratio: Ratio of top features to initialize non-zero.
        """
        super().__init__(estimator=classifier)

        self.max_iter = max_iter
        self.window_length = window_length
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.shap_importance = shap_importance
        self.shap_weight = np.clip(shap_weight, 0.0, 1.0)
        self.shap_init_ratio = np.clip(shap_init_ratio, 0.0, 1.0)
        
        # Preprocess SHAP importance
        if self.shap_importance is not None:
            # Normalize to [0, 1]
            self.shap_importance_norm = self._normalize_shap_importance(shap_importance)
            logger.info(f"SHAP-guided attack initialized with weight={self.shap_weight:.2f}")
        else:
            self.shap_importance_norm = None
            logger.warning("No SHAP importance provided. Falling back to standard MIFace_b.")
        self._check_params()

    def _normalize_shap_importance(self, importance: np.ndarray) -> np.ndarray:
        """Normalize SHAP importance to [0, 1] range and match input shape."""
        importance = np.abs(importance)
        
        # Reshape if needed (timesteps, features) -> (flattened,)
        if importance.ndim > 1:
            importance = importance.flatten()
        
        # Normalize to [0, 1]
        if importance.max() > importance.min():
            importance_norm = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)
        else:
            importance_norm = np.ones_like(importance)
        
        return importance_norm
    
    def _shap_initialized_input(self, y_target: np.ndarray) -> np.ndarray:
        """
        Initialize input using SHAP importance priors.
        
        Instead of zeros, initialize top-SHAP features with class-consistent values.
        """
        n_samples = len(y_target)
        input_shape = self.estimator.input_shape
        
        # Start with zeros
        x_init = np.zeros((n_samples,) + input_shape, dtype=ART_NUMPY_DTYPE)
        
        if self.shap_importance_norm is None:
            return x_init
        
        # Flatten for indexing
        n_features = np.prod(input_shape)
        top_k = int(n_features * self.shap_init_ratio)
        
        # Get indices of top-K important features
        top_indices = np.argsort(self.shap_importance_norm)[-top_k:]
        # Initialize top features with class-consistent values
        for i in range(n_samples):
            if y_target[i] == 1:
                # For class 1: set important features to high values (assuming positive correlation)
                # You can adjust this based on SHAP directionality
                flat_init = np.random.uniform(0.5, 1.0, n_features)
            else:
                # For class 0: set important features to low values
                flat_init = np.random.uniform(0.0, 0.5, n_features)
            
            # Zero out unimportant features
            mask = np.zeros(n_features, dtype=bool)
            mask[top_indices] = True
            flat_init[~mask] = 0.0
            
            # Reshape to input shape
            x_init[i] = flat_init.reshape(input_shape)
        
        return x_init
    
    def _shap_weighted_gradients(self, grads: np.ndarray) -> np.ndarray:
        """
        Weight gradients by SHAP importance.
        
        Amplifies gradients for important features, suppresses unimportant ones.
        """
        if self.shap_importance_norm is None or self.shap_weight == 0.0:
            return grads
        
        # Flatten gradients
        grads_flat = grads.reshape(grads.shape[0], -1)
        
        # Create importance mask (broadcast to batch)
        importance_mask = self.shap_importance_norm.reshape(1, -1)
        
        # Blend: (1 - weight) * raw_grads + weight * (shap_weighted_grads)
        weighted_grads = (
            (1 - self.shap_weight) * grads_flat + 
            self.shap_weight * (grads_flat * importance_mask)
        )
        
        # Reshape back
        return weighted_grads.reshape(grads.shape)

    def infer(self, x: np.ndarray | None, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """
        Extract a thieved classifier.

        :param x: An array with the initial input to the victim classifier. If `None`, then initial input will be
                  initialized as zero array.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: The inferred training samples.
        """
        if x is None and y is None:
            raise ValueError("Either `x` or `y` should be provided.")

        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        else:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        if x is None:
            x = self._shap_initialized_input(y)

        x_infer = x.astype(ART_NUMPY_DTYPE)

        # Compute inversions with implicit batching
        for batch_id in trange(
            int(np.ceil(x.shape[0] / float(self.batch_size))), desc="Model inversion", disable=not self.verbose
        ):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_infer[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]

            active = np.array([True] * len(batch))
            window = np.inf * np.ones((len(batch), self.window_length))

            i = 0

            while i < self.max_iter and sum(active) > 0:
                grads = self.estimator.class_gradient(batch[active], np.argmax(batch_labels[active], axis=1))
                grads = np.reshape(grads, (grads.shape[0],) + grads.shape[2:])
                grads = self._shap_weighted_gradients(grads)
                batch[active] = batch[active] + self.learning_rate * grads

                if self.estimator.clip_values is not None:
                    clip_min, clip_max = self.estimator.clip_values
                    batch[active] = np.clip(batch[active], clip_min, clip_max)

                cost = 1 - self.estimator.predict(batch)[np.arange(len(batch)), np.argmax(batch_labels, axis=1)]
                active = (cost <= self.threshold) + (cost >= np.max(window, axis=1))

                i_window = i % self.window_length
                window[::, i_window] = cost

                i = i + 1

            x_infer[batch_index_1:batch_index_2] = batch

        return x_infer

    def _check_params(self) -> None:
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.window_length, int) or self.window_length < 0:
            raise ValueError("The window length must be a non-negative integer.")

        if not isinstance(self.threshold, float) or self.threshold < 0.0:
            raise ValueError("The threshold must be a non-negative float.")

        if not isinstance(self.learning_rate, float) or self.learning_rate < 0.0:
            raise ValueError("The learning rate must be a non-negative float.")

        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError("The batch size must be a non-negative integer.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")