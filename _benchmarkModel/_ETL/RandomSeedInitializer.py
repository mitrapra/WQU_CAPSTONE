import tensorflow as tf 
import numpy as np 
import random
import torch
import os

class RandomSeedInitializer:
    """
    A class to initialize random seeds for basic Python, TensorFlow, and PyTorch.

    Attributes:
        default_seed (int): Default random seed value.
    """

    def __init__(self, default_seed=41689):
        """
        Initializes the RandomSeedInitializer class.

        Args:
            default_seed (int, optional): Default random seed value. Default is 41689.
        """
        self.default_seed = default_seed

    def set_basic_seed(self, seed=None):
        """
        Set random seeds for basic Python, NumPy, and the environment.

        Args:
            seed (int, optional): Random seed value. If None, use the default seed. Default is None.
        """
        seed = seed if seed is not None else self.default_seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

    def set_tf_seed(self, seed=None):
        """
        Set random seed for TensorFlow.

        Args:
            seed (int, optional): Random seed value. If None, use the default seed. Default is None.
        """
        seed = seed if seed is not None else self.default_seed
        tf.random.set_seed(seed)

    def set_torch_seed(self, seed=None):
        """
        Set random seed for PyTorch.

        Args:
            seed (int, optional): Random seed value. If None, use the default seed. Default is None.
        """
        seed = seed if seed is not None else self.default_seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark 	   = False

    def set_all_seeds(self, seed=None):
        """
        Set random seeds for basic Python, TensorFlow, and PyTorch.

        Args:
            seed (int, optional): Random seed value. If None, use the default seed. Default is None.
        """
        seed = seed if seed is not None else self.default_seed
        self.set_basic_seed(seed)
        self.set_tf_seed(seed)
        self.set_torch_seed(seed)
