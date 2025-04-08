from pathlib import Path
import torch
from dataloaders.dataloader import ImageDataset, InfiniteDataLoader, get_transform
from hyperparam.hyperparam import Hyperparameter

class DataManager:
    """
    Manages data loading and processing for style transfer training
    
    This class is responsible for:
    - Creating and managing dataset objects
    - Setting up data loaders
    - Maintaining iterators for content and style images
    """
    
    def __init__(self, hyper_param: Hyperparameter):
        """
        Initialize the DataManager
        
        Args:
            hyper_param: Configuration parameters for the training
        """
        self.hyper_param = hyper_param
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.setup_dataloaders()
        
    def setup_dataloaders(self):
        """Set up all data loaders for training and testing"""
        # Determine image size for transforms
        if isinstance(self.hyper_param.image_size, int):
            image_shape = (self.hyper_param.image_size, self.hyper_param.image_size)
        else:
            image_shape = self.hyper_param.image_shape
            
        self.image_shape = image_shape
        
        # Load content training data
        content_train_zip = list(Path(self.hyper_param.data_path).glob("content/train*.zip"))[0]
        self.content_train_dataloader = InfiniteDataLoader(
            ImageDataset(
                content_train_zip,
                transform=get_transform(
                    resize=self.hyper_param.resize_size,
                    crop_size=self.hyper_param.image_shape[-1],
                ),
            ),
            batch_size=self.hyper_param.batch_size,
            shuffle=True,
            num_workers=4,
        ).__iter__()

        # Load style training data
        style_train_zip = list(Path(self.hyper_param.data_path).glob("style/train*.zip"))[0]
        self.style_train_dataloader = InfiniteDataLoader(
            ImageDataset(
                style_train_zip,
                transform=get_transform(
                    resize=self.hyper_param.resize_size,
                    crop_size=self.hyper_param.image_shape[-1],
                ),
            ),
            batch_size=self.hyper_param.batch_size,
            shuffle=True,
            num_workers=4,
        ).__iter__()

        # Load content testing data
        content_test_zip = list(Path(self.hyper_param.data_path).glob("content/test*.zip"))[0]
        self.content_test_dataloader = InfiniteDataLoader(
            ImageDataset(
                content_test_zip,
                transform=get_transform(
                    resize=self.hyper_param.resize_size,
                    crop_size=self.hyper_param.image_shape[-1],
                ),
            ),
            batch_size=self.hyper_param.batch_size,
            shuffle=False,
            num_workers=4,
        ).__iter__()

        # Load style testing data
        style_test_zip = list(Path(self.hyper_param.data_path).glob("style/test*.zip"))[0]
        self.style_test_dataloader = InfiniteDataLoader(
            ImageDataset(
                style_test_zip,
                transform=get_transform(
                    resize=self.hyper_param.resize_size,
                    crop_size=self.hyper_param.image_shape[-1],
                ),
            ),
            batch_size=self.hyper_param.batch_size,
            shuffle=False,
            num_workers=4,
        ).__iter__()
        
    def get_batch(self, is_training=True):
        """
        Get a batch of content and style images
        
        Args:
            is_training: Whether to use training or testing data
        
        Returns:
            tuple: (content_batch, style_batch) tensors moved to the appropriate device
        """
        if is_training:
            content_batch = next(self.content_train_dataloader).to(self.device)
            style_batch = next(self.style_train_dataloader).to(self.device)
        else:
            content_batch = next(self.content_test_dataloader).to(self.device)
            style_batch = next(self.style_test_dataloader).to(self.device)
            
        return content_batch, style_batch