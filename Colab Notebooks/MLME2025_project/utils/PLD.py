import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import MiniBatchKMeans 
import torch.nn.functional as F
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import time
import torch.optim as optim
import os
import sys
import math


def extract_features(model, images, device):
    """
    Extract intermediate features from the model, which now returns a tuple (logits, features) in eval mode if required
    """
    model.eval()
    with torch.no_grad():
        # Run the model to get the output (logits, features)
        outputs = model(images, return_features=True)

        if isinstance(outputs, tuple):
            features = outputs[-1]
        else:
            raise RuntimeError("The model was expected to return a tuple (logits, features)")

        # Resize and flatten the features that have dimension (B*H*W, C_feat)
        features = F.interpolate(features, size=images.shape[2:], mode='bilinear', align_corners=False)
        features = features.permute(0, 2, 3, 1).contiguous().view(-1, features.shape[1])
        
        return features



class PLD_Adapter:
    def __init__(self, num_classes, ignore_label=255, confidence_threshold=0.968, feature_dim=256):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.confidence_threshold = confidence_threshold
        self.feature_dim = feature_dim
        self.source_centroids = None


    def compute_source_centroids(self, model, source_data_loader, device, save_path=None):
        """
        Computes source centroids incrementally to avoid excessive RAM consumption.
        This method maintains running sums and counts for each class, updating
        them every batch instead of loading the entire dataset into memory.
        """
        model.eval()

        # Initialize accumulators
        sums_per_class = torch.zeros(self.num_classes, self.feature_dim, dtype=torch.float64)
        counts_per_class = torch.zeros(self.num_classes, dtype=torch.int64)

        with torch.no_grad():
            for batch in tqdm(source_data_loader):
                X_s = batch['x'].to(device)
                Y_s = batch['y'].to(device)

                # Extract features and labels for the current batch of the source dataset
                features_s = extract_features(model, X_s, device)
                labels_s = Y_s.view(-1)
                
                # Filter out pixels with ignore label
                valid_mask = (labels_s != self.ignore_label)
                features_s = features_s[valid_mask]
                labels_s = labels_s[valid_mask]
                
                for cls_idx in range(self.num_classes):
                    # Find pixels of the current class in the batch
                    class_mask = (labels_s == cls_idx)
                    
                    if class_mask.sum() > 0:
                        # Extract corresponding features
                        class_features = features_s[class_mask]
                        
                        # Update the sum
                        sums_per_class[cls_idx] += class_features.sum(dim=0).cpu().to(torch.float64)
                        
                        # Update the count of pixels for this class
                        counts_per_class[cls_idx] += class_features.shape[0]
        
        # Compute final mean centroids that have dimension (num_classes, C)
        epsilon = 1e-8
        source_centroids_tensor = sums_per_class / (counts_per_class.unsqueeze(1) + epsilon)
        self.source_centroids = source_centroids_tensor.float()

        # Manage the case if some classes were not found in the source dataset
        for cls_idx in range(self.num_classes):
            if counts_per_class[cls_idx] == 0:
                print(f"Warning: Class {cls_idx} not found in source dataset. Its centroid will be a zero vector.")

        # Save centroids to file if a save path is provided
        if save_path:
            try:
                torch.save(self.source_centroids, save_path)
                print("Centroids saved")
            except Exception as e:
                print(f"Error while saving centroids: {e}")
        
        self.source_centroids = self.source_centroids.to(device)


    def load_centroids(self, load_path, device):
        """
        Loads precomputed centroids from a .pth file
        """
        if not os.path.exists(load_path):
            print(f"Centroids file not found")
            return False
        
        try:
            self.source_centroids = torch.load(load_path).to(device)
            print("Centroids loaded successfully")
            return True
        except Exception as e:
            print(f"Error while loading centroids: {e}")
            return False


    def __call__(self, model, X_t_batch, Y_t_pseudo_logits):
        """
        Performs label correction and alignment

        Args:
            model: the segmentation model
            X_t_batch (torch.Tensor): batch of target domain images
            Y_t_pseudo_logits (torch.Tensor): logits predicted by the model for the target batch

        Returns:
            torch.Tensor: tensor of corrected pseudo-labels for the target batch
        """
        if self.source_centroids is None:
            raise RuntimeError("Source centroids must be computed before applying PLD")

        # Extract features from the target batch and flatten (B*H*W, C_feat)
        target_features = extract_features(model, X_t_batch, X_t_batch.device)

        # Compute distances between each target feature and all source centroids (B*H*W, num_classes)
        distances = torch.cdist(target_features, self.source_centroids)

        # Convert distances into alignment probabilities, using the inverse of distance as similarity measure
        # The inverse distances are passed through a softmax function to obtain class alignment probabilities.
        # Using a low temperature (0.1) in softmax sharpens the distribution, meaning:
        # - The class with the smallest distance (i.e., closest centroid) will receive a much higher probability.
        # - All other classes will get very low probabilities.
        # This encourages more confident and discrete pseudo-label corrections based on feature similarity.
        aligned_probs = F.softmax(1.0 / (distances + 1e-6) / 0.1, dim=1)

        # Original logits Y_t_pseudo_logits are converted to probabilities and flattened 
        original_probs = F.softmax(Y_t_pseudo_logits, dim=1)
        original_probs = original_probs.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)

        # Combine the original probabilities with the alignment probabilities 
        # using element-wise multiplication between the two distributions
        # This correction reweights the model's predictions by how much they agree with the source-domain features.
        combined_probs = original_probs * aligned_probs
        combined_probs = combined_probs / combined_probs.sum(dim=1, keepdim=True)

        # Apply confidence threshold and assign final labels
        max_probs, assigned_labels = torch.max(combined_probs, dim=1)
        final_labels = torch.full_like(assigned_labels, self.ignore_label)
        confident_mask = max_probs >= self.confidence_threshold
        final_labels[confident_mask] = assigned_labels[confident_mask]

        # Reshape label tensor back to original image shape (B, H, W)
        return final_labels.view(X_t_batch.shape[0], X_t_batch.shape[2], X_t_batch.shape[3])
