import torch
import torch.nn.functional as F
import random
from torchvision import transforms

class DACS_Mixer:
    def __init__(self, num_classes, ignore_label=255, confidence_threshold=0.968):
        self.num_classes = num_classes
        self.ignore_label = ignore_label

        # Threshold to filter pseudo-labels based on model confidence
        self.confidence_threshold = confidence_threshold

        # ImageNet mean and std for (de)normalization
        self.img_mean = torch.tensor([0.485, 0.456, 0.406])
        self.img_std = torch.tensor([0.229, 0.224, 0.225])

        # Post-mix data augmentation pipeline applied after mixing
        self.post_mix_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.5),
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
        ])

    def _denormalize(self, tensor_image):
        # Convert a normalized image back to [0, 1] RGB using ImageNet stats
        mean = self.img_mean.to(tensor_image.device).view(-1, 1, 1)
        std = self.img_std.to(tensor_image.device).view(-1, 1, 1)
        denorm_image = tensor_image * std + mean
        return torch.clamp(denorm_image, 0.0, 1.0)

    def _normalize(self, tensor_image):
        # Normalize an image using ImageNet mean and std
        mean = self.img_mean.to(tensor_image.device).view(-1, 1, 1)
        std = self.img_std.to(tensor_image.device).view(-1, 1, 1)
        return (tensor_image - mean) / std

    def __call__(self, model, X_s_batch, Y_s_batch, X_t_batch):
        """
        Apply DACS-style mixing for semi-supervised learning.

        Parameters:
        - model: segmentation model (used to get pseudo-labels for target images)
        - X_s_batch: source images (B, C, H, W)
        - Y_s_batch: source labels (B, H, W)
        - X_t_batch: target images (B, C, H, W)

        Returns:
        - Mixed images and labels for training
        - Lambda mix (fraction of confident pixels)
        - Pseudo-label map of target
        """
        model.eval()
        with torch.no_grad():
            # Resize target images if different resolution from source
            if X_s_batch.shape[2:] != X_t_batch.shape[2:]:
                X_t_for_pred = F.interpolate(X_t_batch, size=X_s_batch.shape[2:], mode='bilinear', align_corners=False)
            else:
                X_t_for_pred = X_t_batch

            # Get logits of target images from model
            outputs_t_tuple = model(X_t_for_pred)
            Y_t_pseudo_logits = outputs_t_tuple[0] if isinstance(outputs_t_tuple, tuple) else outputs_t_tuple
            # Get pseudo-labels
            pseudo_label_probs = torch.softmax(Y_t_pseudo_logits, dim=1)
            max_probs, pseudo_labels = torch.max(pseudo_label_probs, dim=1)

            # Calcuate Lambda mix (fraction of confident pixels)
            confident_pixel_mask = (max_probs >= self.confidence_threshold).float()
            lambda_mix = confident_pixel_mask.mean().item() if confident_pixel_mask.numel() > 0 else 0.0
            
            # Ignore low-confidence predictions
            pseudo_labels[max_probs < self.confidence_threshold] = self.ignore_label
            Y_t_pseudo_batch = pseudo_labels

        model.train()

        X_m_list, Y_m_list = [], []

        # Iterate over each sample in the batch
        for i in range(X_s_batch.size(0)):
            X_s = X_s_batch[i]
            Y_s = Y_s_batch[i]
            X_t = X_t_batch[i]
            Y_t_pseudo = Y_t_pseudo_batch[i]

            # Resize target image to match source resolution if needed
            if X_s.shape[1:] != X_t.shape[1:]:
                X_t_resized = F.interpolate(X_t.unsqueeze(0), size=X_s.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
            else:
                X_t_resized = X_t

            # Get present classes (excluding ignore label)
            present_classes = torch.unique(Y_s)
            present_classes = present_classes[present_classes != self.ignore_label]

            # # If no valid class present, skip mixing
            if len(present_classes) == 0:
                X_m_list.append(X_s)
                Y_m_list.append(Y_s)
                continue

            # Randomly select a subset of classes from source and mix the images
            num_classes_to_select = random.randint(1, max(1, len(present_classes) // 2))
            perm = torch.randperm(len(present_classes), device=Y_s.device)
            selected_classes = present_classes[perm[:num_classes_to_select]]

            mask = torch.zeros_like(Y_s, dtype=torch.bool)
            for cls_val in selected_classes:
                mask |= (Y_s == cls_val)

            mask_img = mask.unsqueeze(0).repeat(X_s.size(0), 1, 1)
            mask_lbl = mask

            X_m_normalized = torch.where(mask_img, X_s, X_t_resized)
            Y_m = torch.where(mask_lbl, Y_s, Y_t_pseudo)

            # Denormalize, apply post-mix transformations and re-normalize
            X_m_denormalized = self._denormalize(X_m_normalized)
            X_m_transformed = self.post_mix_transform(X_m_denormalized)
            X_m_final_normalized = self._normalize(X_m_transformed)

            # Ensure final shape consistency
            target_h, target_w = X_s.shape[1:]
            if X_m_final_normalized.shape[1:] != (target_h, target_w):
                X_m_final_normalized = F.interpolate(X_m_final_normalized.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
            if Y_m.shape != (target_h, target_w):
                Y_m = F.interpolate(Y_m.unsqueeze(0).unsqueeze(0).float(), size=(target_h, target_w), mode='nearest').squeeze(0).squeeze(0).long()

            X_m_list.append(X_m_final_normalized)
            Y_m_list.append(Y_m)

        if not X_m_list:
            return None, None, lambda_mix, Y_t_pseudo_batch

        return torch.stack(X_m_list), torch.stack(Y_m_list), lambda_mix, Y_t_pseudo_batch
