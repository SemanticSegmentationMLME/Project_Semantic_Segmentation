import torch
import torch.nn.functional as F
import random
from torchvision import transforms

class PLD_DACS_Mixer:
    def __init__(self, num_classes, ignore_label=255, confidence_threshold=0.968):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.confidence_threshold = confidence_threshold
        self.img_mean = torch.tensor([0.485, 0.456, 0.406])
        self.img_std = torch.tensor([0.229, 0.224, 0.225])

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
        mean = self.img_mean.to(tensor_image.device).view(-1, 1, 1)
        std = self.img_std.to(tensor_image.device).view(-1, 1, 1)
        denorm_image = tensor_image * std + mean
        return torch.clamp(denorm_image, 0.0, 1.0)

    def _normalize(self, tensor_image):
        mean = self.img_mean.to(tensor_image.device).view(-1, 1, 1)
        std = self.img_std.to(tensor_image.device).view(-1, 1, 1)
        return (tensor_image - mean) / std

    def __call__(self, model, X_s_batch, Y_s_batch, X_t_batch, Y_t_pseudo_batch_corrected):
        """
        Applies DACS mixing between source and target batches
        using the pseudo labels of target images corrected with PLD
        """
        model.eval()
        with torch.no_grad():
            # Resize target images if resolution differs from source
            if X_s_batch.shape[2:] != X_t_batch.shape[2:]:
                X_t_for_pred = F.interpolate(X_t_batch, size=X_s_batch.shape[2:], mode='bilinear', align_corners=False)
            else:
                X_t_for_pred = X_t_batch

            # Extract logits from the model output on the target batch
            outputs_t = model(X_t_for_pred)
            if isinstance(outputs_t, tuple):
                Y_t_pseudo_logits = outputs_t[0]
            else:
                Y_t_pseudo_logits = outputs_t
            
            # Compute Lambda_mix (ratio of confident pixels)
            pseudo_label_probs = torch.softmax(Y_t_pseudo_logits, dim=1)
            max_probs, _ = torch.max(pseudo_label_probs, dim=1)
            confident_pixel_mask = (max_probs >= self.confidence_threshold).float()
            lambda_mix = confident_pixel_mask.mean().item()
            
        model.train()

        X_m_list, Y_m_list = [], []

        for i in range(X_s_batch.size(0)):
            X_s = X_s_batch[i]
            Y_s = Y_s_batch[i]
            X_t = X_t_batch[i]
            Y_t_pseudo = Y_t_pseudo_batch_corrected[i]

            # Resize target images if resolution differs from source
            if X_s.shape[1:] != X_t.shape[1:]:
                X_t_resized = F.interpolate(X_t.unsqueeze(0), size=X_s.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
            else:
                X_t_resized = X_t
            
            # Resize pseudo-labels to match source label resolution
            if Y_s.shape != Y_t_pseudo.shape:
                Y_t_pseudo_resized = F.interpolate(
                  Y_t_pseudo.unsqueeze(0).unsqueeze(0).float(), size=Y_s.shape, mode='nearest'
                ).squeeze(0).squeeze(0).long()
            else:
                Y_t_pseudo_resized = Y_t_pseudo

            # Mixing images same as before
            present_classes = torch.unique(Y_s)
            present_classes = present_classes[present_classes != self.ignore_label]

            if len(present_classes) == 0:
                X_m_list.append(X_s)
                Y_m_list.append(Y_s)
                continue

            num_classes_to_select = random.randint(1, max(1, len(present_classes) // 2))
            perm = torch.randperm(len(present_classes), device=Y_s.device)
            selected_classes = present_classes[perm[:num_classes_to_select]]

            mask = torch.zeros_like(Y_s, dtype=torch.bool)
            for cls_val in selected_classes:
                mask |= (Y_s == cls_val)

            mask_img = mask.unsqueeze(0).repeat(X_s.size(0), 1, 1)
            mask_lbl = mask

            X_m_normalized = torch.where(mask_img, X_s, X_t_resized)
            Y_m = torch.where(mask_lbl, Y_s, Y_t_pseudo_resized)

            target_h, target_w = X_s.shape[1:]
            X_m_denormalized = self._denormalize(X_m_normalized).cpu()
            X_m_transformed = self.post_mix_transform(X_m_denormalized)
            X_m_final_normalized = self._normalize(X_m_transformed.to(X_s.device))

            if X_m_final_normalized.shape[1:] != (target_h, target_w):
                X_m_final_normalized = F.interpolate(X_m_final_normalized.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
            if Y_m.shape != (target_h, target_w):
                Y_m = F.interpolate(Y_m.unsqueeze(0).unsqueeze(0).float(), size=(target_h, target_w), mode='nearest').squeeze(0).squeeze(0).long()
            
            X_m_list.append(X_m_final_normalized)
            Y_m_list.append(Y_m)

        if not X_m_list:
            return None, None, 0.0, None

        return torch.stack(X_m_list), torch.stack(Y_m_list), lambda_mix, Y_t_pseudo_batch_corrected
