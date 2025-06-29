import torch
import random
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
from PIL import Image

class FDATransform:
    def __init__(self, target_dataset=None, target_img_pil=None, beta=0.1):
        """
        Parameters:
        - target_dataset: dataset with target images
        - target_img_pil: a single PIL target image
        - beta: size of the low-frequency spectral region to transfer
        """
        self.target_dataset = target_dataset
        self.target_img_pil = target_img_pil
        self.beta = beta
        self.to_tensor = transforms.ToTensor()

    def __call__(self, src_img_pil):
        """
        Apply FDA (Fourier Domain Adaptation) transformation to the source image.

        Parameters:
        - src_img_pil: source image (PIL.Image)

        Returns:
        - Transformed source image (PIL.Image) with low-frequency components adapted from the target image.
        """
        if self.target_img_pil is None:
            if self.target_dataset is None:
                raise ValueError("Error: no target image available")

            # Randomly select a target image from the dataset
            tgt_img_pil = self.get_random_target_image()
        else:
            tgt_img_pil = self.target_img_pil

        # Resize target to match source
        tgt_img_pil = tgt_img_pil.resize(src_img_pil.size, Image.BILINEAR)

        # Convert PIL to tensor
        src_tensor = self.pil_to_tensor(src_img_pil)
        tgt_tensor = self.pil_to_tensor(tgt_img_pil)

        # Apply FDA
        fda_tensor = self.FDA_source_to_target(src_tensor, tgt_tensor, beta=self.beta)

        return to_pil_image(fda_tensor.squeeze(0))

    def get_random_target_image(self):
        idx = random.randint(0, len(self.target_dataset) - 1)
        img = self.target_dataset[idx]['x']
        if isinstance(img, torch.Tensor):
            img = to_pil_image(img)
        return img.convert('RGB')

    def pil_to_tensor(self, img):
        if isinstance(img, torch.Tensor):
            return img
        return self.to_tensor(img)

    def extract_ampl_phase(self, fft_im: torch.Tensor):
        # fft_im: complex tensor of shape [B, C, H, W]
        amp = torch.abs(fft_im)
        phase = torch.angle(fft_im)
        return amp, phase

    def low_freq_mutate(self, amp_src: torch.Tensor, amp_trg: torch.Tensor, beta: float):
        # Replace low-frequency components in amp_src with those from amp_trg
        if amp_src.dim() == 3:
            amp_src = amp_src.unsqueeze(0)
            amp_trg = amp_trg.unsqueeze(0)

        B, C, H, W = amp_src.shape
        b = int(min(H, W) * beta)

        amp_src_mut = amp_src.clone()

        amp_src_mut[:, :, 0:b, 0:b] = amp_trg[:, :, 0:b, 0:b]          
        amp_src_mut[:, :, 0:b, -b:] = amp_trg[:, :, 0:b, -b:]          
        amp_src_mut[:, :, -b:, 0:b] = amp_trg[:, :, -b:, 0:b] 
        amp_src_mut[:, :, -b:, -b:] = amp_trg[:, :, -b:, -b:] 

        return amp_src_mut

    def FDA_source_to_target(self, src_img: torch.Tensor, trg_img: torch.Tensor, beta: float):
        """
        Perform Frequency Domain Adaptation:
        - src_img: Source image tensor [B, C, H, W]
        - trg_img: Target image tensor [B, C, H, W]
        - beta: Fraction of lowest frequencies to swap
        """
        src_img = src_img.to(torch.float32)
        trg_img = trg_img.to(torch.float32)

        # FFT
        fft_src = torch.fft.fft2(src_img, dim=(-2, -1))
        fft_trg = torch.fft.fft2(trg_img, dim=(-2, -1))

        # Extract amplitude and phase
        amp_src, pha_src = self.extract_ampl_phase(fft_src)
        amp_trg, _ = self.extract_ampl_phase(fft_trg)

        # Replace low frequency amplitude
        amp_src_mut = self.low_freq_mutate(amp_src, amp_trg, beta=beta)

        # Reconstruct FFT using mutated amplitude and original phase
        fft_mut = amp_src_mut * torch.exp(1j * pha_src)

        # Inverse FFT 
        src_in_trg = torch.fft.ifft2(fft_mut, dim=(-2, -1)).real

        return src_in_trg
