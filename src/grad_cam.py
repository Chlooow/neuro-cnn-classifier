"""Grad-CAM : carte de chaleur des zones qui ont influencé la prédiction du CNN."""

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image

from load_data import CLASS_NAMES, prepare_image, preprocess_for_inference

def find_last_conv(model):
    """Renvoie la dernière couche Conv2d du modèle."""
    last = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last = module
    if last is None:
        raise ValueError("Aucune couche Conv2d trouvée dans le modèle")
    return last

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        self.target_layer = target_layer if target_layer is not None else find_last_conv(model)
        self.activations = None
        self.hook = self.target_layer.register_forward_hook(self._save_activations)

    def _save_activations(self, module, inputs, output):
        self.activations = output

    def remove(self):
        self.hook.remove()

    def compute(self, x, class_idx=None):
        """x : tenseur (1, 1, H, W). Renvoie (carte entre 0 et 1, classe expliquée, probabilités)."""
        device = next(self.model.parameters()).device
        x = x.to(device).requires_grad_(True)

        self.model.zero_grad()
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        if class_idx is None:
            class_idx = int(probs.argmax())

        # gradient du score de la classe par rapport aux cartes de la dernière convolution
        grads = torch.autograd.grad(logits[0, class_idx], self.activations)[0]

        # poids de chaque carte = moyenne de son gradient, puis somme pondérée
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))

        # on agrandit à la taille de l'image d'entrée
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam / (cam.max() + 1e-8)
        return cam, class_idx, probs.detach().cpu().numpy()

def overlay_heatmap(gray, cam, alpha=0.4):
    """Superpose la carte (0-1) sur l'image en niveaux de gris (uint8)."""
    colors = colormaps["jet"](cam)[..., :3]
    base = np.stack([gray] * 3, axis=-1) / 255.0
    mix = (1 - alpha) * base + alpha * colors
    return Image.fromarray((mix * 255).astype(np.uint8))

def explain_image(model, image, stats, target_layer=None, class_idx=None):
    x = preprocess_for_inference(image, stats)
    gray = prepare_image(image, stats["image_size"])

    gradcam = GradCAM(model, target_layer)
    try:
        cam, explained_idx, probs = gradcam.compute(x, class_idx)
    finally:
        gradcam.remove()

    return {
        "pred_class": CLASS_NAMES[int(probs.argmax())],
        "explained_class": CLASS_NAMES[explained_idx],
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
        "heatmap": cam,
        "overlay": overlay_heatmap(gray, cam),
    }