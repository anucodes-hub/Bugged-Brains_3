import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np



class GradCAM:

    def __init__(self):

        # Load pre-trained ResNet50

        self.model = models.resnet50(pretrained=True)
        self.model.eval()
        self.feature_extractor = self.model.layer4  # Last conv layer
        self.gradients = None



    def save_gradient(self, grad):
        self.gradients = grad

    def get_heatmap(self, img_pil):
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

        input_tensor = transform(img_pil).unsqueeze(0)



        # Forward pass with hooks

        features = None

        def hook_feature(module, input, output):

            nonlocal features

            features = output

       

        handle_feat = self.feature_extractor.register_forward_hook(hook_feature)

        handle_grad = self.feature_extractor.register_full_backward_hook(lambda m, i, o: self.save_gradient(o[0]))

        output = self.model(input_tensor)
        idx = torch.argmax(output)

       

        # Backward pass

        self.model.zero_grad()
        output[0, idx].backward()



        # Compute weights and heatmap

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * features, dim=1).squeeze().detach().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        handle_feat.remove()
        handle_grad.remove()
        return cam



def apply_heatmap(img_cv, heatmap):

    # Resize heatmap to match original image

    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

   

    # Overlay heatmap on original image
    overlayed = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)
    return overlayed