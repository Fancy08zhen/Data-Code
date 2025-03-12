import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from my_model import MyModel


def main():


    model = MyModel()
    model.load_state_dict(torch.load("my_model_weights.pth"))
    model.eval()


    target_layers = [model.features[-1]]


    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    img_path = "image.png"
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)


    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)


    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    targets = [ClassifierOutputTarget(predicted.item())]


    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.axis('off')
    plt.savefig('cam_visualization.png', bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()
