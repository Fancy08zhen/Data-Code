import os
import json

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from model import mobilenet_v3_large


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        sum_TP = 0
        sum_FP = 0
        sum_FN = 0
        sum_TN = 0
        precision_list = []
        recall_list = []
        specificity_list = []
        accuracy_list = []
        f1_list = []

        table = PrettyTable()
        table.field_names = ["Class", "Precision", "Recall", "Specificity", "Accuracy", "F1-score"]

        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            sum_TP += TP
            sum_FP += FP
            sum_FN += FN
            sum_TN += TN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            Accuracy = round((TP + TN) / (TP + FN + TN + FP), 3) if TP + FN + TN + FP != 0 else 0.
            F1 = round(2 * (Precision * Recall) / (Precision + Recall), 3) if Precision + Recall != 0 else 0.

            # 将每个类的指标添加到列表中
            precision_list.append(Precision)
            recall_list.append(Recall)
            specificity_list.append(Specificity)
            accuracy_list.append(Accuracy)
            f1_list.append(F1)

            table.add_row([self.labels[i], Precision, Recall, Specificity, Accuracy, F1])


        avg_precision = round(np.mean(precision_list), 4)
        avg_recall = round(np.mean(recall_list), 4)
        avg_specificity = round(np.mean(specificity_list), 4)
        avg_accuracy = round(np.mean(accuracy_list), 4)
        avg_f1 = round(np.mean(f1_list), 4)


        print("\nAverage Metrics:")
        print(f"Average Precision: {avg_precision}")
        print(f"Average Recall: {avg_recall}")
        print(f"Average Specificity: {avg_specificity}")
        print(f"Average Accuracy: {avg_accuracy}")
        print(f"Average F1-score: {avg_f1}")


        print(table)

    def plot(self):
        matrix = self.matrix
        plt.imshow(matrix, cmap=plt.cm.Blues)

        plt.xticks(range(self.num_classes), self.labels, rotation=90)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
    image_path = os.path.join(data_root, "E:/PC/PIC")
    assert os.path.exists(image_path), "Data path {} does not exist.".format(image_path)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test_oringinal"),
                                            transform=data_transform)

    batch_size = 16
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=2)

    # Initialize your model and load pre-trained weights
    net =mobilenet_v3_large(num_classes=60)
    model_weight_path = "E:/PC/MobileNet/weightsZ/model-99.pth"
    assert os.path.exists(model_weight_path), "Cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # Load class labels from JSON file
    json_label_path = 'E:/PC/NEW_DATA/class_locust.json'
    assert os.path.exists(json_label_path), "Cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=60, labels=labels)
    net.eval()
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to('cpu'))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()

