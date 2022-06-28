import torch
import os
from torchvision import transforms
from PIL import Image

from config import settings
from nets.model import ResNet34


class Predict(object):
    def __init__(self, model, weight_path, predict_folder, confidence_thres):
        self.folder_path = predict_folder
        self.beili_name_list = []
        self.image_name_list = []
        self.confidence_thres = confidence_thres
        self.class_indict = settings.get('class_indices')
        self.exp_folder = None
        self.weight_path = weight_path
        self.batch_size = settings.get('batch_size')

        self.model = model
        self.model.load_state_dict(torch.load(self.weight_path))

    def cleaning(self):
        for name in self.image_name_list:
            if name not in self.beili_name_list:
                os.remove(os.path.join(self.folder_path, name))

    def bath_predict(self, image_bath, iteration):
        # read class_indict
        # batch img
        bath_size = len(image_bath)
        batch_img = torch.stack(image_bath, dim=0)

        # create model
        # prediction
        self.model.eval()
        with torch.no_grad():
            # predict class
            output = self.model(batch_img.cuda()).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                if self.class_indict[cla.item()] == "positive" and pro.item() > self.confidence_thres:
                    name_index = idx + bath_size * iteration
                    self.beili_name_list.append(self.image_name_list[name_index])

    def folder_predict(self):

        data_transform = transforms.Compose([transforms.Resize([224, 224]),
                                             transforms.ToTensor(),
                                             ])

        # load image
        img_bath_list = []

        self.image_name_list = os.listdir(self.folder_path)

        name_groups = [self.image_name_list[i:i + self.batch_size] for i in range(0, len(self.image_name_list), self.batch_size)]
        for group in name_groups:
            img_bath = []
            for imageName in group:
                with Image.open(os.path.join(self.folder_path, imageName), 'r') as img:
                    img_bath.append(data_transform(img))
            img_bath_list.append(img_bath)

        for i, bath in enumerate(img_bath_list):
            self.bath_predict(bath, i)

        self.cleaning()


if __name__ == '__main__':
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet34(classes_num=2).to(device)
    a = Predict(model, 'model_data/model_weight_cuda.pth', 'daily_screenshot/a/5', 0.99)
    a.folder_predict()
