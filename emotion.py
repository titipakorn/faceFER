from resnet import resnet50
import torch
from torchvision.transforms import transforms
import numpy as np
import cv2
class FERClassifier: #Facial Expression
    """Wrapper class for a network returning a vector"""
#model-r100-ii
    def __init__(self, image_size=(224, 224), model_path='models/resnet50_rot30_2019Nov14_16.09'):
        self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
        self.FER_2013_EMO_DICT = {
    0: '怒り(Angry)',
    1: '嫌気(Disgust)',
    2: '恐怖(Fear)',
    3: '幸せ(Happy)',
    4: '悲しみ(Sad)',
    5: '驚き(Surprise)',
    6: '通常(Neutral)'
}

        self.model_fer = resnet50(in_channels=3, num_classes=7)
        self.model_fer.cuda()
        state = torch.load(str(model_path))
        self.model_fer.load_state_dict(state['net'])
        self.model_fer.eval()
        self.image_size = image_size


    def ensure_color(self,image):
        if len(image.shape) == 2:
            return np.dstack([image] * 3)
        elif image.shape[2] == 1:
            return np.dstack([image] * 3)
        return image


    def get_fer(self, imgs):
        results=[]
        emotions=[]
        with torch.no_grad():
            img = torch.cat(
                    [self.transform(cv2.resize(self.ensure_color(image), self.image_size)).unsqueeze(0) for image in imgs], dim=0).float().to("cuda")
            output = self.model_fer(img)
            for i in range(len(imgs)):
                proba = torch.softmax(output[i], 0)
                emo={}
                for index,prob in enumerate(proba):
                    emo[self.FER_2013_EMO_DICT[index]] = int(prob.item()*100)
                emo_proba, emo_idx = torch.max(proba, dim=0)
                # emo_proba = emo_proba.item()
                emo_idx = emo_idx.item()
                emotions.append(emo_idx)
                results.append(emo)
        return results,emotions

    def forward(self, images):
        """Performs forward of the underlying network on a given batch"""
        emotions = self.get_fer(images)
        return emotions
