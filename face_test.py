
import os
from emotion import FERClassifier
from face_det import FaceDet
from image_commons import draw_with_alpha,nparray_as_image
import cv2
FER = FERClassifier(model_path=(os.path.dirname(os.path.abspath(__file__)) + '/models/resnet50_rot30_2019Nov14_16.09'))
FACE_DET = FaceDet(os.path.dirname(os.path.abspath(__file__)) + '/models/Resnet50_Final.pth')
fer_colors=["#6490F2","#7DD5A9","#60708F","#EEBE45","#D66A58","#82C6E8","#8C63B4","#F29C56","#469292","#F19EC2"]
def _load_emoticons(emotions):
    """
    Loads emotions images from graphics folder.
    :param emotions: Array of emotions names.
    :return: Array of emotions graphics.
    """
    return [nparray_as_image(cv2.imread('graphics/%s.png' % emotion, -1), mode=None) for emotion in emotions]

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise','neutral']
emoticons = _load_emoticons(emotions)

image_path='test.jpeg'
dets,cords = FACE_DET.get_det(image_path)
result,emos = FER.get_fer(dets)
if(len(emos)>0):
    image_raw = cv2.imread(image_path)
    for idx,emo in enumerate(emos):
        image_to_draw = emoticons[emo]
        b=cords[idx]
        ch = b[3]-b[1]
        cw = b[2]-b[0]
        cc=int(min(ch,cw)/1.5)
        cx = max(b[0] - int(cc/2),0)
        cy = max(b[1]-int(cc/2),0)
        picked_color=fer_colors[idx].lstrip('#')
        cv2.rectangle(image_raw, (b[0], b[1]), (b[2], b[3]), tuple(int(picked_color[i:i+2], 16) for i in (0, 2, 4))[::-1], 5)
        draw_with_alpha(image_raw, image_to_draw, (cx, cy, cc, cc))
cv2.imwrite('test_result.jpg', image_raw)