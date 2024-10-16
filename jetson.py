import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random

from gtts import gTTS
from IPython.display import display, Audio



seed = 42
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def gstreamer_pipeline(
        capture_width=640,
        capture_height=480,
        display_width=640,
        display_height=480,
        framerate=60,
        flip_method=0,
):
    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


def show_camera():
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)

        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, frame = cap.read()  # 카메라 시작
            cv2.imshow("CSI Camera", frame)

            keyCode = cv2.waitKey(30) & 0xFF

            if keyCode == 13:  # 사진 촬영
                cv2.imwrite('framse.jpg', frame)
                print("finish Taking a picture.")

                url = 'framse.jpg'
                frame = Image.open(url)
                display(frame)
                frame = np.array(frame)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (32, 32))
                frame = frame.astype('float32')

                PIL_tensor = transforms.ToTensor()
                frame = PIL_tensor(frame)

                PIL_normalize = transforms.Normalize((0.5,), (0.5,))
                Image_tensor = PIL_normalize(frame).unsqueeze(0)

                output = model(Image_tensor)
                print("output: {output}")

                val, indices = output.max(1)
                prob_percentage = np.exp(val) / np.sum(np.exp(output)) * 100

                pred_label = label_tags[indices.item()]

                print(f"predicted label is <{pred_label}>")
                speak = f"이 사물은 {prob_percentage}퍼센트의 확률로 {pred_label}입니다"
                wav = gTTS(speak, lang='ko')
                wav.save('pred_label.wav')

                display(Audio('pred_label.wav', autoplay=True))

                print("finished.")
            elif keyCode == 27:  # 카메라 종료
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        print("Unable to open camera")

if __name__ == '__main__':
    show_camera()

