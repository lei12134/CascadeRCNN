import os
import time
import json

import torch
import torchvision
import numpy as np
from PIL import Image

from torchvision import transforms
from network_files import FasterRCNN, AnchorsGenerator
from backbone import MobileNetV2
from draw_box_utils import draw_obj

from filter_files import Kalman
from filter_files import utils


def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    backbone = MobileNetV2().features
    backbone.out_channels = 1280

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=[7, 7],
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    # backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    # model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


GREEN = (0, 250, 0)
RED = (0, 0, 250)

COLOR_MEA = GREEN
COLOR_STA = RED
COLOR_MATCH = (255, 255, 0)


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=21)

    # load train weights
    weights_path = "./save_weights/mobile-model-24.pth"  # "./save_weights/model.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    data_transform = transforms.Compose([transforms.ToTensor()])

    img_name_list = os.listdir('./image')  # 获取图像名
    state_list = []  # 目标状态信息，存kalman对象
    model.eval()  # 进入验证模式
    with torch.no_grad():
        for index, value in enumerate(img_name_list):
            # Image object
            orignal_img = Image.open(os.path.join('./image', value))
            img = data_transform(orignal_img)
            # image Tensor
            img = torch.unsqueeze(img, dim=0)
            predictions = model(img.to(device))[0]
            predict_boxes = predictions["boxes"].to("cpu").numpy()
            # predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()
            idxs = np.greater(predict_scores, 0.5)
            # 目标检测的 boxes
            predict_boxes = predict_boxes[idxs]

            # 预测
            for target in state_list:
                target.predict()

            # 关联
            mea_list = [utils.box2meas(mea) for mea in predict_boxes]
            state_rem_list, mea_rem_list, match_list = Kalman.association(state_list, mea_list)

            # 状态未匹配上的，做删除处理
            state_del = []
            for idx in state_rem_list:
                state_del.append(state_list[idx])
            state_list = [val for val in state_list if val not in state_del]

            # 量测没匹配上的，作为新生目标进行航迹起始
            for idx in mea_rem_list:
                state_list.append(Kalman(utils.mea2state(mea_list[idx])))

            # -----------------------------------------------可视化-----------------------------------
            # 显示所有mea到图像上
            orignal_img = draw_obj(orignal_img, predict_boxes)
            # 显示所有的state到图像上
            state_list1 = []
            for kalman in state_list:
                pos = utils.state2box(kalman.X_posterior)
                state_list1.append(pos)
            orignal_img = draw_obj(orignal_img, np.array(state_list1), "Yellow")
            orignal_img.save(f"./saved_image/new_0001/{index}.png")


if __name__ == '__main__':
    main()
