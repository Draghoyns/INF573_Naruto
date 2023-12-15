import csv
import time
import copy
import argparse
import numpy as np

import cv2 as cv

import resnet_use as resnet
from utils import CvDrawText


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)

    parser.add_argument("--skip_frame", type=int, default=0)

    parser.add_argument(
        "--model",
        type=str,
        default="model/yolox/yolox_nano_with_post.onnx",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="416,416",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--score_th",
        type=float,
        default=0.7,
        help="Class confidence",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    fps = args.fps
    skip_frame = args.skip_frame

    model_path = args.model
    input_shape = tuple(map(int, args.input_shape.split(",")))
    score_th = args.score_th
    with_p6 = args.with_p6

    if args.file is not None:
        cap_device = args.file

    frame_count = 0

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap_fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")
    video_writer = cv.VideoWriter(
        filename="output.mp4",
        fourcc=fourcc,
        fps=cap_fps,
        frameSize=(cap_width, cap_height),
    )

    # モデルロード #############################################################
    model = resnet.load_model("resnet18_naruto_local.pth")

    # ラベル読み込み ###########################################################
    with open("setting/labels.csv", encoding="utf8") as f:
        labels = csv.reader(f)
        labels = [row for row in labels]

    while True:
        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            continue
        debug_image = copy.deepcopy(frame)

        frame_count += 1
        if (frame_count % (skip_frame + 1)) != 0:
            continue
        frame_height, frame_width = frame.shape[:2]

        # 検出実施 #############################################################

        class_id, score = resnet.predict_image(frame, model)
        class_id = int(class_id) + 1
        score = float(score)

        if score > score_th:
            cv.putText(
                debug_image,
                f"ID:{str(class_id)} {labels[class_id][0]} {score:.3f}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )
        else:
            cv.putText(
                debug_image,
                "No hand sign detected",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv.LINE_AA,
            )

        # 画面反映 #############################################################
        cv.imshow("NARUTO HandSignDetection Simple Demo", debug_image)
        video_writer.write(debug_image)

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1) if args.file is None else cv.waitKey(0)
        if key == 27:  # ESC
            break

    if video_writer:
        video_writer.release()
    if cap:
        cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
