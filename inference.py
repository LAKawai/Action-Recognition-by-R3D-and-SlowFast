import torch
import numpy as np
import cv2
import os
import argparse
import config
from network import R3D, SlowFast


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_video_dir', dest='in_video_dir', type=str, default=config.Path.inference_in_video_dir())
    parser.add_argument('--out_video_dir', dest='out_video_dir', type=str, default=config.Path.inference_out_video_dir())
    parser.add_argument('--out_img_dir', dest='out_img_dir', type=str, default=config.Path.inference_out_img_dir())
    parser.add_argument('--sample_frequency', dest='sample_frequency', type=int, default=4)
    parser.add_argument('--model', dest='model', type=str, default='SlowFast')
    parser.add_argument('--weights_path', dest='weights_path', type=str, default=config.Path.weights_path())
    _args = parser.parse_args()
    return _args


def center_crop(frame, size):
    h, w = np.shape(frame)[0: 2]
    rh, rw = size
    x = int(round((w - rw) / 2.))
    y = int(round((h - rh) / 2.))
    res_frame = frame[y: y + rh, x: x + rw, :]
    return np.array(res_frame).astype(np.uint8)


if __name__ == '__main__':
    args = arg_parse()
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    with open(os.path.join(config.Path.dataloader_dir(), 'ucf_labels.txt'), 'r') as f:
        class_names = f.readlines()
        f.close()

    print(class_names)

    if args.model == 'R3D':
        model = R3D.R3DModel(num_classes=101, layer_sizes=(2, 2, 2, 2))
    elif args.model == 'SlowFast':
        model = SlowFast.resnet50(class_num=101)

    weights = torch.load(args.weights_path)
    model.load_state_dict(weights['state_dict'])
    model.to(device)
    model.eval()
    torch.no_grad()

    out_img_dir = args.out_img_dir
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)

    out_video_dir = args.out_video_dir
    if not os.path.exists(out_video_dir):
        os.mkdir(out_video_dir)

    in_video_dir = args.in_video_dir

    for video in os.listdir(in_video_dir):
        _out_img_dir = os.path.join(out_img_dir, video)
        if not os.path.exists(_out_img_dir):
            os.mkdir(_out_img_dir)
        
        out_video = os.path.join(out_video_dir, video)
        video = os.path.join(in_video_dir, video)
        capture = cv2.VideoCapture(video)
        if not capture.isOpened():
            print('video: {} cannot use'.format(video))
        else:
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = int(capture.get(cv2.CAP_PROP_FPS))
            print('video: {}, width: {}, height: {}, fps: {}'.format(video, width, height, fps))

        retaining = True

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(out_video, fourcc, fps, (int(width), int(height)))
        clip = []
        frequency = args.sample_frequency
        i = 0

        while retaining:
            retaining, frame = capture.read()
            if not retaining or frame is None:
                continue
            i = i + 1
            if frame.shape[1] > 2000.0:
                font = 4.0
            elif frame.shape[1] > 1500.0:
                font = 2.0
            elif frame.shape[1] > 1000.0:
                font = 1.0
            else:
                font = 0.6

            if i % frequency != 0:
                continue

            resize_frame = cv2.resize(frame, (config.RESIZE_WIDTH, config.RESIZE_HEIGHT))
            _frame = center_crop(resize_frame, (config.CROP_SIZE, config.CROP_SIZE))
            clip.append(_frame - np.array([[[90.0, 98.0, 102.0]]]))

            if len(clip) == 16:
                images = np.array(clip).astype(np.float32)
                images = np.expand_dims(images, axis=0)
                images = np.transpose(images, (0, 4, 1, 2, 3))
                images = torch.from_numpy(images).to(device)
                outputs = model.forward(images)

                probs = torch.nn.Softmax(dim=1)(outputs)
                label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

                cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (50, int(50 * font)),
                            cv2.FONT_HERSHEY_SIMPLEX, font, (0, 255, 255), 2)

                cv2.putText(frame, 'prob: %.4f' % probs[0][label], (50, int(100 * font)),
                            cv2.FONT_HERSHEY_SIMPLEX, font, (0, 255, 255), 2)

                cv2.imwrite(os.path.join(_out_img_dir, str(i) + '.png'), frame)
                video_out.write(frame)
                clip.pop(0)

        capture.release()
        cv2.destroyAllWindows()

        print('one img is ok!')
