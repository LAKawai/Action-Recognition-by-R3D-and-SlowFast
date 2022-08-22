import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import config


class VideoDataset(Dataset):
    def __init__(self, dataset='ucf101', split='train', clip_len=16, preprocess=False):
        self.root_dir, self.output_dir = config.Path.dataset_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split
        self.resize_height = config.RESIZE_HEIGHT
        self.resize_width = config.RESIZE_WIDTH
        self.crop_size = config.CROP_SIZE

        if not self.check_preprocess() or preprocess:
            self.preprocess()

        self.filenames, labels = [], []
        for video_class in sorted(os.listdir(folder)):
            for video_filename in os.listdir(os.path.join(folder, video_class)):
                self.filenames.append(os.path.join(folder, video_class, video_filename))
                labels.append(video_class)

        assert len(labels) == len(self.filenames)
        print('Number of {} videos: {:d}'.format(split, len(self.filenames)))

        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == 'ucf101':
            if not os.path.exists(os.path.join(config.Path.dataloader_dir(), 'ucf_labels.txt')):
                with open(os.path.join(config.Path.dataloader_dir(), 'ucf_labels.txt'), 'w') as f:
                    for i, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(i + 1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        buffer = self.load_frames(self.filenames[index])
        buffer = self.crop_frame(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            buffer = self.random_flip(buffer)

        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        buffer = buffer.transpose((3, 0, 1, 2))

        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_preprocess(self):
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for i, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video_filename in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                frames_path = os.path.join(self.output_dir, 'train', video_class, video_filename)
                single_frame = os.path.join(frames_path, sorted(os.listdir(frames_path))[0])
                image = cv2.imread(single_frame)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                    return False
                else:
                    break
            if i == 10:
                break

        return True

    def preprocess(self):
        print('output_dir is:', str(self.output_dir))
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'valid'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        for video_class in os.listdir(self.root_dir):
            video_class_dir = os.path.join(self.root_dir, video_class)
            files = [file for file in os.listdir(video_class_dir)]
            train_valid, test = train_test_split(files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', video_class)
            val_dir = os.path.join(self.output_dir, 'valid', video_class)
            test_dir = os.path.join(self.output_dir, 'test', video_class)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, video_class, train_dir)

            for video in val:
                self.process_video(video, video_class, val_dir)

            for video in test:
                self.process_video(video, video_class, test_dir)

        print('Preprocessing finished!')

    def process_video(self, video, video_class, save_dir):
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, video_class, video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        extract_frequency = 4
        if frame_count // extract_frequency <= 16:
            extract_frequency -= 1
            if frame_count // extract_frequency <= 16:
                extract_frequency -= 1
                if frame_count // extract_frequency <= 16:
                    extract_frequency -= 1

        frame_cnt, i = 0, 0
        retaining = True

        while frame_cnt < frame_count and retaining:
            retaining, frame = capture.read()
            if frame is None:
                continue
            if frame_cnt % extract_frequency == 0:
                if frame_height != self.resize_height or frame_width != self.resize_width:
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            frame_cnt += 1

        capture.release()

    def load_frames(self, video_filename):
        frames = sorted([os.path.join(video_filename, img) for img in os.listdir(video_filename)])
        frames_count = len(frames)
        buffer = np.empty((frames_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame in enumerate(frames):
            img = np.array(cv2.imread(frame)).astype(np.float64)
            buffer[i] = img
        return buffer

    def crop_frame(self, buffer, clip_len, crop_size):
        if buffer.shape[0] <= clip_len:
            print('not enough frames')
            time_index = 0
        else:
            time_index = np.random.randint(buffer.shape[0] - clip_len)

        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        buffer = buffer[time_index: time_index + clip_len,
                        height_index: height_index + crop_size,
                        width_index: width_index + crop_size, :]

        return buffer

    def random_flip(self, buffer):
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)
        return buffer


if __name__ == '__main__':
    test_data = VideoDataset(dataset='ucf101', split='test', clip_len=8, preprocess=False)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=True, num_workers=4)
    # print(test_data)
    # print(test_loader.dataset)
    # print(len(test_loader.dataset))
    # print(len(test_data))
    for i, sample in enumerate(test_loader):
        img = sample[0]
        label = sample[1]
        print(img.shape)
        print(label)
        print(label.data)
        if i == 1:
            break
