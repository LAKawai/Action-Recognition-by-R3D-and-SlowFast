import os


class Path(object):
    @staticmethod
    def project_dir():
        return '/root/autodl-tmp'

    @staticmethod
    def dataset_dir(dataset):
        if dataset == 'ucf101':
            root_dir = os.path.join(Path.project_dir(), 'UCF_101')
            output_dir = os.path.join(Path.project_dir(), 'UCF_res')
            return root_dir, output_dir

        elif dataset == 'hmdb51':
            root_dir = os.path.join(Path.project_dir(), 'hmdb_51')
            output_dir = os.path.join(Path.project_dir(), 'hmdb_res')
            return root_dir, output_dir

        else:
            print('not find dataset')
            raise NotImplementedError

    @staticmethod
    def dataloader_dir():
        return os.path.join(Path.project_dir(), 'dataloaders')

    @staticmethod
    def inference_in_video_dir():
        return os.path.join(Path.project_dir(), 'videos')

    @staticmethod
    def inference_out_img_dir():
        return os.path.join(Path.project_dir(), 'result_img')

    @staticmethod
    def inference_out_video_dir():
        return os.path.join(Path.project_dir(), 'result_videos')

    @staticmethod
    def weights_path():
        return os.path.join(Path.project_dir(), 'checkpoint/SlowFast-ucf101_epoch_49.pth.tar')

    @staticmethod
    def save_dir_root():
        return os.path.join(Path.project_dir(), 'run_model')


RESIZE_HEIGHT = 128
RESIZE_WIDTH = 171
CROP_SIZE = 112

EXPANSION = 4
