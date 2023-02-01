import os
import cv2
import shutil
from tqdm import tqdm


class VideoToFrames:
    def __init__(
        self,
        args=None,
        base_dir='/home/data/lwb/data',
        data_source_dir_name='AVM_record_ocr',
        save_dir_name='frames',
    ):
        self.videos_dir = os.path.join(base_dir, data_source_dir_name)
        videos_names = []
        self.vid_dir_name_sign = '2022'
        for vid_name in os.listdir(self.videos_dir):
            if self.vid_dir_name_sign in vid_name:
                videos_names.append(vid_name)
        self.videos_names = sorted(videos_names)
        self.camera_list = ['front', 'back', 'left', 'right']
        self.frames_total_dir = os.path.join(self.videos_dir, save_dir_name)
        if os.path.exists(self.frames_total_dir):
            shutil.rmtree(self.frames_total_dir)
        self.img_type = '.jpg'
        self.video_type = '.mp4'

    def extract_frames_kernel(self, video_path, save_dir, img_type, frames_len):
        cap = cv2.VideoCapture(video_path)
        # video_frames_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        # frames_len = min(frames_len, video_frames_len)
        for k in tqdm(range(frames_len)):
            _, img = cap.read()
            save_path = os.path.join(save_dir, f'{k+1:05d}' + img_type)
            try:
                cv2.imwrite(save_path, img)
            except:
                print(f'image shape:   {img.shape}')
                print(f'imwrite error: {save_path}')
                print(f'frames_length: {frames_len}')
        cap.release()

    def vs_read_video():
        import cv2
        import time
        import torch
        from torchvision.io import read_video

        video_path = '/home/data/lwb/data/AVM_record_ocr/20221205141455/front.mp4'
        cap = cv2.VideoCapture(video_path)
        frames_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        f1 = []
        t1 = time.time()
        for k in range(frames_len):
            _, frame = cap.read()
            f1.append(torch.from_numpy(frame))
        f1 = torch.stack(f1)
        # f1 = f1.cuda(6)
        t2 = time.time()
        f2, _, fps = read_video(str(video_path))
        # f2 = f2.cuda(7)
        t3 = time.time()
        assert f1.shape == f2.shape
        print(t2 - t1)
        print(t3 - t2)

    def extract_frames(self):
        print(f'videos_dir: {self.videos_dir}\n')
        for idx, vid_name in tqdm(enumerate(self.videos_names)):
            vid_frames_dir = os.path.join(self.frames_total_dir, vid_name)
            os.makedirs(vid_frames_dir, exist_ok=True)
            cameras_frames_len_list = []
            for cam in self.camera_list:
                vid_cam_path = os.path.join(
                    self.videos_dir, vid_name, cam + self.video_type
                )
                cap = cv2.VideoCapture(vid_cam_path)
                frames_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
                cameras_frames_len_list.append(frames_len)
                cap.release()
            frames_len = min(cameras_frames_len_list)
            prt_str = f'\n  vid: [{idx+1:02}/{len(self.videos_names)}]'
            prt_str += f', vid_name: {vid_name}, frames: {frames_len}'
            print(prt_str)
            for cam in self.camera_list:
                frames_sv_dir = os.path.join(vid_frames_dir, cam)
                os.makedirs(frames_sv_dir, exist_ok=True)
                vid_cam_path = os.path.join(
                    self.videos_dir, vid_name, cam + self.video_type
                )
                assert os.path.exists(vid_cam_path), f'vid_cam_path: {vid_cam_path}'
                cap = cv2.VideoCapture(vid_cam_path)
                # extract frames
                print(f'\t\t\t{cam}')
                for k in range(frames_len):
                    _, frame = cap.read()
                    assert len(frame.shape) == 3, f'image shape: {frame.shape}'
                    save_path = os.path.join(
                        frames_sv_dir, f'{k+1:05d}' + self.img_type
                    )
                    # ipdb.set_trace()
                    cv2.imwrite(save_path, frame)
                    if not os.path.exists(save_path):
                        print(f'imwrite error: {save_path}')
                        print(f'frames_length: {frames_len}')
                        raise ValueError
                cap.release()

        return


if __name__ == "__main__":

    frames_handle = VideoToFrames()
    frames_handle.extract_frames()
