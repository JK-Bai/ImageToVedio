import os
import cv2
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None, target_frames=50):
        self.video_dir = video_dir
        self.transform = transform
        self.target_frames = target_frames
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        # 加载视频
        video_name = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_name)
        video = self.load_video(video_path)

        return video

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return torch.tensor([])  # 返回一个空张量以避免后续错误

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)  # 对每一帧应用变换
            frames.append(frame)
        cap.release()

        if len(frames) > self.target_frames:
            frames = frames[:self.target_frames]  # 裁剪到 target_frames 帧
        elif len(frames) < self.target_frames:
            padding = [torch.zeros_like(frames[0]) for _ in range(self.target_frames - len(frames))]
            frames.extend(padding)  # 填充黑帧

        if len(frames) > 0:
            video_tensor = torch.stack(frames)  # 将帧列表转换为张量
        else:
            video_tensor = torch.tensor([])  # 如果没有帧，返回空张量

        return video_tensor
