import os
import cv2


def remove_corrupted_videos(directory):
    # 遍历指定目录下的所有文件
    for video_file in os.listdir(directory):
        video_path = os.path.join(directory, video_file)

        # 检查文件是否为视频文件
        if not video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        # 尝试打开视频文件
        cap = cv2.VideoCapture(video_path)

        # 如果无法打开视频，删除该文件
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_file}, 正在删除...")
            os.remove(video_path)

        # 释放视频捕获对象
        cap.release()


if __name__ == "__main__":
    video_directory = "D:/ImageToVedio/Vedio"  # 替换为你的视频路径
    remove_corrupted_videos(video_directory)
