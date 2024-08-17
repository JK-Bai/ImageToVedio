import cv2
import os


def split_video(input_path, output_path, segment_duration=2):
    # 创建输出路径
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历输入路径下的所有文件
    for video_file in os.listdir(input_path):
        if not video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # 只处理视频文件
            continue

        video_path = os.path.join(input_path, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"无法打开视频文件: {video_file}")
            continue

        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # 视频帧率
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
        video_duration = frame_count / fps  # 视频时长（秒）

        segment_frames = int(segment_duration * fps)  # 每段视频的帧数
        success, frame = cap.read()
        segment_index = 0

        while success:
            segment_index += 1
            output_file = os.path.join(output_path, f"{os.path.splitext(video_file)[0]}_segment_{segment_index}.mp4")
            out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))

            frame_idx = 0
            while success and frame_idx < segment_frames:
                out.write(frame)
                success, frame = cap.read()
                frame_idx += 1

            out.release()

        cap.release()

        # 处理不足两秒的剩余部分
        if video_duration % segment_duration != 0:
            segment_index += 1
            output_file = os.path.join(output_path, f"{os.path.splitext(video_file)[0]}_segment_{segment_index}.mp4")
            out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))
            remaining_frames = int((video_duration % segment_duration) * fps)

            for _ in range(remaining_frames):
                out.write(frame)
                success, frame = cap.read()

            out.release()


if __name__ == "__main__":
    input_video_directory = "D:/ImageToVedio/ucf/JumpingJack"  # 替换为你的输入路径
    output_directory = "D:/ImageToVedio/Vedio"  # 替换为你的输出路径
    segment_duration = 2  # 设定每段视频的时间（秒）

    split_video(input_video_directory, output_directory, segment_duration)

