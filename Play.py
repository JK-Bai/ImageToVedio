import torch
import cv2
from Generator import output_frames

output_frames = output_frames.squeeze(0)
output_frames = output_frames.permute(0,2,3,1)
output_frames = output_frames.detach().numpy()

for frame in output_frames:
    frame = (frame * 255).astype('uint8')
    cv2.imshow('Video', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()