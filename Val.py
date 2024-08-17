import torch
from Generator import Generator  # 假设Generator类在Generator.py文件中
import cv2
import numpy as np
from SimpleGenerator import SimpleGenerator

# 指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化生成器模型
generator = Generator().to(device)
print("Before loading weights:")
for name, param in generator.named_parameters():
    if "weight" in name:
        print(f"{name}: {param[0][0][:5]}")  # 打印第一个权重的前五个值
        break
# 加载模型权重
generator.load_state_dict(torch.load("generator_epoch_199.pth", map_location=device, weights_only=True))
print("\nAfter loading weights:")
for name, param in generator.named_parameters():
    if "weight" in name:
        print(f"{name}: {param[0][0][:5]}")  # 打印相同位置的前五个值
        break
# 切换模型到评估模式
generator.eval()

# 输入的测试图像
test_image_path = 'D:/ImageToVedio/Image/v_ApplyEyeMakeup_g01_c01_first_frame.png'  # 替换为你自己的测试图像路径
test_image = cv2.imread(test_image_path)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
test_image = cv2.resize(test_image, (320, 240))  # 调整大小到模型输入大小

# 转换为Tensor并放入模型中
test_image_tensor = torch.from_numpy(test_image).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
test_image_tensor = (test_image_tensor - 0.5) / 0.5  # 标准化到[-1, 1]

# 生成噪声向量
noise = torch.randn_like(test_image_tensor).to(device)

# 生成新图像
with torch.no_grad():
    generated_image_tensor = generator(test_image_tensor,noise)

# 将生成的图像转换回原始图像格式
generated_image_tensor = (generated_image_tensor * 0.5 + 0.5) * 255.0  # 反向标准化
generated_image_tensor = generated_image_tensor.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)

# 显示或保存生成的图像
generated_image_bgr = cv2.cvtColor(generated_image_tensor, cv2.COLOR_RGB2BGR)
cv2.imshow("Generated Image", generated_image_bgr)
cv2.imwrite("generated_image.png", generated_image_bgr)  # 保存生成的图像
cv2.waitKey(0)
cv2.destroyAllWindows()
