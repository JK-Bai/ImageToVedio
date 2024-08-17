import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Generator import Generator
from SimpleGenerator import SimpleGenerator
from Discriminator import Discriminator
from VedioDataset import VideoDataset
from torchvision import transforms

generator = SimpleGenerator()
discriminator = Discriminator()

# 优化器
lr_g = 0.00002
lr_d = 0.00002
beta1 = 0.5
optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, 0.999))

# 损失函数
criterion = nn.BCELoss()

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)
# 用于记录损失的列表
g_losses = []
d_losses = []

# 训练过程
num_epochs = 20
batch_size = 16

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

dataset = VideoDataset(video_dir='D:\ImageToVedio\Vedio', transform=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

def show_generated_image(image_tensor):
    # 将 [-1, 1] 的值范围转换为 [0, 1] 范围
    image = (image_tensor + 1) / 2
    np_image = image[0].cpu().detach().numpy().transpose(1, 2, 0)  # 不使用 squeeze(0)

    plt.imshow(np_image)
    plt.axis('off')
    plt.show()

# 训练循环中的调整
for epoch in range(num_epochs):
    for i, (real_videos) in enumerate(data_loader):
        # 数据准备
        real_videos = real_videos.to(device)

        for j in range(real_videos.size(1)):
            real_frame = real_videos[:, j, :, :, :]  # 获取第j帧
            if j != 0:
                last_frame = real_videos[:, j-1, :, :, :]
            else:
                last_frame = real_videos[:, j, :, :, :]

            # 生成随机噪声
            noise = torch.randn(batch_size, 3, 240, 320).to(last_frame.device)
            # 训练判别器
            optimizer_d.zero_grad()

            # 使用真实帧训练判别器
            outputs = discriminator(real_frame)
            real_labels = torch.ones_like(outputs).to(device)* 0.9
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()

            # 生成虚假视频帧
            fake_frame = generator(last_frame, noise)
            # 使用虚假帧训练判别器
            outputs = discriminator(fake_frame.detach())
            fake_labels = torch.zeros_like(outputs).to(device)
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()

            # 更新判别器
            optimizer_d.step()

            # 训练生成器
            optimizer_g.zero_grad()
            outputs = discriminator(fake_frame)
            real_labels = torch.ones_like(outputs).to(device)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()

            # 更新生成器
            optimizer_g.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss_real.item() + d_loss_fake.item())

        # 输出训练进度
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {d_loss_real + d_loss_fake:.4f}, Loss G: {g_loss:.4f}")

    # 每个 epoch 后保存模型
    torch.save(generator.state_dict(), f"generator_epoch_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch}.pth")

# 绘制损失曲线
def plot_loss(g_losses, d_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="G Loss")
    plt.plot(d_losses, label="D Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Generator and Discriminator Loss During Training")
    plt.show()

# 在训练结束后调用可视化函数
plot_loss(g_losses, d_losses)
