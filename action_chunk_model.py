from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data_chunk"
MODEL_DIR = SCRIPT_DIR / "skill"

IMAGE_SIZE = (224, 224)
JOINT_DIM = 6
CHUNK_SIZE = 16  # 每次生成的连续帧数 K

BATCH_SIZE = 16
NUM_EPOCHS = 2000
LEARNING_RATE = 1e-3
SAVE_LOSS_THRESHOLD = 10.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class ActionChunkCNN(nn.Module):
    """图像 -> (CHUNK_SIZE, JOINT_DIM) 连续帧动作 chunk。"""

    def __init__(self, chunk_size=CHUNK_SIZE, joint_dim=JOINT_DIM):
        super().__init__()
        self.chunk_size = chunk_size
        self.joint_dim = joint_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, chunk_size * joint_dim),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x.view(x.size(0), self.chunk_size, self.joint_dim)


def parse_joint_angles(image_path):
    parts = image_path.stem.split("_")
    if len(parts) < JOINT_DIM:
        raise ValueError(f"图片文件名中关节角数量不足: {image_path.name}")

    # 关节角取文件名最后 JOINT_DIM 个可解析数字，兼容带帧序号前缀的命名。
    numeric_parts = []
    for part in parts:
        try:
            numeric_parts.append(float(part))
        except ValueError:
            numeric_parts.append(None)

    joints = [value for value in numeric_parts if value is not None]
    if len(joints) < JOINT_DIM:
        raise ValueError(f"图片文件名无法解析关节角: {image_path.name}")

    return joints[-JOINT_DIM:]


def load_image_tensor(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"图片无法读取: {image_path}")
    return preprocess_image(image)


def preprocess_image(frame_bgr):
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, IMAGE_SIZE)
    image_transposed = image_resized.transpose((2, 0, 1))
    return torch.FloatTensor(image_transposed) / 255.0


def _discover_demos(skill_dir):
    skill_dir = Path(skill_dir)
    subdirs = sorted(path for path in skill_dir.iterdir() if path.is_dir())
    demos = []
    for subdir in subdirs:
        if any(p.suffix.lower() in IMAGE_EXTENSIONS for p in subdir.iterdir() if p.is_file()):
            demos.append(subdir)

    if not demos:
        has_images = any(
            p.suffix.lower() in IMAGE_EXTENSIONS for p in skill_dir.iterdir() if p.is_file()
        )
        if has_images:
            demos = [skill_dir]
    return demos


def _ordered_frames(demo_dir):
    return sorted(
        path
        for path in Path(demo_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class ActionChunkDataset(Dataset):
    """从按帧排序的示教目录里构造 (图像, 未来 K 帧关节角) 样本。"""

    def __init__(self, skill_dir, chunk_size=CHUNK_SIZE):
        self.chunk_size = chunk_size
        demos = _discover_demos(skill_dir)
        if not demos:
            raise FileNotFoundError(f"没有找到示教数据: {skill_dir}")

        self.samples = []
        for demo in demos:
            frames = _ordered_frames(demo)
            if len(frames) < 2:
                continue
            joints = [parse_joint_angles(frame) for frame in frames]
            for start in range(len(frames)):
                chunk = joints[start : start + chunk_size]
                if not chunk:
                    continue
                # 末尾不足 K 帧时用最后一帧补齐，让模型学会「保持静止 = 结束」。
                while len(chunk) < chunk_size:
                    chunk.append(chunk[-1])
                self.samples.append((frames[start], chunk))

        if not self.samples:
            raise FileNotFoundError(f"示教帧数量不足，无法构造 chunk: {skill_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, chunk = self.samples[idx]
        image = load_image_tensor(image_path)
        target = torch.FloatTensor(chunk)
        return image, target


def predict_chunk(model, device, frame_bgr):
    """对单帧相机图像生成一段 (CHUNK_SIZE, JOINT_DIM) 的连续帧动作。"""
    image_tensor = preprocess_image(frame_bgr).unsqueeze(0).to(device)
    with torch.no_grad():
        chunk = model(image_tensor).cpu().numpy()[0]
    return chunk.tolist()


def train_skill(skill_dir, output_path, chunk_size=CHUNK_SIZE):
    skill_dir = Path(skill_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ActionChunkDataset(skill_dir, chunk_size=chunk_size)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ActionChunkCNN(chunk_size=chunk_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"使用设备: {device}")
    print(f"示教目录: {skill_dir}")
    print(f"chunk 样本数: {len(dataset)}，chunk 大小: {chunk_size}")

    best_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predicted = model(images)
            loss = criterion(predicted, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)

        avg_loss = epoch_loss / len(dataset)
        if avg_loss < best_loss:
            best_loss = avg_loss
            if avg_loss < SAVE_LOSS_THRESHOLD:
                torch.save(model.state_dict(), output_path)
                print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f} (saved -> {output_path})")
                continue
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")

    print(f"训练完成，最佳 Loss: {best_loss:.6f}")
    if best_loss >= SAVE_LOSS_THRESHOLD:
        print(f"注意：最佳 Loss 仍 >= {SAVE_LOSS_THRESHOLD}，没有保存模型。")


def train_all():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"训练数据目录不存在: {DATA_DIR}\n"
            "请按 data_chunk/<技能>/<示教>/<帧序号>_<6个关节角>.jpg 的格式准备数据。"
        )

    skill_dirs = sorted(path for path in DATA_DIR.iterdir() if path.is_dir())
    if not skill_dirs:
        raise FileNotFoundError(f"{DATA_DIR} 下没有技能子目录。")

    for skill_dir in skill_dirs:
        output_path = MODEL_DIR / f"{skill_dir.name}.pth"
        print(f"\n===== 训练技能: {skill_dir.name} -> {output_path} =====")
        train_skill(skill_dir, output_path)


if __name__ == "__main__":
    train_all()
