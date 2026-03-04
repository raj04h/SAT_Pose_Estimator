import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loader import satellitePose
from pose_model import poseNet


json_path = r"D:\Data centr\IMG_data\satellite_pose\speed\train.json"

image_path = r"D:\Data centr\IMG_data\satellite_pose\speed\images\train"


# create dataset
dataset = satellitePose(json_path, image_path)


# create dataloader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


trained_model = poseNet()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = trained_model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epochs = 20


# pose loss estimation
def pose_loss(pred, gt):

    q_pred = pred[:, :4]
    t_pred = pred[:, 4:]

    q_gt = gt[:, :4]
    t_gt = gt[:, 4:]

    # rotation loss using quaternion dot product
    rot_loss = 1 - torch.abs(torch.sum(q_pred * q_gt, dim=1))

    # translation loss using L2 distance
    trans_loss = torch.norm(t_pred - t_gt, dim=1)

    loss = rot_loss + trans_loss

    return loss.mean()


# training loop
for epoch in range(epochs):

    total_loss = 0

    for images, poses in loader:

        # move data to device
        images = images.to(device)
        poses = poses.to(device)

        # forward pass
        preds = model(images)

        # compute loss
        loss = pose_loss(preds, poses)

        # reset gradients
        optimizer.zero_grad()

        # backpropagation
        loss.backward()

        # update weights
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f}")


torch.save(model.state_dict(), "SAT_Pose_model.pth")

print("Model saved")