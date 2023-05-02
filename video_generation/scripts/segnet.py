from typing import Optional
from pathlib import Path

import typer
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np

from torchvision import transforms
from tqdm import tqdm
from einops import rearrange

from .utils import create_mask, generate_square_subsequent_mask
from ..models.segnet import SegNet
from ..models.transformer import Seq2SeqTransformer

from ..datasets.segnet_dataset import SegmentationDataset

app = typer.Typer()

def train_epoch(model, criterion, optimizer, train_dataloader, device=None):
    model.train()
    losses = 0

    for (img_batch, mask_batch) in tqdm(train_dataloader):
        optimizer.zero_grad()
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        y = model(img_batch)
        loss = criterion(y, mask_batch)

        losses += loss.item()
        loss.backward()
        optimizer.step()

    return losses / len(train_dataloader)

def evaluate(model, criterion, val_dataloader, device=None):
    model.eval()
    losses = 0

    for (img_batch, mask_batch) in tqdm(val_dataloader):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        y = model(img_batch)
        loss = criterion(y, mask_batch)
        losses += loss.item()

    return losses / len(val_dataloader)

@app.command()
def train(
    segnet_checkpoint : Optional[Path] = None,
    output_dir : Optional[Path] = Path("./"),
    dataset_dir : Optional[Path] = Path("./"),
    num_classes : Optional[int] = 49,
    batch_size : Optional[int] = 3,
    epochs :  Optional[int] = 30,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = SegNet(num_classes)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    if segnet_checkpoint is not None:
        checkpoint = torch.load(segnet_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        best_val_loss = checkpoint["best_val_loss"]
        curr_epoch = checkpoint["epoch"]

    else: 
        best_val_loss = float('inf')
        curr_epoch = 0

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = SegmentationDataset(dataset_dir / "train", transform=transform)
    val_dataset   = SegmentationDataset(dataset_dir/"val", transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    val_losses = []
    train_losses = []


    for epoch in tqdm(range(curr_epoch, curr_epoch+epochs)):
        print("="*50, "train step", "=" *50)
        train_loss = train_epoch(model, criterion, optimizer, train_dataloader, device=device)
        print("="*50, "val step", "=" *50)
        val_loss = evaluate(model, criterion, val_dataloader, device=device)
        

        val_losses.append(val_loss)
        train_losses.append(train_loss)

        print(f"train_loss={train_loss}, val_loss={val_loss}, epoch={epoch}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, output_dir/f"checkpoint_{epoch}.pt")


    plt.figure()
    plt.plot(train_losses)
    plt.savefig(output_dir/f"final_train_loss.png")

    plt.figure()
    plt.plot(val_losses)
    plt.savefig(output_dir/f"final_val_loss.png")

@app.command()
def inference(
    segnet_checkpoint : Optional[Path] = None,
    output_dir : Optional[Path] = Path("./"),
    dataset_dir : Optional[Path] = Path("./"),
    num_classes : Optional[int] = 49,
    batch_size : Optional[int] = 100
):
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = SegNet(num_classes)
    model = model.to(device)


    if segnet_checkpoint is not None:
        checkpoint = torch.load(segnet_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset   = SegmentationDataset(dataset_dir/"hidden", transform=transform)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    pred_tensor = torch.zeros((len(test_dataloader), 160, 240))
    for idx, src in enumerate(tqdm(test_dataloader)):
        out = model(src)
        pred_tensor[idx, :, :] = out[-1]

    np.save(output_dir / "energy_based_mokey.npy", pred_tensor.numpy()) 


def main():
    app()