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

from ..datasets.transformer_dataset import TrainDataset, InferenceDataset

app = typer.Typer()

def get_logits(src, tgt, masks, model, segnet_model):
    src_flat = rearrange(src, 'b frames c h w -> (b frames) c h w', frames=11)
    tgt_flat = rearrange(tgt, 'b frames c h w -> (b frames) c h w', frames=11)

    src_encoded_flat, src_indx, src_sizes = segnet_model.encode(src_flat)
    tgt_encoded_flat, _, _ = segnet_model.encode(tgt_flat)

    tgt_decoded_flat = rearrange(masks, 'b frames h w -> (b frames) h w', frames=11)

    torch.nn.functional.relu(tgt_decoded_flat, inplace=True)

    src_encoded = rearrange(src_encoded_flat, '(b frames) c h w -> b frames (c h w)', frames=11)
    tgt_encoded = rearrange(tgt_encoded_flat, '(b frames) c h w -> b frames (c h w)', frames=11)
    tgt_encoded = torch.cat((src_encoded[:, -1:, :], tgt_encoded), dim = 1)

    src_mask, tgt_mask = create_mask()

    logits = model(src_encoded, tgt_encoded, src_mask, tgt_mask)
    logits = logits[:, 1:, :]

    logits_flat = rearrange(logits, 'b frames (c h w) -> (b frames) c h w', frames=11, c=64, h=2, w=3)
    logits_decoded_flat = segnet_model.decode(logits_flat, src_indx, src_sizes)

    return logits_decoded_flat, tgt_decoded_flat

def train_epoch(model, segnet_model, criterion, optimizer, train_dataloader, device=None):
    model.train()
    losses = 0

    for idx, (src, tgt, masks) in enumerate(tqdm(train_dataloader)):
        src = src.to(device)
        tgt = tgt.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        loss = criterion(*get_logits(src, tgt, masks, model, segnet_model))
        loss.backward()
        optimizer.step()
        loss_item = loss.item()
        losses += loss_item

        if idx > 0 and idx % 50 == 0:
          print(f"step={idx} train_loss={loss_item}")


    return losses / len(train_dataloader)

def evaluate(model, segnet_model, criterion, val_dataloader, device=None):
    model.eval()
    losses = 0

    for idx, (src, tgt, masks) in enumerate(tqdm(val_dataloader)):
        src = src.to(device)
        tgt = tgt.to(device)
        masks = masks.to(device)

        loss = criterion(*get_logits(src, tgt, masks, model, segnet_model))

        loss_item = loss.item()
        losses += loss_item

        if idx > 0 and idx % 50 == 0:
          print(f"step={idx} val_loss={loss_item}")

    return losses / len(val_dataloader)

def greedy_decode(src, segnet_model, transformer_model, device=None):
    src = src.to(device)
    src_mask = torch.zeros((11, 11), device=device).type(torch.bool)
    src_mask = src_mask.to(device=device)

    src_encoded, src_indx, src_sizes = segnet_model.encode(src)
    src_encoded = rearrange(src_encoded, 'frames c h w -> frames (c h w)')

    memory = transformer_model.encode(src_encoded.unsqueeze(1), src_mask)
    memory = memory.to(device)

    ys = src_encoded[-1:, :].unsqueeze(1)
    for i in range(12):
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device).type(torch.bool)).to(device)
        out = transformer_model.decode(ys[-11:], memory, tgt_mask[-11:, -11:])
        out = out.transpose(0, 1)

        prob = transformer_model.generator(out[:, -1])

        ys = torch.cat([ys.squeeze(1), prob], dim=0).unsqueeze(1)

    to_decode = rearrange(ys.squeeze(1)[-11:], 'frames (c h w) -> frames c h w', c=64, h=2, w=3)
    decoded = segnet_model.decode(to_decode, src_indx, src_sizes)
    _, idxs = torch.topk(decoded, 1, dim=1)

    return idxs.squeeze(1).cpu()

@app.command()
def train(
    segnet_checkpoint : Optional[Path] = None,
    transformer_checkpoint : Optional[Path] = None,
    output_dir : Optional[Path] = Path("./"),
    dataset_dir : Optional[Path] = Path("./"),
    enc_dim : Optional[int] = 64*2*3,
    num_heads : Optional[int] = 8,
    hidden_dim : Optional[int] = 512,
    enc_layers : Optional[int] = 2,
    dec_layers : Optional[int] = 2,
    num_classes : Optional[int] = 49,
    batch_size : Optional[int] = 3,
    epochs :  Optional[int] = 30,
):
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        device_0 = torch.device("cuda")
        device_1 = torch.device("cuda:1")
    else:
        device_0 = torch.device("cpu")
        device_1 = torch.device("cpu")

    segnet_model = SegNet(num_classes)
    labeler_model = SegNet(num_classes)

    if segnet_checkpoint is not None:
        checkpoint = torch.load(segnet_checkpoint)
        segnet_model.load_state_dict(checkpoint["model"])
        labeler_model.load_state_dict(checkpoint["model"])

    segnet_model = segnet_model.to(device_0)
    labeler_model = labeler_model.to(device_1)
    labeler_model.eval()

    transformer = Seq2SeqTransformer(
        enc_layers,
        dec_layers,
        enc_dim,
        num_heads,
        enc_dim,
        hidden_dim
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(transformer.parameters()) + list(segnet_model.parameters()), 
        lr=0.0001,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    if transformer_checkpoint is not None:
        checkpoint = torch.load(transformer_checkpoint)
        
        segnet_model.load_state_dict(checkpoint["segnet_state_dict"])
        transformer.load_state_dict(checkpoint["segnet_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        best_val_loss = checkpoint["best_val_loss"]
        curr_epoch = checkpoint["epoch"]

    else: 
        best_val_loss = float('inf')
        curr_epoch = 0
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    transformer = transformer.to(device_0)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = TrainDataset( dataset_dir / "unlabeled",  dataset_dir / "train", labeler_model, transform=transform)
    val_dataset   = TrainDataset(None, dataset_dir/"val", None, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    val_losses = []
    train_losses = []


    for epoch in tqdm(range(curr_epoch, curr_epoch+epochs)):
        print("="*50, "train step", "=" *50)
        train_loss = train_epoch(transformer, segnet_model, criterion, optimizer, train_dataloader, device=device_0)
        print("="*50, "val step", "=" *50)
        val_loss = evaluate(transformer, segnet_model, criterion, val_dataloader, device=device_0)
        

        val_losses.append(val_loss)
        train_losses.append(train_loss)

        print(f"train_loss={train_loss}, val_loss={val_loss}, epoch={epoch}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'segnet_state_dict': segnet_model.state_dict(),
                'transformer_state_dict': transformer.state_dict(),
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
    transformer_checkpoint : Optional[Path] = None,
    output_dir : Optional[Path] = Path("./"),
    dataset_dir : Optional[Path] = Path("./"),
    enc_dim : Optional[int] = 64*2*3,
    num_heads : Optional[int] = 8,
    hidden_dim : Optional[int] = 512,
    enc_layers : Optional[int] = 2,
    dec_layers : Optional[int] = 2,
    num_classes : Optional[int] = 49,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    segnet_model = SegNet(num_classes)
    segnet_model = segnet_model.to(device)

    transformer = Seq2SeqTransformer(
        enc_layers,
        dec_layers,
        enc_dim,
        num_heads,
        enc_dim,
        hidden_dim
    )

    checkpoint = torch.load(transformer_checkpoint)
    
    segnet_model.load_state_dict(checkpoint["segnet_state_dict"])
    transformer.load_state_dict(checkpoint["transformer_state_dict"])

    transformer = transformer.to(device)
    segnet_model = segnet_model.to(device)

    transformer.eval()
    segnet_model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = InferenceDataset(f"{dataset_dir}/unlabeled", transform=transform)

    pred_tensor = torch.zeros((len(test_dataset), 160, 240))
    for idx, src in enumerate(tqdm(test_dataset)):
        out = greedy_decode(src, segnet_model, transformer, device=device)
        pred_tensor[idx, :, :] = out[-1]

    np.save(output_dir / "energy_based_mokey.npy", pred_tensor.numpy())


def main():
    app()