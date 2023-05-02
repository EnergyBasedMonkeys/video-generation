import torch
import os
import imageio.v3 as iio
import numpy as np

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, unlabeled_dir, labeled_dir, segnet_model, transform=None, device=None):
        self.segnet_model = segnet_model if segnet_model is not None else None
        self.transform = transform
        self.device = device

        self.unlabeled_vids = [os.path.join(unlabeled_dir, i) for i in os.listdir(unlabeled_dir)] if unlabeled_dir is not None else []
        self.labeled_vids = [os.path.join(labeled_dir, i) for i in os.listdir(labeled_dir)]


        self.idx_boundry = len(self.labeled_vids)

    def __len__(self):
        return len(self.unlabeled_vids) + len(self.labeled_vids)

    def __getitem__(self, idx):
        if idx < self.idx_boundry:
          vid_dir = self.labeled_vids[idx]
        else:
          vid_dir = self.unlabeled_vids[idx-self.idx_boundry]
        src_frames = []
        tgt_frames = []
        
        for i in range(11):
          image = iio.imread(os.path.join(vid_dir, f"image_{i}.png"))
          src_frames.append(self.transform(image))

        for i in range(11, 22):
          image = iio.imread(os.path.join(vid_dir, f"image_{i}.png"))
          tgt_frames.append(self.transform(image))
        
        src_frames =  torch.stack(src_frames, 0)
        tgt_frames =  torch.stack(tgt_frames, 0)

        if idx < self.idx_boundry:
          tgt_segmentations = torch.from_numpy(np.load(os.path.join(vid_dir, "mask.npy")).astype(int))[11:]
        elif self.segnet_model is not None:
          tgt_segmentations = []
          for i in range(0, 11, 5):
            segnet_input = tgt_frames[i: i+5].to(self.device)
            segnet_out = self.segnet_model(segnet_input)
            _, idxs = torch.topk(segnet_out, 3, dim=1)
            idxs = idxs.cpu()
            tgt_segmentations.append(idxs[:, 0, :, :])

            del segnet_input
            del y

          tgt_segmentations = torch.cat(tgt_segmentations, 0)
        else:
          tgt_segmentations = []

        return src_frames, tgt_frames, tgt_segmentations

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, unlabeled_dir, transform=None):
        self.transform = transform

        self.unlabeled_vids = [os.path.join(unlabeled_dir, i) for i in os.listdir(unlabeled_dir)] if unlabeled_dir is not None else []

    def __len__(self):
        return len(self.unlabeled_vids)

    def __getitem__(self, idx):
        vid_dir = self.unlabeled_vids[idx]
        src_frames = []
        
        for i in range(11):
          image = iio.imread(os.path.join(vid_dir, f"image_{i}.png"))
          src_frames.append(self.transform(image))

        return torch.stack(src_frames, 0)