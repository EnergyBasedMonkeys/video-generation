import torch

def generate_square_subsequent_mask(sz, device=None):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src_seq_len=11, tgt_seq_len=12, device=None):
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    return src_mask, tgt_mask