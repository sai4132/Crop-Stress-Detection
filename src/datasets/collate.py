import torch

def collate_fn(batch):
    collated_batch = dict()
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            collated_batch[key] = torch.stack([sample[key] for sample in batch])
        else:
            collated_batch[key] = [sample[key] for sample in batch]
    return collated_batch