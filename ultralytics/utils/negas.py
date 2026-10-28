"""Fixed NegAS mask construction from Sec. 3.2 of the paper."""
import torch


def background_mask(features, strides, boxes, valid):
    """Return B x sum(H_l*W_l) background cells in feature flatten order.

    Boxes are padded pixel-space xyxy GT. Any cell overlapped by a box is
    conservatively foreground; padding is excluded by ``valid``.
    """
    masks = []
    for feature, stride in zip(features, strides):
        height, width = feature.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(height, device=boxes.device),
                                torch.arange(width, device=boxes.device), indexing="ij")
        grid = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
        lower = torch.floor(boxes[..., :2] / stride)
        upper = torch.ceil(boxes[..., 2:] / stride)
        per_image = []
        for image in range(boxes.shape[0]):
            inside = ((grid[None] >= lower[image, :, None]) &
                      (grid[None] < upper[image, :, None])).all(dim=-1)
            foreground = (inside & valid[image].reshape(-1, 1).bool()).any(dim=0)
            per_image.append(~foreground)
        masks.append(torch.stack(per_image))
    return torch.cat(masks, dim=1)
