import torch
import torch.nn.functional as F

def roi_pool(input, rois, size=(7, 7), spatial_scale=1.0):
    assert rois.dim() == 2
    assert rois.size(1) == 5
    output = []
    rois = rois.data.float()
    num_rois = rois.size(0)

    rois[:, 1:].mul_(spatial_scale)
    rois = rois.long()
    for i in range(num_rois):
        roi = rois[i]
        im_idx = roi[0]
        im = input.narrow(0, im_idx, 1)[..., roi[2]:(roi[4]+1), roi[1]:(roi[3]+1)]
        output.append(F.adaptive_max_pool2d(im, size))

    output = torch.cat(output, 0)

    return output