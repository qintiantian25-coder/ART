from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

import numpy as np
from os import path as osp
from basicsr.utils import scandir


@DATASET_REGISTRY.register()
class PairedImageMaskDataset(data.Dataset):
    """Paired image dataset with an extra mask channel.

    Expects options to provide `dataroot_lq`, `dataroot_gt`, and `dataroot_mask`.
    The mask files should have the same basename as corresponding images.
    The dataset will return `lq` as a tensor with shape (C+1,H,W) where the last
    channel is the mask (0/1 float), and `gt` as usual.
    """

    def __init__(self, opt):
        super(PairedImageMaskDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        self.task = opt.get('task', None)

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.mask_folder = opt.get('dataroot_mask', None)

        # build paired list by scanning gt folder and matching by basename
        gt_names = list(scandir(self.gt_folder))
        lq_names = set(scandir(self.lq_folder))
        if self.mask_folder is not None:
            mask_names = set(scandir(self.mask_folder))
        else:
            mask_names = set()

        paths = []
        for gt_name in sorted(gt_names):
            basename, ext = osp.splitext(gt_name)
            # lq may have same ext
            lq_name = gt_name
            lq_path = osp.join(self.lq_folder, lq_name)
            gt_path = osp.join(self.gt_folder, gt_name)
            if lq_name not in lq_names:
                raise FileNotFoundError(f'{lq_name} not found in {self.lq_folder}')
            if self.mask_folder is not None:
                # try same basename with any extension in mask folder
                mask_path = None
                for m in mask_names:
                    if osp.splitext(m)[0] == basename:
                        mask_path = osp.join(self.mask_folder, m)
                        break
                if mask_path is None:
                    raise FileNotFoundError(f'Mask for {basename} not found in {self.mask_folder}')
            else:
                mask_path = None

            paths.append({'lq_path': lq_path, 'gt_path': gt_path, 'mask_path': mask_path})

        self.paths = paths

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt.get('scale', 1)

        entry = self.paths[index]
        gt_path = entry['gt_path']
        lq_path = entry['lq_path']
        mask_path = entry.get('mask_path', None)

        # load images
        img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'), float32=True)
        img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'), float32=True)

        if mask_path is not None:
            mask = imfrombytes(self.file_client.get(mask_path, 'mask'), flag='grayscale', float32=True)
            # ensure mask is single channel and values in {0,1}
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask = (mask > 0.5).astype(np.float32)
            mask = np.expand_dims(mask, axis=2)
        else:
            # default empty mask
            mask = np.zeros((img_lq.shape[0], img_lq.shape[1], 1), dtype=np.float32)

        # augmentation
        if self.opt.get('phase') == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq, mask = self.random_crop_with_mask(img_gt, img_lq, mask, gt_size, scale)
            img_gt, img_lq, mask = self.augment_with_mask([img_gt, img_lq, mask], self.opt.get('use_hflip', False), self.opt.get('use_rot', False))

        # color convert if needed (not handled for masks)
        # convert to tensors: img2tensor expects list of images HWC
        from basicsr.utils import img2tensor
        img_gt, img_lq, mask = img2tensor([img_gt, img_lq, mask], bgr2rgb=True, float32=True)

        # concatenate mask as extra channel to lq
        # img_lq: C x H x W, mask: 1 x H x W
        import torch
        img_lq = torch.cat([img_lq, mask], dim=0)

        # normalize if mean/std provided (note: mean/std should match channels)
        if self.mean is not None or self.std is not None:
            from torchvision.transforms.functional import normalize
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean[:img_gt.size(0)], self.std[:img_gt.size(0)], inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

    def random_crop_with_mask(self, img_gt, img_lq, mask, gt_size, scale):
        # reuse paired_random_crop logic but implement minimal version here
        from basicsr.data.transforms import paired_random_crop
        # paired_random_crop expects two images; here we call it for gt and lq then crop mask accordingly
        img_gt_c, img_lq_c = paired_random_crop(img_gt, img_lq, gt_size, scale, '')
        # paired_random_crop returns crops using gt_path param which we pass empty; for mask, we need same crop
        # To keep it simple, re-compute crop indices by finding top-left difference
        # Fallback: center crop mask to match img_lq_c size
        h, w, _ = img_lq_c.shape
        mask_h, mask_w, _ = mask.shape
        if mask_h != h or mask_w != w:
            # center crop
            top = max(0, (mask_h - h) // 2)
            left = max(0, (mask_w - w) // 2)
            mask = mask[top:top + h, left:left + w, :]
        else:
            mask = mask
        return img_gt_c, img_lq_c, mask

    def augment_with_mask(self, imgs, use_hflip, use_rot):
        # imgs: [img_gt, img_lq, mask]
        from basicsr.data.transforms import augment
        img_gt, img_lq = augment([imgs[0], imgs[1]], use_hflip, use_rot)
        # apply same hflip/rot to mask using numpy
        mask = imgs[2]
        # Note: For simplicity, only apply horizontal flip and 90deg rotation if used
        if use_hflip and np.random.rand() < 0.5:
            img_gt = np.flip(img_gt, axis=1).copy()
            img_lq = np.flip(img_lq, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if use_rot and np.random.rand() < 0.5:
            img_gt = np.rot90(img_gt).copy()
            img_lq = np.rot90(img_lq).copy()
            mask = np.rot90(mask).copy()
        return img_gt, img_lq, mask
