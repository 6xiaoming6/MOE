import os
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
totensor = torchvision.transforms.ToTensor()

class MyDataSet(Dataset):
    #type为train或者val
    def __init__(self, root_dir, type = 'train', num_classes = 162, device='cuda:0', window_size=4):
        self.num_classes = num_classes
        self.window_size = window_size
        self.device = device

        self.path_8          = os.path.join(root_dir, f'50/inf_from_{type}')
        self.path_16         = os.path.join(root_dir, f'50/sup16_from_{type}')
        self.path_32         = os.path.join(root_dir, f'50/joint_from_{type}')
        self.path_32_gt      = os.path.join(root_dir, f'50/joint_from_{type}')
        path_mask            = os.path.join(root_dir, 'mask.csv')
        path_mapping         = os.path.join(root_dir, f'mapping_{type}.csv')

        img_list_8     = sorted(os.listdir(self.path_8))
        img_list_16    = sorted(os.listdir(self.path_16))
        img_list_32    = sorted(os.listdir(self.path_32))
        img_list_32_gt = sorted(os.listdir(self.path_32_gt))
        mapping_df = pd.read_csv(path_mapping, header=0, index_col=0, dtype=float)
        mask_df = pd.read_csv(path_mask, header=None)
        
        
        # Filter images based on naming pattern(sr是输入数据，hr是真值)
        self.img_list_8 = [img for img in img_list_8 if img.startswith('50_') and img.endswith('_sr.png')]
        self.img_list_16 = [img for img in img_list_16 if img.startswith('50_') and img.endswith('_sr.png')]
        self.img_list_32 = [img for img in img_list_32 if img.startswith('50_') and img.endswith('_sr.png')]
        self.img_list_32_gt = [img for img in img_list_32_gt if img.startswith('50_') and img.endswith('_hr.png')]
        self.mapping = torch.tensor(mapping_df.values)
        self.mask = torch.tensor(mask_df.values)
        if self.mask.min() == 1:
            self.mask = self.mask - 1 #要从0开始编号，减去1保证编号从0开始

        mask_flat = self.mask.reshape(-1).long()
        self.projection_mask_matrix = F.one_hot(mask_flat, num_classes=self.num_classes).float()
        self.category_counts = torch.bincount(mask_flat, minlength=self.num_classes)


    def __len__(self):
        return len(self.img_list_32)
    
    def __getitem__(self, index):

        def to_single_channel(image):
            if image.size(0) == 3 and torch.equal(image[0], image[1]) and torch.equal(image[1], image[2]):
                return image[0].unsqueeze(0)
            return image

        data_8      = to_single_channel(totensor(Image.open(os.path.join(self.path_8, self.img_list_8[index]))))
        data_16     = to_single_channel(totensor(Image.open(os.path.join(self.path_16, self.img_list_16[index]))))
        data_32     = to_single_channel(totensor(Image.open(os.path.join(self.path_32, self.img_list_32[index]))))
        data_32_gt  = to_single_channel(totensor(Image.open(os.path.join(self.path_32_gt, self.img_list_32_gt[index]))))
        mapping     = self.mapping[self.window_size + index]

        vmin, vmax = mapping[0], mapping[1]

        # Map data back to original range
        data_8 = data_8 * (vmax - vmin) + vmin
        data_16 = data_16 * (vmax - vmin) + vmin
        data_32 = data_32 * (vmax - vmin) + vmin
        data_32_gt = data_32_gt * (vmax - vmin) + vmin

        data_32_gt = data_32_gt.view(-1).float()
        data_32_gt = torch.matmul(data_32_gt, self.projection_mask_matrix)

        vmin, vmax = data_32_gt.min(), data_32_gt.max()

        return data_8, data_16, data_32, data_32_gt, vmin, vmax
    
    
if __name__ == "__main__":
    dataset = MyDataSet('./data')

    data_loader = DataLoader(dataset, 16)
    x = 0
    for x8, x16, x32, x32_gt in data_loader:
        print(x8.shape, x16.shape, x32.shape, x32_gt.shape)