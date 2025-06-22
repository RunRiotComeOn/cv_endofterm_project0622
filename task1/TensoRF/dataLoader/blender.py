# import torch,cv2
# from torch.utils.data import Dataset
# import json
# from tqdm import tqdm
# import os
# from PIL import Image
# from torchvision import transforms as T


# from .ray_utils import *


# class BlenderDataset(Dataset):
#     def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):

#         self.N_vis = N_vis
#         self.root_dir = datadir
#         self.split = split
#         self.is_stack = is_stack
#         self.img_wh = (int(800/downsample),int(800/downsample))
#         self.define_transforms()

#         self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
#         self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#         self.read_meta()
#         self.define_proj_mat()

#         self.white_bg = True
#         self.near_far = [2.0,6.0]
        
#         self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
#         self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
#         self.downsample=downsample

#     def read_depth(self, filename):
#         depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
#         return depth
    
#     def read_meta(self):

#         with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
#             self.meta = json.load(f)

#         w, h = self.img_wh
#         self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
#         self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh


#         # ray directions for all pixels, same for all images (same H, W, focal)
#         self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)
#         self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
#         self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()

#         self.image_paths = []
#         self.poses = []
#         self.all_rays = []
#         self.all_rgbs = []
#         self.all_masks = []
#         self.all_depth = []
#         self.downsample=1.0

#         img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
#         idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
#         for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#

#             frame = self.meta['frames'][i]
#             pose = np.array(frame['transform_matrix']) @ self.blender2opencv
#             c2w = torch.FloatTensor(pose)
#             self.poses += [c2w]

#             image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
#             self.image_paths += [image_path]
#             img = Image.open(image_path)
            
#             if self.downsample!=1.0:
#                 img = img.resize(self.img_wh, Image.LANCZOS)
#             img = self.transform(img)  # (4, h, w)
#             img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
#             img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
#             self.all_rgbs += [img]


#             rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
#             self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)


#         self.poses = torch.stack(self.poses)
#         if not self.is_stack:
#             self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
#             self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

# #             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
#         else:
#             self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
#             self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
#             # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)


#     def define_transforms(self):
#         self.transform = T.ToTensor()
        
#     def define_proj_mat(self):
#         self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

#     def world2ndc(self,points,lindisp=None):
#         device = points.device
#         return (points - self.center.to(device)) / self.radius.to(device)
        
#     def __len__(self):
#         return len(self.all_rgbs)

#     def __getitem__(self, idx):

#         if self.split == 'train':  # use data in the buffers
#             sample = {'rays': self.all_rays[idx],
#                       'rgbs': self.all_rgbs[idx]}

#         else:  # create data for each image separately

#             img = self.all_rgbs[idx]
#             rays = self.all_rays[idx]
#             mask = self.all_masks[idx] # for quantity evaluation

#             sample = {'rays': rays,
#                       'rgbs': img,
#                       'mask': mask}
#         return sample

import torch
import cv2
from torch.utils.data import Dataset
import json
from tqdm.auto import tqdm
import os
from PIL import Image
from torchvision import transforms as T
import numpy as np
from .ray_utils import *

class BlenderDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1):
        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.img_wh = (int(800/downsample), int(800/downsample))
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        # Initialize attributes to avoid AttributeError
        self.focal = None
        self.directions = None
        self.intrinsics = None
        self.poses = []
        self.all_rgbs = []
        self.all_rays = []
        
        self.white_bg = True
        self.near_far = [2.0, 6.0]
        
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample = downsample

        self.read_meta()
        self.define_proj_mat()

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        return depth
    
    def read_meta(self):
        try:
            # with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
                self.meta = json.load(f)

            w, h = self.img_wh
            self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])
            self.focal *= self.img_wh[0] / 800

            # ray directions for all pixels
            self.directions = get_ray_directions(h, w, [self.focal, self.focal])
            self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
            self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()

            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_masks = []
            self.all_depth = []

            img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
            idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
            
            for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):
                frame = self.meta['frames'][i]
                print(frame)
                pose = np.array(frame['transform_matrix']) @ self.blender2opencv
                c2w = torch.FloatTensor(pose)
                self.poses.append(c2w)

                # # image_path = os.path.join(self.root_dir, frame['file_path'])
                # print(self.root_dir)
                # # image_path = os.path.join(self.root_dir, frame['file_path'])
                # image_path = frame['file_path']
                # print(image_path)
                # self.image_paths.append(image_path)
                
                # img = Image.open(image_path)

                # 统一处理不同格式的file_path
                file_path = frame['file_path'].strip()
                
                # 方案1：自动移除重复的路径前缀
                if file_path.startswith('data/my_video/'):
                    file_path = file_path[len('data/my_video/'):]
                
                # 方案2：或者强制统一格式（如果知道所有图像都在images目录下）
                # if not file_path.startswith('images/'):
                #     file_path = 'images/' + os.path.basename(file_path)
                
                image_path = os.path.join(self.root_dir, file_path)

                self.image_paths.append(image_path)
                
                print(f"Loading image from: {image_path}")  # 调试输出
                
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
                self.image_paths.append(image_path)
                img = Image.open(image_path)

                ##
                if img.size != self.img_wh:
                    print(f"Resizing image from {img.size} to {self.img_wh}")
                    img = img.resize(self.img_wh, Image.LANCZOS)

                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img = self.transform(img)  # (4, h, w)

                ## 
                if img.shape[1:] != torch.Size(self.img_wh[::-1]):
                    raise ValueError(
                        f"Image {image_path} has wrong size {img.shape[1:]}, "
                        f"expected {self.img_wh[::-1]}"
                    )

                # img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
                # img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
                img = img.view(3, -1).permute(1, 0)  

                self.all_rgbs.append(img)

                rays_o, rays_d = get_rays(self.directions, c2w)
                self.all_rays.append(torch.cat([rays_o, rays_d], 1))  # (h*w, 6)

            self.poses = torch.stack(self.poses)
            
            if not self.is_stack:
                self.all_rays = torch.cat(self.all_rays, 0)  # (N*h*w, 6)
                self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (N*h*w, 3)
            else:
                # # Correct reshape operation
                # num_frames = len(self.all_rgbs)
                # expected_elements = num_frames * self.img_wh[0] * self.img_wh[1] * 3
                # actual_elements = torch.stack(self.all_rgbs, 0).numel()
                
                # if actual_elements != expected_elements:
                #     raise ValueError(
                #         f"Shape mismatch: Cannot reshape {actual_elements} elements "
                #         f"into {num_frames}x{self.img_wh[1]}x{self.img_wh[0]}x3 tensor. "
                #         f"Expected {expected_elements} elements."
                #     )
                num_frames = len(self.all_rgbs)
                expected_elements = num_frames * self.img_wh[0] * self.img_wh[1] * 3
                actual_elements = torch.stack(self.all_rgbs, 0).numel()
                
                print(f"Debug: Preparing to reshape - frames:{num_frames}, "
                    f"expected:{expected_elements}, actual:{actual_elements}")
                    
                if actual_elements != expected_elements:
                    raise ValueError(
                        f"Shape mismatch: Cannot reshape {actual_elements} elements "
                        f"into {num_frames}x{self.img_wh[1]}x{self.img_wh[0]}x3 tensor. "
                        f"Expected {expected_elements} elements."
                    )
                
                self.all_rays = torch.stack(self.all_rays, 0)  # (N, h*w, 6)
                self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(
                    num_frames, *self.img_wh[::-1], 3)  # (N, h, w, 3)

        except Exception as e:
            print(f"Error in read_meta(): {str(e)}")
            raise

    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)
        
    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        if self.split == 'train':
            sample = {
                'rays': self.all_rays[idx],
                'rgbs': self.all_rgbs[idx]
            }
        else:
            sample = {
                'rays': self.all_rays[idx],
                'rgbs': self.all_rgbs[idx],
                'mask': self.all_masks[idx] if hasattr(self, 'all_masks') else None
            }
        return sample