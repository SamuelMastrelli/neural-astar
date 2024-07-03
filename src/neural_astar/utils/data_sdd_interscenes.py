"""
Sdd dataset 

"""



from __future__ import annotations, print_function

import numpy as np
import torch
import torch.utils.data as data
from neural_astar.planner.differentiable_astar import AstarOutput
from PIL import Image
from torchvision.utils import make_grid
import os


def create_sdd_dataloader(
        dirname: str,
        locations: list,
        batch_size: int,
) -> data.DataLoader:
    
    dataset = SDD_Dataset(dirname, locations)
    return data.DataLoader(
        dataset, batch_size, num_workers=4
    )


class SDD_Dataset(data.Dataset):
    def __init__(
            self,
            dirname: str, 
            locations: list,       
    ):
        self.dirname = dirname
        dir = os.fsdecode(dirname)
        images = []
        start_images = []
        goal_images = []
        traj_images = []
        for location in os.listdir(dir):
            location_name = os.fsdecode(location)
            if location_name in locations:
                for file in os.listdir(dir + location_name):
                    video = os.fsdecode(file)
                    for fileV in os.listdir(dir + location_name + "/"+ video):
                        filename = os.fsdecode(fileV)
                        if filename.endswith("npz"):
                            (
                                image,
                                start_image,
                                goal_image,
                                traj_image,
                            ) = self._process(filename, video, location_name)
                
                            images.append(image)
                            start_images.append(start_image)
                            goal_images.append(goal_image)
                            traj_images.append(traj_image)
                    self.images = self.toTensor(tlist=images).permute(0, 3, 1, 2)
                    self.start_images = self.toTensor(tlist=start_images).unsqueeze(1)
                    self.goal_images = self.toTensor(tlist=goal_images).unsqueeze(1)
                    self.traj_images = self.toTensor(tlist=traj_images).unsqueeze(1)

                


    def _process(self,  filename: str, video: str, location_name: str):
        with np.load(self.dirname + location_name + "/"+ video + "/" +filename) as f:
            image = f["image"]
            start_image = f["start_image"]
            goal_image = f["goal_image"]
            traj_image = f["traj_image"]
        image = image.astype(np.float32)
        start_image = start_image.astype(np.float32)
        goal_image = goal_image.astype(np.float32)
        traj_image = traj_image.astype(np.float32)

        return torch.tensor(image), torch.tensor(start_image), torch.tensor(goal_image), torch.tensor(traj_image)


    def __getitem__(self, index: int):
        image = self.images[index]
        start_map = self.start_images[index]
        goal_map = self.goal_images[index]
        traj_image = self.traj_images[index]

        return image, start_map, goal_map, traj_image
    
    def __len__(self):
        return self.images.shape[0]


    def toTensor(self, tlist: list):
        dims = list(tlist[0].shape)
        result = torch.zeros(len(tlist), *dims)
        i = 0
        for image in tlist:
            result[i] = image
            i+=1
        return result

                
