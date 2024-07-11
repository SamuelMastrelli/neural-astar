from __future__ import annotations, print_function

import numpy as np
import torch
import torch.utils.data as data
from neural_astar.planner.differentiable_astar import AstarOutput
from PIL import Image
from torchvision.utils import make_grid
import os

def create_dataloader(
        self,
        dir: str,
        cluster: str,
        batch_size: int
) -> data.DataLoader:
    dataset = Map_dataset(
        dir, cluster
    )
    return data.DataLoader(dataset, batch_size=batch_size, num_workers=0)


class Map_dataset(data.Dataset):
    def __init__(
            self,
            dir: str,
            cluster: str,
        ):
        self.dir = dir 
        dirname = os.fsdecode(dir)
        maps_design = []
        starts = []
        goals = []
        opt_trajs = []

        for image in os.listdir(dirname + "/" + cluster):
            (
                map_design,
                start,
                goal,
                opt_traj,
            ) = self._process(image)

            maps_design.append(map_design)
            starts.append(start)
            goals.append(goal)
            opt_trajs.append(opt_traj)
        
        self.maps_design = self.toTensor(tlist=maps_design).unsqueeze(1)
        self.starts = self.toTensor(tlist=starts).unsqueeze(1)
        self.goals = self.toTensor(tlist=goals).unsqueeze(1)
        self.opt_trajs = self.toTensor(tlist=opt_trajs).unsqueeze(1)


    def _process(self, image: str):
        #Da immagine a tensore
        #Grafo di voronoi
        #Scelta goal, dijkstra 
        #Scelta start e ricavo percorso


    def __getitem__(self, index):
        


    def __len__(self):
        return self.maps_design.shape[0]

    def toTensor(self, tlist: list):
        dims = list(tlist[0].shape)
        result = torch.zeros(len(tlist), *dims)
        i = 0
        for image in tlist:
            result[i] = image
            i+=1
        return result