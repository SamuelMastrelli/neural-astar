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
    

    def resize_tensor(tensor, new_shape):
        """
        Ridimensiona un tensore a nuove dimensioni, aggiungendo padding di zeri se necessario
        o tagliando il tensore se le nuove dimensioni sono più piccole.
        
        Parameters:
        tensor (np.ndarray): Il tensore originale.
        new_shape (tuple): Le nuove dimensioni desiderate (rows, cols).
        
        Returns:
        np.ndarray: Il tensore ridimensionato.
        """
        original_shape = tensor.shape
        
        # Crea un nuovo tensore pieno di zeri con le dimensioni specificate
        new_tensor = np.zeros(new_shape, dtype=tensor.dtype)
        
        # Calcola gli offset per centrare il tensore originale nel nuovo tensore
        offset_row = (new_shape[0] - original_shape[0]) // 2
        offset_col = (new_shape[1] - original_shape[1]) // 2
        
        # Se il nuovo tensore è più piccolo, prendiamo solo una porzione centrale del tensore originale
        start_row = max(0, -offset_row)
        start_col = max(0, -offset_col)
        end_row = min(original_shape[0], new_shape[0] - offset_row)
        end_col = min(original_shape[1], new_shape[1] - offset_col)
        
        # Posiziona la parte appropriata del tensore originale nel nuovo tensore
        new_tensor[max(0, offset_row):max(0, offset_row) + (end_row - start_row),
                max(0, offset_col):max(0, offset_col) + (end_col - start_col)] = tensor[start_row:end_row, start_col:end_col]
        
        return new_tensor
