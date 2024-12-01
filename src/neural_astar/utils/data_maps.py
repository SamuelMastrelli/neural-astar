from __future__ import annotations, print_function

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import os
from neural_astar.utils.voronoi_utilities.voronoi_graph_generator import VoronoiGraphGenerator
from neural_astar.utils.voronoi_utilities.Graph.voronoi_graph import Coordinate, Node, Graph
import cv2
from torchvision.utils import make_grid


def visualize_results_voronoi(
    map_designs: torch.tensor, planner_outputs: torch.tensor, scale: int = 1
) -> np.ndarray:
    """
    Create a visualization of search results

    Args:
        map_designs (torch.tensor): input maps
        planner_outputs (torch.tensor): outout from voronoi graph
        scale (int): scale factor to enlarge output images. Default to 1.

    Returns:
        np.ndarray: visualized results
    """


  
    paths = planner_outputs
    results = make_grid(map_designs).permute(1, 2, 0) #make_grid fa una griglia di immagini, permute scambia le dimensioni
    p = make_grid(paths).permute(1, 2, 0).float()
    results[p[..., 0] == 1] = torch.tensor([1.0, 0.0, 0])

    results = ((results.numpy()) * 255.0).astype("uint8")

    if scale > 1:
        results = Image.fromarray(results).resize(
            [x * scale for x in results.shape[:2]], resample=Image.NEAREST
        )
        results = np.asarray(results)

    return results

def create_dataloader(
        dir: str,
        filename: str,
        batch_size: int
) -> data.DataLoader:
    dataset = Map_dataset(
        dir, filename
    )
    return data.DataLoader(dataset, batch_size=batch_size, num_workers=0)


class Map_dataset(data.Dataset):

    def __init__(
         self,
         dir: str,
         filename: str   
    ):
        self.dirname = os.fsdecode(dir)
        self.filepath = os.path.join(self.dirname, filename)

        # Solo metadati, senza caricare tutto il dataset in memoria
        with np.load(self.filepath) as f:
            self.num_samples = f['arr_0'].shape[0]  # Numero totale di esempi
            self.data_shapes = {
                'map_designs': f['arr_0'].shape[1:],
                'start_maps': f['arr_1'].shape[1:],
                'goal_maps': f['arr_2'].shape[1:],
                'opt_trajs': f['arr_3'].shape[1:]
            }
        


    def _load_single_sample(self, index):
        """Carica solo il campione richiesto dall'indice specificato."""
        with np.load(self.filepath) as f:
            map_design = torch.from_numpy(f['arr_0'][index])
            start_map = torch.from_numpy(f['arr_1'][index])
            goal_map = torch.from_numpy(f['arr_2'][index])
            opt_traj = torch.from_numpy(f['arr_3'][index])
        return map_design, start_map, goal_map, opt_traj

    def __getitem__(self, index: int):
        # Carica solo il campione richiesto
        return self._load_single_sample(index)

    def __len__(self):
        # Ritorna il numero di campioni totale, basato sui metadati
        return self.num_samples

   