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
    # def __init__(
    #         self,
    #         dir: str,
    #         cluster: str,
    #     ):
        
    #     self.dir = dir 
    #     self.cluster = cluster
    #     dirname = os.fsdecode(dir)
    #     maps_design = []
    #     starts = []
    #     goals = []
    #     opt_trajs = []
    #     histories = []

    #     for image in os.listdir(dirname + "/" + cluster):
    #         (
    #             map_design,
    #             start,
    #             goal,
    #             opt_traj,
    #             historie,
    #         ) = self._process(image)

            

    #         maps_design.append(map_design)
    #         starts.append(start)
    #         goals.append(goal)
    #         opt_trajs.append(opt_traj)
    #         histories.append(historie)
         
    
        
    #     self.maps_design = self.toTensor(tlist=maps_design).reshape(len(maps_design)*map_design.shape[0], 1, map_design.shape[3], map_design.shape[3])
    #     self.starts = self.toTensor(tlist=starts).reshape(len(maps_design)*map_design.shape[0], 1, map_design.shape[3], map_design.shape[3])
    #     self.goals = self.toTensor(tlist=goals).reshape(len(maps_design)*map_design.shape[0], 1, map_design.shape[3], map_design.shape[3])
    #     self.opt_trajs = self.toTensor(tlist=opt_trajs).reshape(len(maps_design)*map_design.shape[0], 1, map_design.shape[3], map_design.shape[3])
    #     self.histories = self.toTensor(tlist=histories).reshape(len(maps_design)*map_design.shape[0], 1, map_design.shape[3], map_design.shape[3])

    def __init__(
         self,
         dir: str,
         filename: str   
    ):
        self.dirname = os.fsdecode(dir)
        self.filename = filename
        (
            self.map_designs,
            self.start_maps,
            self.goal_maps,
            self.opt_trajs,
        ) = self._process(self.dirname, filename)
        

    def _process(self, dir, filename):
        with np.load(dir +'/'+filename) as f:
            maps_desings = torch.from_numpy(f['arr_0'])
            start_maps = torch.from_numpy(f['arr_1'])
            goal_maps = torch.from_numpy(f['arr_2'])
            opt_trajs = torch.from_numpy(f['arr_3'])

        return maps_desings, start_maps, goal_maps, opt_trajs

        


    # def _process(self, image: str):
    #     #Da immagine a tensore

    #     img = cv2.imread(self.dir + "/" + self.cluster + "/" + image)

    #     print(image)
    #     print(img.shape)
    #     # Converti l'immagine in scala di grigi
    #     gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #     # Creare una maschera binaria delle aree bianche
    #     _, binary_mask = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)

    #     # Invertire la maschera
    #     binary_mask_inv = cv2.bitwise_not(binary_mask)

    #     # Trova i contorni dell'edificio
    #     contours, _ = cv2.findContours(binary_mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     # Creare una maschera di riempimento
    #     fill_mask = np.zeros_like(gray_image)

    #     # Riempire solo le aree esterne all'edificio
    #     cv2.drawContours(fill_mask, contours, -1, (255), thickness=cv2.FILLED)

    #     # Invertire la maschera di riempimento
    #     fill_mask_inv = cv2.bitwise_not(fill_mask)

    #     # Applicare la maschera per cambiare le aree esterne bianche in nero
    #     img[fill_mask_inv == 255] = [0, 0, 0]

    #     transform = transforms.Compose([
    #                 transforms.ToTensor()
    #             ])

    #     image_tensor = transform(img) 


    #     map_design = torch.clamp(image_tensor.mean(0), 0, 1)
    #     map_design[map_design<0.9] = 0
    #     map_design[map_design>=0.9] = 1
       
    #     #Grafo di voronoi
    #     split = image.split('_floor_')
    #     env_name = split[0]
    #     floor = int(split[1].split('.')[0])

        
        
    #     voronoi_graph_generator = VoronoiGraphGenerator(cluster=self.cluster, env_name=env_name, floor=floor)
    #     voronoi_bitmap = voronoi_graph_generator.generate_voronoi_bitmap()
    #     graph = voronoi_graph_generator.get_voronoi_graph()
        
    #     starts: list[torch.Tensor] = []
    #     goals: list[torch.Tensor] = []
    #     opt_trajs: list[torch.Tensor] = []
    #     histories_list: list[torch.Tensor] = []

    #     #Scelta goal e start
    #     start_goal_list = voronoi_graph_generator.select_reachable_nodes()
    #     for (start, goal) in start_goal_list:
    #         starts.append(torch.from_numpy(voronoi_graph_generator.to_numpy_array([start])))
    #         goals.append(torch.from_numpy(voronoi_graph_generator.to_numpy_array([goal])))
    #         #path ottimo
    #         path, histories = voronoi_graph_generator.find_shortest_path(start, goal)
    #         path = voronoi_graph_generator.to_numpy_array(path)
    #         histories_list.append(torch.from_numpy(histories))
    #         opt_trajs.append(torch.from_numpy(path))

    #     map_design = map_design.permute(1,0).unsqueeze(0).expand(len(starts), -1, -1, -1)
    #     print(map_design.shape)
    #     return map_design, self.toTensor(starts), self.toTensor(goals), self.toTensor(opt_trajs), self.toTensor(opt_trajs)




    def __getitem__(self, index: int):
        map = self.map_designs[index]
        start_map = self.start_maps[index]
        goal_map = self.goal_maps[index]
        opt_traj = self.opt_trajs[index]

    

        return map, start_map, goal_map, opt_traj

    def __len__(self):
        return self.map_designs.shape[0]

    # def toTensor(self, tlist: list):
    #     dims = list(tlist[0].shape)
    #     result = torch.zeros(len(tlist), *dims)
        
    #     i = 0
    #     for image in tlist:
    #         result[i] = image
    #         i+=1
    #     return result
    
    '''
    def resize_tensor(self, tensor, new_shape: tuple):
        """
        Ridimensiona un tensore a nuove dimensioni, aggiungendo padding di zeri se necessario
        o tagliando il tensore se le nuove dimensioni sono più piccole.
        
        Parameters:
        tensor (torch.Tensor): Il tensore originale.
        new_shape (tuple): Le nuove dimensioni desiderate (rows, cols).
        
        Returns:
        torch.Tensor: Il tensore ridimensionato.
        """
        original_shape = tensor.shape
        
        # Crea un nuovo tensore pieno di zeri con le dimensioni specificate
        new_tensor = torch.zeros(new_shape, dtype=tensor.dtype)
        
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
    '''