from __future__ import annotations, print_function

import numpy as np
import torch
import torch.utils.data as data
from neural_astar.planner.differentiable_astar import AstarOutput
from PIL import Image
import torchvision.transforms as transforms
import os
from neural_astar.utils.voronoi_utilities.voronoi_graph_generator import VoronoiGraphGenerator
from neural_astar.utils.voronoi_utilities.Graph.voronoi_graph import Coordinate, Node, Graph
import cv2

def create_dataloader(
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
        self.cluster = cluster
        dirname = os.fsdecode(dir)
        maps_design = []
        starts = []
        goals = []
        opt_trajs = []
        histories = []

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

        img = cv2.imread(self.dir + "/" + self.cluster + "/" + image)

        # Converti l'immagine in scala di grigi
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Creare una maschera binaria delle aree bianche
        _, binary_mask = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY)

        # Invertire la maschera
        binary_mask_inv = cv2.bitwise_not(binary_mask)

        # Trova i contorni dell'edificio
        contours, _ = cv2.findContours(binary_mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Creare una maschera di riempimento
        fill_mask = np.zeros_like(gray_image)

        # Riempire solo le aree esterne all'edificio
        cv2.drawContours(fill_mask, contours, -1, (255), thickness=cv2.FILLED)

        # Invertire la maschera di riempimento
        fill_mask_inv = cv2.bitwise_not(fill_mask)

        # Applicare la maschera per cambiare le aree esterne bianche in nero
        img[fill_mask_inv == 255] = [0, 0, 0]

        transform = transforms.Compose([
                    transforms.ToTensor()
                ])

        image_tensor = transform(img) 


        map_design = torch.clamp(image_tensor.mean(0), 0, 1)
        map_design[map_design<0.9] = 0
        map_design[map_design>=0.9] = 1
       
        #Grafo di voronoi
        split = image.split('_floor_')
        env_name = split[0]
        floor = int(split[1].split('.')[0])

        
         
        voronoi_graph_generator = VoronoiGraphGenerator(cluster=self.cluster, env_name=env_name, floor=floor)
        voronoi_bitmap = voronoi_graph_generator.generate_voronoi_bitmap()
        graph = voronoi_graph_generator.get_voronoi_graph()
        
        #Scelta goal e start
        start, goal = voronoi_graph_generator.select_reachable_nodes()
        start_map = torch.from_numpy(voronoi_graph_generator.to_numpy_array([start]))
        goal_map = torch.from_numpy(voronoi_graph_generator.to_numpy_array([goal]))
        #path ottimo
        path, histories = voronoi_graph_generator.find_shortest_path(start, goal)
        path = voronoi_graph_generator.to_numpy_array(path)
        opt_traj = torch.from_numpy(path)

        return map_design.permute(1,0), start_map, goal_map, opt_traj




    def __getitem__(self, index: int):
        map = self.maps_design[index]
        start_map = self.starts[index]
        goal_map = self.goals[index]
        opt_traj = self.opt_trajs[index]
    

        return map, start_map, goal_map, opt_traj

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