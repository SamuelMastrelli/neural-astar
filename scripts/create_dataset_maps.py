from __future__ import annotations, print_function

import numpy as np
import torch
import torchvision.transforms as transforms
import os
from neural_astar.utils.voronoi_utilities.voronoi_graph_generator import VoronoiGraphGenerator
import cv2




def process(dir: str, cluster: str, image: str):
        #Da immagine a tensore

        img = cv2.imread(dir + "/" + cluster + "/" + image)
        print(image)

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

        
        
        voronoi_graph_generator = VoronoiGraphGenerator(cluster=cluster, env_name=env_name, floor=floor)
        voronoi_graph_generator.generate_voronoi_bitmap()
        
        
        starts: list[torch.Tensor] = []
        goals: list[torch.Tensor] = []
        opt_trajs: list[torch.Tensor] = []


        #Scelta goal e start
        start_goal_list = voronoi_graph_generator.select_reachable_nodes()
        for (start, goal) in start_goal_list:
            starts.append(torch.from_numpy(voronoi_graph_generator.to_numpy_array([start])))
            goals.append(torch.from_numpy(voronoi_graph_generator.to_numpy_array([goal])))
            #path ottimo
            path, _ = voronoi_graph_generator.find_shortest_path(start, goal)
            path = voronoi_graph_generator.to_numpy_array(path)
      
            opt_trajs.append(torch.from_numpy(path))

        map_design = map_design.permute(1,0).unsqueeze(0).expand(len(starts), -1, -1, -1)
        return map_design, toTensor(starts), toTensor(goals), toTensor(opt_trajs)


def toTensor(tlist: list):
        dims = list(tlist[0].shape)
        result = torch.zeros(len(tlist), *dims)
        
        i = 0
        for image in tlist:
            result[i] = image
            i+=1
        return result

def generate(dir: str, cluster: str):
    
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

            ) = process(dirname, cluster,  image)

            

            maps_design.append(map_design)
            starts.append(start)
            goals.append(goal)
            opt_trajs.append(opt_traj)

         
    
        
        maps_design_tensor = toTensor(tlist=maps_design).reshape(len(maps_design)*map_design.shape[0], 1, map_design.shape[3], map_design.shape[3])
        starts_tensor = toTensor(tlist=starts).reshape(len(maps_design)*map_design.shape[0], 1, map_design.shape[3], map_design.shape[3])
        goals_tensor = toTensor(tlist=goals).reshape(len(maps_design)*map_design.shape[0], 1, map_design.shape[3], map_design.shape[3])
        opt_trajs_tensor = toTensor(tlist=opt_trajs).reshape(len(maps_design)*map_design.shape[0], 1, map_design.shape[3], map_design.shape[3])

        return maps_design_tensor, starts_tensor, goals_tensor, opt_trajs_tensor



dir = '/home/mastrelli/neural-astar/src/neural_astar/utils/voronoi_utilities/maps_data/maps' 

maps_test, starts_test, goals_test, opt_trajs_test = generate(dir, 'test_resized')
maps_train, starts_train, goals_train, opt_trajs_train = generate(dir, 'train_resized')
maps_validation, starts_validation, goals_validation, opt_trajs_validation = generate(dir, 'validation_resized')

np.savez('/home/mastrelli/neural-astar/src/maps_npz/', maps_test.numpy(), starts_test.numpy(), goals_test.numpy(), opt_trajs_test.numpy())
np.savez('/home/mastrelli/neural-astar/src/maps_npz/', maps_train.numpy(), starts_train.numpy(), goals_train.numpy(), opt_trajs_train.numpy())
np.savez('/home/mastrelli/neural-astar/src/maps_npz/', maps_validation.numpy(), starts_validation.numpy(), 
          goals_validation.numpy(), opt_trajs_validation.numpy())
