import torch
import tracemalloc

from neural_astar.planner.astar import NeuralAstar
from neural_astar.utils.data_maps import create_dataloader


model = NeuralAstar(encoder_arch="CNN").to("cuda").eval()  
dataloader = create_dataloader("/home/mastrelli/neural-astar/src/maps_npz", "validation_ds.npz", 1)
map_designs, start_maps, goal_maps, opt_trajs= next(iter(dataloader))

print(torch.cuda.memory_allocated())

output = model(map_designs.to('cuda'), start_maps.to('cuda'), goal_maps.to('cuda'))  
  
print(torch.cuda.memory_allocated())



