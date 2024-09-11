

from PIL import Image
import torchvision.transforms as transforms
import os
import torch

for cluster in os.listdir('/home/sam/Desktop/Tesi/neural-astar/src/neural_astar/utils/voronoi_utilities/maps_data/maps'):
    if not cluster.endswith('_resized'):
        for img in os.listdir('/home/sam/Desktop/Tesi/neural-astar/src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster):
            image = Image.open('/home/sam/Desktop/Tesi/neural-astar/src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster+'/'+img)

            tr = transforms.Resize(200)
            res = tr(image)

            transform = transforms.Compose([
                        transforms.ToTensor()
                    ])

            image_tensor = transform(res) 

            image_tensor = torch.clamp(image_tensor.mean(0), 0, 1)
            image_tensor[image_tensor<0.9] = 0
            image_tensor[image_tensor>=0.9] = 1

            transforms.ToPILImage()(image_tensor).save('/home/sam/Desktop/Tesi/neural-astar/src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster+'_resized/'+img)
