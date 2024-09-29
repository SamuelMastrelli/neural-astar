

from PIL import Image
import torchvision.transforms as transforms
import os
import torch

for cluster in os.listdir('src/neural_astar/utils/voronoi_utilities/maps_data/maps'):
    if not cluster.endswith('_resized') and cluster != "DiscardedImages":
        for img in os.listdir('src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster):
            image = Image.open('src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster+'/'+img)

            if image.size[0] >= 800 and image.size[1] >= 800:
                tr = transforms.Resize(500)
                res = tr(image)

                transform = transforms.Compose([
                            transforms.ToTensor()
                        ])

                image_tensor = transform(res) 

                image_tensor = torch.clamp(image_tensor.mean(0), 0, 1)
                image_tensor[image_tensor<0.9] = 0
                image_tensor[image_tensor>=0.9] = 1

                transforms.ToPILImage()(image_tensor).save('src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster+'_resized/'+img)
            else:
                image.save('src/neural_astar/utils/voronoi_utilities/maps_data/maps/DiscardedImages/' + img )