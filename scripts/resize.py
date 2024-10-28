

from PIL import Image
import torchvision.transforms as transforms
import os
import torch

for cluster in os.listdir('src/neural_astar/utils/voronoi_utilities/maps_data/maps'):
    if not cluster.endswith('_resized') and cluster != "DiscardedImages":
        for img in os.listdir('src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster):
            image = Image.open('src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster+'/'+img).convert('1')

            if image.size[0] >= 800 and image.size[1] >= 800:
                res=image.resize((500, 500), Image.LANCZOS)

                transform = transforms.Compose([
                            transforms.ToTensor()
                        ])

                image_tensor = transform(res) 

                name = img.split(".")[0]

                transforms.ToPILImage()(image_tensor).save('src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster+'_resized/'+name+".jpg", quality=100)
            else:
                image.save('src/neural_astar/utils/voronoi_utilities/maps_data/maps/DiscardedImages/' + img )