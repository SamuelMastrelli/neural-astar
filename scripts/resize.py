

from PIL import Image
import torchvision.transforms as transforms
import os
import torch

for cluster in os.listdir('src/neural_astar/utils/voronoi_utilities/maps_data/maps'):
    if not cluster.endswith('_resized') and cluster != "DiscardedImages":
        for img in os.listdir('src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster):
            image = Image.open('src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster+'/'+img)



            if image.size[0] >= 800 and image.size[1] >= 800:
                    res=transforms.Resize(400)(image)

                    transform = transforms.Compose([
                                transforms.ToTensor()
                            ])

                    image_tensor = transform(res) 

                    image_tensor = torch.clamp(image_tensor.mean(0), 0, 1)

                    image_tensor[image_tensor<0.9] = 0
                    image_tensor[image_tensor>=0.9] = 1

                    newImage = transforms.ToPILImage()(image_tensor.detach())

                    # pixel_data = newImage.getdata()

                    # # Conta i pixel neri (0) e bianchi (255)
                    # black = sum(1 for pixel in pixel_data if pixel == 0)
                    # total_pixel = len(pixel_data)

                    # # Calcola la percentuale di nero
                    # perc_black = (black / total_pixel) * 100

                    # if perc_black >= 0.0:

                    name = img.split(".")[0]

                    newImage.save('src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster+'_resized/'+name+".jpg", quality=100)
                    
            else:
                    image.save('src/neural_astar/utils/voronoi_utilities/maps_data/maps/DiscardedImages/' + img )
            