

from PIL import Image
import torchvision.transforms as transforms
import os

for cluster in os.listdir(r'C:\Users\User\Desktop\uni\Tesi\neural-astar\src\neural_astar\utils\voronoi_utilities\maps_data\maps'):
    if not cluster.endswith('_resized'):
        for img in os.listdir('C:/Users/User/Desktop/uni/Tesi/neural-astar/src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster):
            image = Image.open('C:/Users/User/Desktop/uni/Tesi/neural-astar/src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster+'/'+img)

            tr = transforms.Resize(300)
            res = tr(image)

            res.save('C:/Users/User/Desktop/uni/Tesi/neural-astar/src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster+'_resized/'+img)
