

from PIL import Image
import torchvision.transforms as transforms
import os

for cluster in os.listdir('/home/mastrelli/neural-astar/src/neural_astar/utils/voronoi_utilities/maps_data/maps'):
    if not cluster.endswith('_resized'):
        for img in os.listdir('/home/mastrelli/neural-astar/src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster):
            image = Image.open('/home/mastrelli/neural-astar/src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster+'/'+img)

            tr = transforms.Resize(200)
            res = tr(image)

            res.save('/home/mastrelli/neural-astar/src/neural_astar/utils/voronoi_utilities/maps_data/maps/'+cluster+'_resized/'+img)
