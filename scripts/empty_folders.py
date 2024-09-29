import os
import shutil

dir = '/home/mastrelli/neural-astar/src/neural_astar/utils/voronoi_utilities/maps_data/maps/'

shutil.rmtree(dir + 'test_resized')
shutil.rmtree(dir + 'train_resized')
shutil.rmtree(dir + 'validation_resized')

os.mkdir('src/neural_astar/utils/voronoi_utilities/maps_data/maps/test_resized')
os.mkdir('src/neural_astar/utils/voronoi_utilities/maps_data/maps/train_resized')
os.mkdir('src/neural_astar/utils/voronoi_utilities/maps_data/maps/validation_resized')