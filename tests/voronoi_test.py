import numpy as np
import pytest
import cv2
from neural_astar.utils.voronoi_utilities.voronoi_graph_generator import VoronoiGraphGenerator
import os

def test_voronoi_bitmap():
    with pytest.raises(FileNotFoundError):
        VoronoiGraphGenerator(cluster='train_resized', env_name="house-46", floor=0)

    voornoi_graph_generator = VoronoiGraphGenerator(cluster='train_resized', env_name='house46', floor=0)
    voronoi_bitmap = voornoi_graph_generator.generate_voronoi_bitmap()
    graph = voornoi_graph_generator.get_voronoi_graph()


    assert np.array_equal(graph.get_graph_bitmap(), voronoi_bitmap)

def test_graph_nodes():
    voronoi_graph_generator = VoronoiGraphGenerator(cluster='train_resized',env_name='area3', floor=0)
    voronoi_bitmap = voronoi_graph_generator.generate_voronoi_bitmap()
    graph = voronoi_graph_generator.get_voronoi_graph()

    # Test nodes
    black_pixels = voronoi_bitmap[voronoi_bitmap == 0]
    assert len(graph.get_nodes().values()) == len(black_pixels)


def test_graph_connected_components():
    voronoi_graph_generator = VoronoiGraphGenerator(cluster='train_resized', env_name='area3', floor=0)
    voronoi_bitmap = voronoi_graph_generator.generate_voronoi_bitmap()
    graph = voronoi_graph_generator.get_voronoi_graph()

    components_image = np.array([[255 for _ in range(voronoi_bitmap.shape[0])] for _ in range(voronoi_bitmap.shape[1])], dtype=np.uint8)
    for component in graph.get_connected_components().values():
        for node in component:
            components_image[node.get_coordinate().to_img_index()] = 0

    assert np.array_equal(components_image, voronoi_bitmap)

def test_start_goal_shortest_path():
    voronoi_graph_generator = VoronoiGraphGenerator(cluster='train_resized', env_name='area4', floor=0)
    vb = voronoi_graph_generator.generate_voronoi_bitmap(True)

    se = voronoi_graph_generator.select_reachable_nodes()

    sh, _ = voronoi_graph_generator.find_shortest_path(se[0][0], se[0][1])
    hs, _ = voronoi_graph_generator.find_shortest_path(se[0][1], se[0][0])
  
    path_bitmap = voronoi_graph_generator.draw_path_on_bitmap(sh)
    path_bitmap1 = voronoi_graph_generator.draw_path_on_bitmap(hs)

                # Save voronoi bitmap
    cv2.imwrite(os.path.join(
        os.path.dirname('/home/mastrelli/neural-astar/src/neural_astar/utils/voronoi_utilities/'), 'maps_data', 'voronoi_bitmaps',
        "area5_2" + '_floor_' + str(0) + '.png'),
        path_bitmap)
    cv2.imwrite(os.path.join(
        os.path.dirname('/home/mastrelli/neural-astar/src/neural_astar/utils/voronoi_utilities/'), 'maps_data', 'voronoi_bitmaps',
        "area5_2rev" + '_floor_' + str(0) + '.png'),
        path_bitmap1)

    hs.reverse()
    
        

    assert np.array_equal(voronoi_graph_generator.to_numpy_array(sh), voronoi_graph_generator.to_numpy_array(hs))

