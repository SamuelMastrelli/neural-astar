"""
Voronoi Graph Generator, partially copied from 
https://github.com/micheleantonazzi/gibson-env-utilities/blob/main/gibson_env_utilities/
"""


import os
from typing import List, Tuple, Dict, Set, Union
import heapq
import cv2
import yaml
import numpy as np
import sys
from skimage.morphology import skeletonize
from termcolor import colored
from collections import deque
import random

from neural_astar.utils.voronoi_utilities.Graph.voronoi_graph import Coordinate, Graph, Node


class VoronoiGraphGenerator:
    def __init__(self, cluster: str, env_name: str, floor: int):
        self._env_name = env_name
        self._floor = floor

        try:
            self._map = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'maps_data', 'maps', cluster, env_name + '_floor_' + str(floor) + '.png'))
            with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'maps_data', 'maps_metadata', env_name + '_floor_' + str(floor)) + '.yaml', mode='r') as f:
                map_metadata: Dict = yaml.load(f, Loader=yaml.FullLoader)
            self._scale = map_metadata['scale']
            self._map_origin = Coordinate(x=map_metadata['origin']['x'], y=map_metadata['origin']['y'],
                                          map_origin=(map_metadata['origin']['x'], map_metadata['origin']['y']), scale=self._scale)
        except FileNotFoundError:
            print(colored(
                'The map or its metadata of the {0} world do not exist! Create them before using this VoronoiGraphGenerator'.format(
                    env_name), 'red'))
            raise FileNotFoundError

        self._map: np.array = cv2.cvtColor(self._map, cv2.COLOR_RGB2GRAY)
        self._voronoi_bitmap = np.array([], dtype=int)

        # Graph structure
        # Contains the black point of the voronoi bitmap (which are all graph nodes)
        self._graph = Graph(self._map.shape[0], self._map.shape[1], self._map_origin, self._scale)

    def _generate_voronoi_graph(self, voronoi_bitmap) -> Graph:
        """
        Extracts the graph from voronoi bitmap.
        The graph can be composed by multiple connected components, the graph entity stores all of them.
        Typically the robot positions are chosen from the longest one.
        The side lines are pruned directly by the graph.
        :return: the graph
        """

        graph = Graph(self._map.shape[0], self._map.shape[1], self._map_origin, self._scale)
        # Creates graph nodes converting black pixels
        for y, x in np.ndindex(voronoi_bitmap.shape[:2]):
            # If the pixel is black, it represents a graph node
            if voronoi_bitmap[y, x] == 0:
                node = Node(Coordinate(x=x, y=y, map_origin=(self._map_origin.x, self._map_origin.y), scale=self._scale))
                graph.add_node(node)

        # Search connection between nodes
        # Two nodes are connected it their image coordinates are adjacent
        # For each node, its surroundings is checked to find other black pixels (that are connected nodes).
        nodes = graph.get_nodes()
        for node in nodes.values():
            y, x = node.get_coordinate().to_img_index()
            mask_indexes = [(y1, x1)
                            for y1 in range(max(0, y - 1), min(y + 2, voronoi_bitmap.shape[0]))
                            for x1 in range(max(0, x - 1), min(x + 2, voronoi_bitmap.shape[1]))
                            if x1 != x or y1 != y]

            for y1, x1 in mask_indexes:
                if voronoi_bitmap[y1, x1] == 0:
                    graph.add_connection(node1_coordinates=Coordinate(x=x, y=y, map_origin=self._map_origin.get_x_y_tuple(), scale=self._scale),
                                         node2_coordinates=Coordinate(x=x1, y=y1, map_origin=self._map_origin.get_x_y_tuple(), scale=self._scale))

        graph.find_connected_components()
        return graph

    def get_voronoi_graph(self) -> Graph:
        return self._graph

    def generate_voronoi_bitmap(self, save_to_file: bool = False) -> np.array:
        """
        This method generates a voronoi bitmap starting from a floor map.
        Steps:
            1) a thresholding procedure is applied to the original floor map (the values between 0 and 250 are turned to 0)
            2) then the resulting image is eroded and dilated
            3) the resulting image is processed to find the contours
            4) the building's outline is identified (searching the longest contour)
            5) the external area of the building's contour is black filled
            6) the contour inside the building's outline are drawn and black filled
               (now the floor plan is black outside the building and over the obstacles)
            7) using these simplified contours, it is calculated the voronoi diagram
            8) the segments of the voronoi facets perimeter are examined.
               They are drawn only if they are inside the building's outline and not overlap an obstacle
               (in other words, if the extreme points that define a segment are inside the image and the correspondent pixel is white)
            9) The voronoi bitmap is used to create the graph, which finds the connected components and prunes the side lines
            10) To remove other imperfection, the voronoi bitmap generated by the graph is dilated and its skeleton
                is found using scikit-image. The old graph is replaced with a new one generated using the skeletonized voronoi bitmap
        :return: the voronoi bitmap
        """
        # 1) Threshold map
        ret, threshed_image = cv2.threshold(self._map, 250, 255, cv2.THRESH_BINARY)
        # cv2.imshow('thresh image', threshed_image)
        # cv2.waitKey(0)

        # 2) Map erosion and dilation
        eroded_image = cv2.erode(threshed_image, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_REFLECT)
        # cv2.imshow('eroded image', threshed_image)
        # cv2.waitKey(0)

        dilated_image = cv2.dilate(eroded_image, np.ones((3, 3), np.uint8))
        # cv2.imshow('dilate image', threshed_image)
        # cv2.waitKey(0)

        # 3) Find contours
        (image_width, image_height) = dilated_image.shape
        contour_image = np.array([0 for _ in range(image_width * image_height)], dtype='uint8').reshape(
            (image_width, image_height))
        contours, hierarchy = cv2.findContours(dilated_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # 4) Find the building contour (it is assumed to be the longest one)
        l_contour_index, l_contour = max(enumerate(contours[1:]), key=lambda v: cv2.arcLength(v[1], closed=True))
        l_contour_index += 1

        #  5) Fill the area outside the building's contour
        cv2.drawContours(contour_image, contours, contourIdx=l_contour_index, color=255, thickness=cv2.FILLED)
        # cv2.imshow('external contour image', contour_image)
        # cv2.waitKey(0)

        # 6) Draw only contours inside the longest one and fill them (hierarchy = [Next, Previous, First_Child, Parent])
        filled_image = contour_image.copy()
        enumerate_hierarchy = list(enumerate(hierarchy[0]))

        def draw_internal_contours(e_hierarchy: Tuple[int, List]):
            index = e_hierarchy[0]
            hierarchy_data = e_hierarchy[1]

            # Draw and fill contour
            cv2.drawContours(filled_image, [contours[index]], contourIdx=-1, color=0, thickness=cv2.FILLED)

            # If this contour has a next one at the same hierarchy level
            if hierarchy_data[0] != -1:
                draw_internal_contours(enumerate_hierarchy[hierarchy_data[0]])

            # If this contour has a child
            if hierarchy_data[2] != -1:
                draw_internal_contours(enumerate_hierarchy[hierarchy_data[2]])

        # Get first child of the external contour and all the internal contours are drawn and black filled
        first_child = hierarchy[0][l_contour_index][2]
        if first_child != -1:
            draw_internal_contours(enumerate_hierarchy[first_child])
        # cv2.imshow('filled image', filled_image)
        # cv2.waitKey(0)

        # 7) The voronoi diagram is calculated using Delaunay triangulation
        rect = (0, 0, self._map.shape[1], self._map.shape[0])
        subdiv = cv2.Subdiv2D(rect)

        for contour, contour_hierarchy in zip(contours, hierarchy[0]):
            # Insert the all contours' points into subdiv
            for point in [np.array(p[0], dtype=float) for p in contour]:
                subdiv.insert(point)

        # 8) Draw voronoi facets contours and create the voronoi bitmap
        eroded_filled_map = cv2.erode(filled_image, kernel=np.ones((3, 3), dtype=int), iterations=1)
        # cv2.imshow('eroded filled', eroded_filled_map)
        # cv2.waitKey()
        voronoi_bitmap = np.array([255 for _ in range(image_width * image_height)], dtype=np.uint8).reshape(
            (image_width, image_height))
        (facets, centers) = subdiv.getVoronoiFacetList([])

        for facet in facets:
            facet_points = np.array(facet, int)

            # Draw voronoi facets contour lines only if they are inside image boundaries
            facet_lines = zip(np.roll(facet_points, 1, axis=0), facet_points)

            for p1, p2 in facet_lines:
                if 0 <= p1[0] < contour_image.shape[1] and 0 <= p1[1] < contour_image.shape[0] and \
                        0 <= p2[0] < contour_image.shape[1] and 0 <= p2[1] < contour_image.shape[0] \
                        and eroded_filled_map[p1[1], p1[0]] > 0 and eroded_filled_map[p2[1], p2[0]] > 0:
                    cv2.line(voronoi_bitmap, p1, p2, color=0, thickness=1)

        # cv2.imshow('voronoi bitmap', voronoi_bitmap)
        # cv2.waitKey()

        # 9) Create the voronoi graph
        graph = self._generate_voronoi_graph(voronoi_bitmap)
        graph.prune_side_lines()
        voronoi_bitmap = graph.get_graph_bitmap()

        # 10) Generate the skeletonized voronoi bitmap and replace graph
        dilated_voronoi_bitmap = cv2.bitwise_not(voronoi_bitmap)
        dilated_voronoi_bitmap = cv2.dilate(dilated_voronoi_bitmap, kernel=np.ones((3, 3), dtype=int),
                                            borderType=cv2.BORDER_CONSTANT)
        # cv2.imshow('dilated voronoi bitmap', dilated_voronoi_bitmap)
        # cv2.waitKey()

        dilated_voronoi_bitmap[dilated_voronoi_bitmap == 255] = 1
        skeletonized_voronoi_bitmap = cv2.bitwise_not((skeletonize(dilated_voronoi_bitmap) * 255).astype(np.uint8))
        # cv2.imshow('skeletonized voronoi bitmap', skeletonized_voronoi_bitmap)
        # cv2.waitKey()
        self._graph = self._generate_voronoi_graph(skeletonized_voronoi_bitmap)
        self._voronoi_bitmap = self._graph.get_graph_bitmap()

        if save_to_file:
            # Save voronoi bitmap
            cv2.imwrite(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'maps_data', 'voronoi_bitmaps',
                self._env_name + '_floor_' + str(self._floor) + '.png'),
                self._voronoi_bitmap)

            # Save map + voronoi bitmap
            map_voronoi_bitmap = self._map.copy()
            map_voronoi_bitmap[self._voronoi_bitmap == 0] = 0
            cv2.imwrite(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'maps_data', 'maps_with_voronoi_bitmaps',
                self._env_name + '_floor_' +  str(self._floor) + '.png'),
                map_voronoi_bitmap)

        return self._voronoi_bitmap

    def get_map(self) -> np.array:
        return self._map
    
    def get_reachable_nodes(self, start_node: Node) -> List[Node]:
        '''
        This method returns a list of all reachable nodes starting from
        start_node
        '''
        visited = set()
        queue = deque([start_node])
        reachable_nodes = []

        while queue:
            current_node = queue.popleft()
            if current_node in visited:
                continue
            visited.add(current_node)
            reachable_nodes.append(current_node)
            for neighbor in current_node.get_connected_nodes():
                if neighbor not in visited:
                    queue.append(neighbor)
        return reachable_nodes
                    
    def select_reachable_nodes(self) -> Tuple[Node, Node]:
        '''
        This method provides a tuple of nodes, start and goal nodes.
        The start is choosen randomly and the goal ramdomly within the reachable nodes
        '''
        nodes = list(self._graph.get_nodes().values())
        random_start_node = random.choice(nodes)
        reachable_nodes = self.get_reachable_nodes(random_start_node)

        if reachable_nodes is None or len(reachable_nodes) < 2:
            raise ValueError("Not enough reachable nodes")
        
        random_end_node = random.choice(reachable_nodes)
        while random_start_node == random_end_node:
            random_end_node = random.choice(reachable_nodes)

        return random_start_node, random_end_node

    def find_shortest_path(self, start: Node, end: Node) -> List[Node]:
        '''
        This methods is able to find the path between start and end(goal),
        using Dijkstra algorithm and using method dist_between ad heuristic 
        '''
        graph = self._graph
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {node: float('inf') for node in graph.get_nodes().values()}
        g_score[start] = 0
        f_score = {node: float('inf') for node in graph.get_nodes().values()}
        f_score[start] = self.dist_between(start, end)

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == end:
                return self.reconstruct_path(came_from, current)
            

            for neighbor in current.get_connected_nodes():
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.dist_between(neighbor, end)
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []


    def dist_between(self, a: Node, b: Node) -> float:
        '''
        As heuristic distance between two nodes, we use the euclidean norm of Numpy
        '''
        return np.linalg.norm(np.array(a.get_coordinate().get_x_y_tuple()) - np.array(b.get_coordinate().get_x_y_tuple()))

    def reconstruct_path(self, came_from: Dict[Node, Node], current: Node) -> List[Node]:
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path

    def draw_path_on_bitmap(self, path: List[Node]) -> np.array:
        path_bitmap = self._voronoi_bitmap.copy()
        for i in range(len(path) - 1):
            x, y = path[i].get_coordinate().get_x_y_tuple()
            x1, y1 = path[i + 1].get_coordinate().get_x_y_tuple()
            cv2.line(path_bitmap, (x, y), (x1, y1), color=0, thickness=2)
        return path_bitmap
    
    def to_numpy_array(self, st: List[Node]) -> np.array:
        array = np.full(self._map.shape, 0.0)
        for node in st:
            x,y = node.get_coordinate().get_x_y_tuple()
            array[x, y] = 1.0
        return array