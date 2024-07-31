"""Differentiable A* module and helper functions
Author: Ryo Yonetani, Mohammadamin Barekatain 
Affiliation: OSX
"""

from __future__ import annotations

import math
from typing import List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class AstarOutput(NamedTuple):
    """
    Output structure of A* search planners
    """

    histories: torch.tensor
    paths: torch.tensor
    intermediate_results: Optional[List[dict]] = None


def get_heuristic(goal_maps: torch.tensor, tb_factor: float = 0.001) -> torch.tensor:
    """
    Get heuristic function for A* search (chebyshev + small const * euclidean)

    Args:
        goal_maps (torch.tensor): one-hot matrices of goal locations
        tb_factor (float, optional): small constant weight for tie-breaking. Defaults to 0.001.

    Returns:
        torch.tensor: heuristic function matrices
    """

    # some preprocessings to deal with mini-batches
    num_samples, H, W = goal_maps.shape[0], goal_maps.shape[-2], goal_maps.shape[-1]  #numero dei campioni, Altezza e lasrghezza delle matrici
    grid = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))  #griglia con righe 0 a H e colonne da 0 a W
    loc = torch.stack(grid, dim=0).type_as(goal_maps) #Vengono poi stackate insieme con lo stesso tipo di goal_maps
    loc_expand = loc.reshape(2, -1).unsqueeze(0).expand(num_samples, 2, -1) #Ridimensione a 2, -1(ultima dimensione), viene aggiunta la dimensione grande 1 e viene ripetuto per numsamples
    goal_loc = torch.einsum("kij, bij -> bk", loc, goal_maps)  #Somma di einstein:
    #Operazione per generalizzare operazioni tra matrici e tensori(perdiamo un po' di ottimizzazione solo in alcuni casi)
    #Moltiplicazione tra loc e goal maps che ha come dimensioni b e k: se loc e' 4, 5, 5 e goal maps e' 10, 5, 5 -> matrice 4 x 10
    goal_loc_expand = goal_loc.unsqueeze(-1).expand(num_samples, 2, -1) #Aggiunge una dimensione 1 alla fine, espande per num_samples

    # chebyshev distance
    dxdy = torch.abs(loc_expand - goal_loc_expand)  #chebyshev
    h = dxdy.sum(dim=1) - dxdy.min(dim=1)[0] #somma su dimensione - minimo sulla dimensione 1
    euc = torch.sqrt(((loc_expand - goal_loc_expand) ** 2).sum(1)) #distanza euclidea
    h = (h + tb_factor * euc).reshape_as(goal_maps) #euristica con la forma di goal_maps

    return h


def _st_softmax_noexp(val: torch.tensor) -> torch.tensor: #Softmax per trovare il nodo migliore
    """
    Softmax + discretized activation
    Used a detach() trick as done in straight-through softmax

    Args:
        val (torch.tensor): exponential of inputs.

    Returns:
        torch.tensor: one-hot matrices for input argmax.
    """

    val_ = val.reshape(val.shape[0], -1)
    print("val_, ", val_)
    y = val_ / (val_.sum(dim=-1, keepdim=True))
    _, ind = y.max(dim=-1)
    print("y, ",  y, " ind ", ind)
    y_hard = torch.zeros_like(y)
    print("y_hard, ", y_hard)
    y_hard[range(len(y_hard)), ind] = 1
    print("y_hard, ", y_hard)

    y_hard = y_hard.reshape_as(val)
    print("y_hard, ", y_hard)

    y = y.reshape_as(val)
    print("y2, ", y)

    return (y_hard - y).detach() + y


def expand(x: torch.tensor, neighbor_filter: torch.tensor) -> torch.tensor: #Nodi vicini a x, Neighbo filter : [[111], [101], [111]]
    """
    Expand neighboring node

    Args:
        x (torch.tensor): selected nodes
        neighbor_filter (torch.tensor): 3x3 filter to indicate 8 neighbors

    Returns:
        torch.tensor: neighboring nodes of x
    """

    x = x.unsqueeze(0)
    num_samples = x.shape[1]
    y = F.conv2d(x, neighbor_filter, padding=1, groups=num_samples).squeeze()
    y = y.squeeze(0)
    return y


def backtrack(
    start_maps: torch.tensor,
    goal_maps: torch.tensor,
    parents: torch.tensor, #Come funziona?
    current_t: int,
) -> torch.tensor: #Chiedere
    """
    Backtrack the search results to obtain paths

    Args:
        start_maps (torch.tensor): one-hot matrices for start locations
        goal_maps (torch.tensor): one-hot matrices for goal locations
        parents (torch.tensor): parent nodes
        current_t (int): current time step

    Returns:
        torch.tensor: solution paths
    """

    num_samples = start_maps.shape[0]
    parents = parents.type(torch.long)
    goal_maps = goal_maps.type(torch.long)
    start_maps = start_maps.type(torch.long)
    path_maps = goal_maps.type(torch.long)
    num_samples = len(parents)  ##Why?
    loc = (parents * goal_maps.view(num_samples, -1)).sum(-1)
    print(loc)
    print(parents.shape)
    for _ in range(current_t):
        path_maps.view(num_samples, -1)[range(num_samples), loc] = 1
        loc = parents[range(num_samples), loc]
    return path_maps


class DifferentiableAstar(nn.Module):
    def __init__(self, g_ratio: float = 0.5, Tmax: float = 1.0):
        """
        Differentiable A* module

        Args:
            g_ratio (float, optional): ratio between g(v) + h(v). Set 0 to perform as best-first search. Defaults to 0.5.
            Tmax (float, optional): how much of the map the planner explores during training. Defaults to 1.0.
        """

        super().__init__()

        neighbor_filter = torch.ones(1, 1, 3, 3)
        neighbor_filter[0, 0, 1, 1] = 0

        self.neighbor_filter = nn.Parameter(neighbor_filter, requires_grad=False)
        self.get_heuristic = get_heuristic

        self.g_ratio = g_ratio
        assert (Tmax > 0) & (Tmax <= 1), "Tmax must be within (0, 1]"
        self.Tmax = Tmax

    def forward(
        self,
        cost_maps: torch.tensor,
        start_maps: torch.tensor,
        goal_maps: torch.tensor,
        obstacles_maps: torch.tensor,
        store_intermediate_results: bool = False,
    ) -> AstarOutput:
        """
        Perform differentiable A* search

        Args:
            cost_maps (torch.tensor): cost maps
            start_maps (torch.tensor): start maps indicating the start location with one-hot binary map
            goal_maps (torch.tensor): goal maps indicating the goal location with one-hot binary map
            obstacle_maps (torch.tensor): binary maps indicating obstacle locations
            store_intermediate_results (bool, optional): If the intermediate search results are stored in Astar output. Defaults to False.

        Returns:
            AstarOutput: search histories and solution paths, and optionally intermediate search results.
        """

        assert cost_maps.ndim == 4
        assert start_maps.ndim == 4
        assert goal_maps.ndim == 4
        assert obstacles_maps.ndim == 4

        cost_maps = cost_maps[:, 0]
        start_maps = start_maps[:, 0]
        goal_maps = goal_maps[:, 0]
        obstacles_maps = obstacles_maps[:, 0]

        print("cost ", cost_maps)

        num_samples = start_maps.shape[0]
        neighbor_filter = self.neighbor_filter
        neighbor_filter = torch.repeat_interleave(neighbor_filter, num_samples, 0)
        size = start_maps.shape[-1]

        open_maps = start_maps
        histories = torch.zeros_like(start_maps)
        intermediate_results = []

        h = self.get_heuristic(goal_maps)
        h = h + cost_maps 
        g = torch.zeros_like(start_maps)

        parents = (
            #Vettori dei starting point
            torch.ones_like(start_maps).reshape(num_samples, -1)
            * goal_maps.reshape(num_samples, -1).max(-1, keepdim=True)[-1] #vettori dei goal, poi gli indici di essi
        )
        print("parent1: ", parents)

        size = cost_maps.shape[-1]
        Tmax = self.Tmax if self.training else 1.0
        Tmax = int(Tmax * size * size)
        for t in range(Tmax):

            # select the node that minimizes cost
            f = self.g_ratio * g + (1 - self.g_ratio) * h #f di a start con ratio tra g e h
            f_exp = torch.exp(-1 * f / math.sqrt(cost_maps.shape[-1])) #attivazione di hubara, con temperatura come radice della dimensione -1 dei costi [width]
            print("f_exp ", f_exp)
            f_exp = f_exp * open_maps #Scherma con i nodi aperti
            selected_node_maps = _st_softmax_noexp(f_exp) #Selezione nodo migliore
            print("Selected node maps", selected_node_maps)
            if store_intermediate_results:
                intermediate_results.append(
                    {
                        "histories": histories.unsqueeze(1).detach(),
                        "paths": selected_node_maps.unsqueeze(1).detach(),
                    }
                )

            # break if arriving at the goal
            dist_to_goal = (selected_node_maps * goal_maps).sum((1, 2), keepdim=True)
            is_unsolved = (dist_to_goal < 1e-8).float()

            histories = histories + selected_node_maps
            histories = torch.clamp(histories, 0, 1) #Min 0, max 1
            open_maps = open_maps - is_unsolved * selected_node_maps
            open_maps = torch.clamp(open_maps, 0, 1)

            # open neighboring nodes, add them to the openlist if they satisfy certain requirements
            neighbor_nodes = expand(selected_node_maps, neighbor_filter)
            neighbor_nodes = neighbor_nodes * obstacles_maps

            # update g if one of the following conditions is met
            # 1) neighbor is not in the close list (1 - histories) nor in the open list (1 - open_maps)
            # 2) neighbor is in the open list but g < g2
            g2 = expand((g + cost_maps) * selected_node_maps, neighbor_filter)
            idx = (1 - open_maps) * (1 - histories) + open_maps * (g > g2)
            idx = idx * neighbor_nodes
            idx = idx.detach()
            g = g2 * idx + g * (1 - idx)
            g = g.detach()

            # update open maps
            open_maps = torch.clamp(open_maps + idx, 0, 1)
            open_maps = open_maps.detach()
 
            # for backtracking
            idx = idx.reshape(num_samples, -1)
            snm = selected_node_maps.reshape(num_samples, -1)
            new_parents = snm.max(-1, keepdim=True)[1]
            print("idx: ", idx)
            print("newparents: ", new_parents)
            parents = new_parents * idx + parents * (1 - idx)
            print("parent2: ", parents)

            if torch.all(is_unsolved.flatten() == 0):
                break

        # backtracking
        path_maps = backtrack(start_maps, goal_maps, parents, t)

        if store_intermediate_results:
            intermediate_results.append(
                {
                    "histories": histories.unsqueeze(1).detach(),
                    "paths": path_maps.unsqueeze(1).detach(),
                }
            )

        return AstarOutput(
            histories.unsqueeze(1), path_maps.unsqueeze(1), intermediate_results
        )
