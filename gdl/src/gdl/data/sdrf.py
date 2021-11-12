# import logging
import numpy as np
import torch
from torch_geometric.utils import (
    to_networkx,
    from_networkx,
    to_dense_adj,
    remove_self_loops,
    to_undirected,
)

from gdl.curvature.cuda import ricci, ricci_post_delta
from gdl.data.base import BaseDataset

# _logger = logging.getLogger(__name__)


def softmax(a, tau=1):
    exp_a = np.exp(a * tau)
    return exp_a / exp_a.sum()


def sdrf_w_cuda(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
):
    edge_index = data.edge_index
    if is_undirected:
        edge_index = to_undirected(edge_index)
    A = to_dense_adj(remove_self_loops(edge_index)[0])[0]
    N = A.shape[0]
    G = to_networkx(data)
    if is_undirected:
        G = G.to_undirected()
    A = A.cuda()
    C = torch.zeros(N, N).cuda()

    for x in range(loops):
        can_add = True
        # _logger.warn(f'\n#######\nLoop {x}!\n#######\n')
        ricci(A, C=C)
        ix_min = C.argmin().item()
        x = ix_min // N
        y = ix_min % N
        # _logger.warn(f'Min curvature {C[x,y]} at {x}->{y}')

        if is_undirected:
            x_neighbors = list(G.neighbors(x)) + [x]
            y_neighbors = list(G.neighbors(y)) + [x]
        else:
            x_neighbors = list(G.successors(x)) + [x]
            y_neighbors = list(G.predecessors(y)) + [x]
        # _logger.warn(f'x has successors {x_neighbors}')
        # _logger.warn(f'y has successors {y_neighbors}')
        candidates = []
        for i in x_neighbors:
            for j in y_neighbors:
                if (i != j) and (not G.has_edge(i, j)):
                    candidates.append((i, j))

        if len(candidates):
            D = ricci_post_delta(A, x, y, x_neighbors, y_neighbors)
            improvements = []
            for (i, j) in candidates:
                improvements.append(
                    (D - C[x, y])[x_neighbors.index(i), y_neighbors.index(j)].item()
                )

            k, l = candidates[
                np.random.choice(
                    range(len(candidates)), p=softmax(np.array(improvements), tau=tau)
                )
            ]
            # _logger.warn(f'New edge chosen is {k,l}')
            G.add_edge(k, l)
            if is_undirected:
                A[k, l] = A[l, k] = 1
            else:
                A[k, l] = 1
        else:
            can_add = False
            if not remove_edges:
                # _logger.warn(f'Nothing changed this round - breaking')
                break

        if remove_edges:
            ix_max = C.argmax().item()
            x = ix_max // N
            y = ix_max % N
            if C[x, y] > removal_bound:
                # _logger.warn(f'Max curvature {C[x,y]} at {x}->{y} - removing edge')
                G.remove_edge(x, y)
                if is_undirected:
                    A[x, y] = A[y, x] = 0
                else:
                    A[x, y] = 0
            else:
                # _logger.warn(f'Max curvature is <= {removal_bound} - leaving in place')
                if can_add is False:
                    # _logger.warn(f'Nothing changed this round - breaking')
                    break

    return from_networkx(G)


class SDRFDataset(BaseDataset):
    """
    Dataset preprocessed with SDRF (Cuda version).
    """

    def __init__(
        self,
        name: str = "Cora",
        use_lcc: bool = True,
        max_steps: int = None,
        remove_edges: bool = True,
        removal_bound: float = 0.5,
        tau: float = 1,
        undirected: bool = False,
        data_dir: str = None,
    ):
        self.name = name
        self.use_lcc = use_lcc
        self.max_steps = max_steps
        self.remove_edges = remove_edges
        self.removal_bound = removal_bound
        self.tau = tau
        self.undirected = undirected
        super(SDRFDataset, self).init(data_dir)

    def process(self):
        base = self.get_dataset()
        altered_data = sdrf_w_cuda(
            base.data,
            loops=self.max_steps,
            remove_edges=self.remove_edges,
            tau=self.tau,
            is_undirected=self.is_undirected,
        )
        edge_index = altered_data.edge_index
        self.to_dataset(base, edge_index, None)

    def __str__(self) -> str:
        return (
            f"{self.name}_sdrf_ms={self.max_steps}_re={self.remove_edges}_rb={self.removal_bound}_tau={self.tau}_lcc={self.use_lcc}"
            + ("_undirected" if self.undirected else "")
        )
