"""Iterated graph BFS reachability (Branch 1 Result AM candidate).

Per-example: random undirected graph on V_BFS nodes with random adjacency.
State at loop r = bitmask of nodes reachable in ≤ r hops from a chosen start node.

Per-step rule (position-self-referential per node, position-invariant):
  next[i] = current[i] OR ANY_j (adj[i,j] AND current[j])

Each output position's update uses ITS OWN current state plus its neighbors via
the adjacency matrix (which is per-example shared state). Satisfies Result N's
position-self-referential criterion → recipe should generalize.

Layout (length = V + V*V + 2 + V):
  positions 0..V-1            initial bitmask (V tokens, each 0 or 1)
  positions V..V+V*V-1        adjacency matrix flattened (V² binary entries)
  position  V+V*V             EQ
  positions V+V*V+1..V+V*V+V  output positions (predict bitmask each loop)
"""
from __future__ import annotations

import torch

from .data_chain import EQ, PAD, VOCAB_SIZE  # noqa: F401

V_BFS = 6                                       # 6 nodes
BFS_INPUT_LEN = V_BFS + V_BFS * V_BFS            # 6 + 36 = 42
BFS_SEQ_LEN = BFS_INPUT_LEN + 1 + V_BFS          # 42 + 1 + 6 = 49
BFS_OUTPUT_START = BFS_INPUT_LEN + 1             # 43


def make_batch_bfs_iter(batch_size: int, n_steps: int,
                          edge_density: float = 0.30,
                          device: str = "cuda"
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (tokens, iter_targets):
      tokens       : [B, BFS_SEQ_LEN]
      iter_targets : [n_steps+1, B, V_BFS]  (bitmask state at each loop)
    """
    # Random undirected adjacency (symmetric, no self-loops).
    rand = torch.rand((batch_size, V_BFS, V_BFS), device=device)
    adj = (rand < edge_density).long()
    # Symmetrize and zero diagonal
    adj = adj | adj.transpose(-2, -1)
    eye = torch.eye(V_BFS, device=device, dtype=torch.long).unsqueeze(0)
    adj = adj * (1 - eye)

    # Random start: exactly one node set initially.
    start_idx = torch.randint(0, V_BFS, (batch_size,), device=device)
    initial = torch.zeros((batch_size, V_BFS), dtype=torch.long, device=device)
    initial.scatter_(1, start_idx.unsqueeze(1), 1)

    # Iterate BFS for n_steps loops.
    iters = [initial.clone()]
    cur = initial.clone()
    for _ in range(n_steps):
        # cur: [B, V_BFS]; adj: [B, V_BFS, V_BFS]
        # next[i] = cur[i] OR (adj[i, j] AND cur[j] for any j)
        # = cur OR (adj * cur.unsqueeze(1)).any(-1)
        reachable = (adj * cur.unsqueeze(1)).any(dim=-1).long()
        cur = (cur | reachable).long()
        iters.append(cur.clone())
    iter_targets = torch.stack(iters, dim=0)   # [n_steps+1, B, V_BFS]

    # Build tokens. Adjacency tokens are 0/1; bitmask tokens are 0/1.
    tokens = torch.full((batch_size, BFS_SEQ_LEN), PAD,
                         dtype=torch.long, device=device)
    tokens[:, :V_BFS] = initial
    tokens[:, V_BFS:V_BFS + V_BFS * V_BFS] = adj.reshape(batch_size, -1)
    tokens[:, BFS_INPUT_LEN] = EQ
    # Output positions left as PAD; model fills via per-loop logits.
    return tokens, iter_targets
