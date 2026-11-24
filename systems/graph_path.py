import networkx as nx
import numpy as np
import polars as pl
from sklearn.metrics.pairwise import cosine_distances


def build_graph(df: pl.DataFrame, max_distance: float = 0.95) -> tuple[nx.DiGraph, np.ndarray]:
    emb = np.array(df["dino_embeddings"].to_list(), dtype=np.float32)
    dist = cosine_distances(emb)
    graph = nx.DiGraph()
    graph.add_nodes_from((i, {"prize": p}) for i, p in enumerate(df["aesthetic"].to_numpy()))
    for i, j in np.argwhere(np.triu(dist < max_distance, k=1)):
        graph.add_edge(int(i), int(j), weight=float(dist[i, j]))
    return graph, dist


def _extend(graph: nx.DiGraph, states: dict, budget: float, beam: int) -> dict:
    nxt: dict[int, list] = {}
    for last, paths in states.items():
        for prize, cost, path in paths:
            for m in graph.successors(last):
                new_cost = cost + graph[last][m]["weight"]
                if new_cost <= budget:
                    nxt.setdefault(m, []).append((prize + graph.nodes[m]["prize"], new_cost, path + [m]))
    return {m: sorted(p, key=lambda s: -s[0])[:beam] for m, p in nxt.items()}


def best_budget_path(graph: nx.DiGraph, k: int, budget: float, beam: int = 50) -> dict:
    """Maximize summed node prizes over k nodes with total edge cost <= budget.

    Beam search: keep the `beam` best partial paths for each last node.
    """
    states = {n: [(graph.nodes[n]["prize"], 0.0, [n])] for n in graph.nodes}
    for _ in range(k - 1):
        states = _extend(graph, states, budget, beam)
    finished = [s for paths in states.values() for s in paths]
    if not finished:
        return {"path": None, "prize": None, "cost": None}
    prize, cost, path = max(finished, key=lambda s: s[0])
    return {"path": path, "prize": prize, "cost": cost}


def sort_by_time(df: pl.DataFrame) -> pl.DataFrame:
    taken = pl.col("datetaken").str.to_datetime(strict=False)
    return df.sort(taken, nulls_last=True)


def select_graph_path(
    df: pl.DataFrame, k: int, budget: float, beam: int = 50, by_time: bool = False
) -> pl.DataFrame:
    """With by_time, edges go from earlier to later photos, so the path is in time order."""
    df = sort_by_time(df) if by_time else df
    graph, _ = build_graph(df)
    path = best_budget_path(graph, k, budget, beam)["path"]
    return df.clear() if path is None else df[path, :]
