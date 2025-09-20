import itertools

import plotly.graph_objects as go
from torch import Tensor

from binding_affinity.data.constants import ATOM_COLORS, ATOM_NAMES
from binding_affinity.data.types import AtomicInterfaceGraph


class PlotlyVis:
    @classmethod
    def create_figure(
        cls,
        graph: AtomicInterfaceGraph,
        receptor_color: str = "teal",
        ligand_color: str = "coral",
        figsize: tuple[int, int] = (900, 600),
    ) -> go.Figure:
        traces = cls.plot_graph(graph, receptor_color, ligand_color)
        width, height = figsize
        figure = go.Figure(
            data=traces,
            layout=dict(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                ),
                showlegend=False,
                width=width,
                height=height,
                # plot_bgcolor="rgba(0, 0, 0, 1)",
                # paper_bgcolor="rgba(0, 0, 0, 1)",
            ),
        )
        return figure

    @classmethod
    def plot_graph(
        cls,
        graph: AtomicInterfaceGraph,
        receptor_color: str,
        ligand_color: str,
    ) -> go.Figure:
        assert graph.coordinates is not None
        assert graph.edge_index is not None
        # получим ковалентные связи, чтобы нарисовать их по-другому
        ligand_cov, receptor_cov = cls.get_covalent_edges_masks(graph, distance_threshold=2.0)
        # нарисуем
        data = [
            # вершины рецептора
            cls.draw_nodes(graph, graph.is_receptor == 1),
            # вершины лиганда
            cls.draw_nodes(graph, graph.is_receptor == 0),
            # ковалентные связи лиганда
            cls.draw_edges(
                graph,
                edges_mask=ligand_cov,
                add_annotation=False,
                color=ligand_color,
                dash="solid",
                width=5,
            ),
            # ковалентные связи рецептора
            cls.draw_edges(
                graph,
                edges_mask=receptor_cov,
                add_annotation=False,
                color=receptor_color,
                dash="solid",
                width=5,
            ),
            # все связи в графе
            cls.draw_edges(
                graph,
                edges_mask=None,
                add_annotation=True,
                color="lightgray",
                dash="dot",
                width=1,
            ),
        ]
        return data

    @staticmethod
    def get_covalent_edges_masks(
        graph: AtomicInterfaceGraph, distance_threshold: float = 2.2
    ) -> list[Tensor]:
        src, tgt = graph.edge_index
        covalent_masks = []
        for chain_id in graph.is_receptor.unique():  # type: ignore[no-untyped-call]
            chain_atoms = graph.is_receptor == chain_id
            chain_edges = (
                chain_atoms[src] * chain_atoms[tgt] * (graph.distances <= distance_threshold)
            )
            covalent_masks.append(chain_edges)
        return covalent_masks

    @staticmethod
    def draw_nodes(graph: AtomicInterfaceGraph, nodes_mask: Tensor | None = None) -> go.Scatter3d:
        x, y, z = graph.coordinates[nodes_mask].T
        atom_types = [ATOM_NAMES[x.item()][0] for x in graph.atoms[nodes_mask]]
        atom_colors = [ATOM_COLORS[x] for x in atom_types]
        nodes = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            hoverinfo="text",
            text=[ATOM_NAMES[x.item()] for x in graph.atoms[nodes_mask]],
            marker=dict(
                size=4,
                color=atom_colors,
                cmin=0,
                cmax=1,
                opacity=0.8,
            ),
        )
        return nodes

    @staticmethod
    def draw_edges(
        graph: AtomicInterfaceGraph,
        edges_mask: Tensor | None = None,
        add_annotation: bool = False,
        color: str = "lightgray",
        dash: str = "dot",
        width: int = 1,
    ) -> go.Scatter3d:
        selected_edges, distances = graph.edge_index.T, graph.distances
        if edges_mask is not None:
            selected_edges = graph.edge_index.T[edges_mask]
            distances = graph.distances[edges_mask]

        edges_plot = go.Scatter3d(
            x=list(
                itertools.chain(
                    *(
                        (graph.coordinates[i, 0], graph.coordinates[j, 0], None)
                        for i, j in selected_edges
                    )
                )
            ),
            y=list(
                itertools.chain(
                    *(
                        (graph.coordinates[i, 1], graph.coordinates[j, 1], None)
                        for i, j in selected_edges
                    )
                )
            ),
            z=list(
                itertools.chain(
                    *(
                        (graph.coordinates[i, 2], graph.coordinates[j, 2], None)
                        for i, j in selected_edges
                    )
                )
            ),
            mode="lines",
            line=dict(
                color=color,
                width=width,
                dash=dash,
            ),
            text=(
                list(
                    itertools.chain(*((f"{d:.3f}Å", f"{d:.3f}Å", None) for d in distances.tolist()))
                )
                if add_annotation
                else None
            ),
            hoverinfo="text",
        )
        return edges_plot
