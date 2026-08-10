"""
Enterprise AI Memory Platform — Knowledge Graph Store
=======================================================
NetworkX-backed per-tenant knowledge graphs with JSON
serialisation, entity deduplication, and multi-hop traversal.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import (
    EntityCreate, RelationshipCreate, GraphQueryRequest,
    GraphEntity, GraphRelationship, GraphStats,
)


class KnowledgeGraphStore:
    """Per-tenant knowledge graph store backed by NetworkX + JSON persistence."""

    def __init__(self, data_dir: str = "data/graphs"):
        self.data_dir = data_dir
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        self._graphs: dict[str, nx.DiGraph] = {}

    def _get_graph(self, tenant_id: str) -> nx.DiGraph:
        """Load or create the graph for a tenant."""
        if tenant_id not in self._graphs:
            path = os.path.join(self.data_dir, f"{tenant_id}_graph.json")
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._graphs[tenant_id] = nx.node_link_graph(data)
                except Exception:
                    self._graphs[tenant_id] = nx.DiGraph()
            else:
                self._graphs[tenant_id] = nx.DiGraph()
        return self._graphs[tenant_id]

    def _save_graph(self, tenant_id: str):
        """Save a tenant's graph to JSON."""
        graph = self._graphs.get(tenant_id)
        if graph is None:
            return
        path = os.path.join(self.data_dir, f"{tenant_id}_graph.json")
        data = nx.node_link_data(graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def add_entity(self, tenant_id: str, entity: EntityCreate) -> GraphEntity:
        """Add or update an entity in the tenant's graph."""
        graph = self._get_graph(tenant_id)
        name = entity.name.strip()

        # Deduplication: check existing aliases
        resolved = self._resolve_name(graph, name)
        if resolved and resolved != name:
            node_data = graph.nodes[resolved]
            aliases = set(node_data.get("aliases", []))
            aliases.add(name)
            aliases.update(entity.aliases)
            node_data["aliases"] = list(aliases)
            if entity.description:
                node_data["description"] = entity.description
            name = resolved
        elif name in graph.nodes:
            node_data = graph.nodes[name]
            existing = set(node_data.get("aliases", []))
            existing.update(entity.aliases)
            node_data["aliases"] = list(existing)
            if entity.description:
                node_data["description"] = entity.description
        else:
            graph.add_node(
                name,
                entity_type=entity.entity_type,
                description=entity.description,
                aliases=entity.aliases,
                metadata=entity.metadata,
            )

        self._save_graph(tenant_id)
        return GraphEntity(
            name=name,
            entity_type=graph.nodes[name].get("entity_type", "concept"),
            description=graph.nodes[name].get("description", ""),
            aliases=graph.nodes[name].get("aliases", []),
            connections=graph.degree(name),
        )

    def add_relationship(self, tenant_id: str, rel: RelationshipCreate) -> GraphRelationship:
        """Add a relationship to the tenant's graph."""
        graph = self._get_graph(tenant_id)

        source = self._resolve_name(graph, rel.source) or rel.source
        target = self._resolve_name(graph, rel.target) or rel.target

        if source not in graph.nodes:
            graph.add_node(source, entity_type="unknown", description="", aliases=[])
        if target not in graph.nodes:
            graph.add_node(target, entity_type="unknown", description="", aliases=[])

        graph.add_edge(
            source, target,
            relation_type=rel.relation_type,
            confidence=rel.confidence,
            evidence=rel.evidence,
        )

        self._save_graph(tenant_id)
        return GraphRelationship(
            source=source, target=target,
            relation_type=rel.relation_type,
            confidence=rel.confidence,
        )

    def query(self, tenant_id: str, request: GraphQueryRequest) -> dict[str, Any]:
        """Query the knowledge graph."""
        graph = self._get_graph(tenant_id)
        result: dict[str, Any] = {"entities": [], "relationships": [], "paths": []}

        if request.entity:
            resolved = self._resolve_name(graph, request.entity)
            if not resolved:
                return {"error": f"Entity '{request.entity}' not found", **result}

            node_data = dict(graph.nodes[resolved])
            result["center"] = {
                "name": resolved,
                "type": node_data.get("entity_type", "unknown"),
                "description": node_data.get("description", ""),
            }

            # Outgoing
            for _, tgt, edata in graph.out_edges(resolved, data=True):
                result["relationships"].append({
                    "source": resolved, "target": tgt,
                    "relation": edata.get("relation_type", "related_to"),
                    "confidence": edata.get("confidence", 0.8),
                })

            # Incoming
            for src, _, edata in graph.in_edges(resolved, data=True):
                result["relationships"].append({
                    "source": src, "target": resolved,
                    "relation": edata.get("relation_type", "related_to"),
                    "confidence": edata.get("confidence", 0.8),
                })

            # Multi-hop expansion
            if request.depth >= 2:
                visited = {resolved}
                frontier = [resolved]
                for _hop in range(request.depth - 1):
                    next_frontier = []
                    for node in frontier:
                        for _, nbr, edata in graph.out_edges(node, data=True):
                            if nbr not in visited:
                                visited.add(nbr)
                                next_frontier.append(nbr)
                                result["relationships"].append({
                                    "source": node, "target": nbr,
                                    "relation": edata.get("relation_type", ""),
                                    "confidence": edata.get("confidence", 0.5),
                                })
                        for nbr, _, edata in graph.in_edges(node, data=True):
                            if nbr not in visited:
                                visited.add(nbr)
                                next_frontier.append(nbr)
                                result["relationships"].append({
                                    "source": nbr, "target": node,
                                    "relation": edata.get("relation_type", ""),
                                    "confidence": edata.get("confidence", 0.5),
                                })
                    frontier = next_frontier

        return result

    def find_path(self, tenant_id: str, source: str, target: str) -> list[dict]:
        """Find shortest path between two entities."""
        graph = self._get_graph(tenant_id)
        src = self._resolve_name(graph, source)
        tgt = self._resolve_name(graph, target)
        if not src or not tgt:
            return []
        try:
            undirected = graph.to_undirected()
            path_nodes = nx.shortest_path(undirected, src, tgt)
            steps = []
            for i in range(len(path_nodes) - 1):
                a, b = path_nodes[i], path_nodes[i + 1]
                if graph.has_edge(a, b):
                    rel = graph.edges[a, b].get("relation_type", "related_to")
                elif graph.has_edge(b, a):
                    rel = f"(reverse) {graph.edges[b, a].get('relation_type', 'related_to')}"
                else:
                    rel = "connected_to"
                steps.append({"from": a, "to": b, "relation": rel})
            return steps
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_stats(self, tenant_id: str) -> GraphStats:
        """Get statistics for a tenant's graph."""
        graph = self._get_graph(tenant_id)
        entity_types: dict[str, int] = {}
        for _, data in graph.nodes(data=True):
            etype = data.get("entity_type", "unknown")
            entity_types[etype] = entity_types.get(etype, 0) + 1

        rel_types: dict[str, int] = {}
        for _, _, data in graph.edges(data=True):
            rtype = data.get("relation_type", "related_to")
            rel_types[rtype] = rel_types.get(rtype, 0) + 1

        return GraphStats(
            tenant_id=tenant_id,
            total_entities=graph.number_of_nodes(),
            total_relationships=graph.number_of_edges(),
            entity_types=entity_types,
            relationship_types=rel_types,
        )

    def get_all_entities(self, tenant_id: str) -> list[GraphEntity]:
        """Get all entities in a tenant's graph."""
        graph = self._get_graph(tenant_id)
        entities = []
        for node, data in graph.nodes(data=True):
            entities.append(GraphEntity(
                name=node,
                entity_type=data.get("entity_type", "concept"),
                description=data.get("description", ""),
                aliases=data.get("aliases", []),
                connections=graph.degree(node),
            ))
        return sorted(entities, key=lambda e: e.connections, reverse=True)

    def get_all_relationships(self, tenant_id: str) -> list[GraphRelationship]:
        """Get all relationships in a tenant's graph."""
        graph = self._get_graph(tenant_id)
        rels = []
        for src, tgt, data in graph.edges(data=True):
            rels.append(GraphRelationship(
                source=src, target=tgt,
                relation_type=data.get("relation_type", "related_to"),
                confidence=data.get("confidence", 0.8),
            ))
        return rels

    def delete_entity(self, tenant_id: str, entity_name: str) -> bool:
        """Delete an entity and its edges."""
        graph = self._get_graph(tenant_id)
        resolved = self._resolve_name(graph, entity_name)
        if not resolved:
            return False
        graph.remove_node(resolved)
        self._save_graph(tenant_id)
        return True

    def entity_count(self, tenant_id: str) -> int:
        return self._get_graph(tenant_id).number_of_nodes()

    def relationship_count(self, tenant_id: str) -> int:
        return self._get_graph(tenant_id).number_of_edges()

    def _resolve_name(self, graph: nx.DiGraph, name: str) -> Optional[str]:
        if name in graph.nodes:
            return name
        name_lower = name.lower()
        for node, data in graph.nodes(data=True):
            if node.lower() == name_lower:
                return node
            aliases = data.get("aliases", [])
            if any(a.lower() == name_lower for a in aliases):
                return node
        return None
