"""Quick integration test — verifies all stores, router, and consolidation work together."""
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models import (
    EpisodicMemoryCreate, SemanticMemoryCreate,
    ProceduralRuleCreate, EntityCreate, RelationshipCreate,
    EpisodicQueryRequest, SemanticQueryRequest,
    ProceduralQueryRequest, GraphQueryRequest,
    RuleDomain,
)
from router import MemoryRouter
from consolidation import ConsolidationWorker

TEST_DIR = "data_test"

def main():
    # Cleanup first
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    print("=== Enterprise Memory Platform — Integration Test ===\n")

    # 1. Init
    router = MemoryRouter(data_dir=TEST_DIR)
    worker = ConsolidationWorker(router)
    print("[OK] Router and worker initialised")

    # 2. Create tenants
    router.tenants.create_tenant("alpha", "Alpha Corp")
    router.tenants.create_tenant("beta", "Beta Inc")
    assert router.tenants.tenant_exists("alpha")
    assert router.tenants.tenant_exists("beta")
    print("[OK] 2 tenants created")

    # 3. Episodic
    router.store_episodic("alpha", EpisodicMemoryCreate(
        session_id="s1", content="Alpha's secret data", importance_score=0.9,
    ))
    router.store_episodic("beta", EpisodicMemoryCreate(
        session_id="s1", content="Beta's project plan", importance_score=0.7,
    ))
    assert router.episodic.count("alpha") == 1
    assert router.episodic.count("beta") == 1
    print("[OK] Episodic memories stored (isolated)")

    # 4. Tenant isolation check
    alpha_eps = router.episodic.query("alpha", EpisodicQueryRequest(limit=100))
    for ep in alpha_eps:
        assert "beta" not in ep.content.lower(), "ISOLATION BREACH!"
    beta_eps = router.episodic.query("beta", EpisodicQueryRequest(limit=100))
    for ep in beta_eps:
        assert "alpha" not in ep.content.lower(), "ISOLATION BREACH!"
    print("[OK] Multi-tenant isolation verified")

    # 5. Semantic
    router.store_semantic("alpha", SemanticMemoryCreate(
        fact="Alpha CEO is Alice", category="profile", confidence=0.95,
    ))
    result = router.query_semantic("alpha", SemanticQueryRequest(query="Who is CEO?"))
    assert result.count > 0
    print("[OK] Semantic store and search working")

    # 6. Procedural
    router.store_procedural("alpha", ProceduralRuleCreate(
        original_mistake="Used informal tone",
        correction="Use formal tone",
        rule_text="Always use formal language",
        domain=RuleDomain.TONE,
        confidence=0.9,
    ))
    assert router.procedural.count("alpha") == 1
    print("[OK] Procedural rule stored")

    # 7. Knowledge Graph
    router.add_entity("alpha", EntityCreate(name="Alice", entity_type="person"))
    router.add_entity("alpha", EntityCreate(name="Alpha Corp", entity_type="company"))
    router.add_relationship("alpha", RelationshipCreate(
        source="Alice", target="Alpha Corp", relation_type="ceo_of",
    ))
    stats = router.graph.get_stats("alpha")
    assert stats.total_entities == 2
    assert stats.total_relationships == 1
    print("[OK] Knowledge graph entities and relationships")

    # 8. Path finding
    path_result = router.find_path("alpha", "Alice", "Alpha Corp")
    assert path_result.count > 0
    print("[OK] Graph path finding")

    # 9. Consolidation
    record = worker.force_consolidation("alpha")
    print(f"[OK] Consolidation: {record.status.value} — {record.summary_text}")

    # 10. Platform stats
    platform = router.get_platform_stats()
    assert platform["total_tenants"] == 2
    print(f"[OK] Platform stats: {platform['total_tenants']} tenants, {platform['total_memories']} memories")

    # Cleanup
    shutil.rmtree(TEST_DIR, ignore_errors=True)

    print("\n=== ALL TESTS PASSED ===")

if __name__ == "__main__":
    main()
