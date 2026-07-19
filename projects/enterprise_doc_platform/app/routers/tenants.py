"""Tenant management API endpoints."""

from fastapi import APIRouter, HTTPException

from app.models import TenantCreate, TenantResponse, PlatformStats
from app.services.document_processor import (
    create_tenant, get_tenant, list_tenants, delete_tenant, list_tenant_documents,
)
from app.services.vector_store import vector_store

router = APIRouter(prefix="/tenants", tags=["Tenants"])


@router.post("/", response_model=TenantResponse, status_code=201)
async def create_new_tenant(tenant: TenantCreate):
    """Create a new tenant with an isolated document namespace."""
    result = create_tenant(tenant.name, tenant.description)
    vector_store.get_or_create_collection(result["tenant_id"])

    stats = vector_store.get_tenant_stats(result["tenant_id"])
    return TenantResponse(
        tenant_id=result["tenant_id"],
        name=result["name"],
        description=result["description"],
        document_count=stats["document_count"],
        chunk_count=stats["chunk_count"],
        created_at=result["created_at"],
    )


@router.get("/", response_model=list[TenantResponse])
async def list_all_tenants():
    """List all registered tenants with their stats."""
    tenants = list_tenants()
    responses = []
    for t in tenants:
        stats = vector_store.get_tenant_stats(t["tenant_id"])
        responses.append(TenantResponse(
            tenant_id=t["tenant_id"],
            name=t["name"],
            description=t["description"],
            document_count=stats["document_count"],
            chunk_count=stats["chunk_count"],
            created_at=t["created_at"],
        ))
    return responses


@router.get("/{tenant_id}", response_model=TenantResponse)
async def get_tenant_details(tenant_id: str):
    """Get details for a specific tenant."""
    tenant = get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")

    stats = vector_store.get_tenant_stats(tenant_id)
    return TenantResponse(
        tenant_id=tenant["tenant_id"],
        name=tenant["name"],
        description=tenant["description"],
        document_count=stats["document_count"],
        chunk_count=stats["chunk_count"],
        created_at=tenant["created_at"],
    )


@router.delete("/{tenant_id}")
async def delete_existing_tenant(tenant_id: str):
    """Delete a tenant and all their documents."""
    if not get_tenant(tenant_id):
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")

    vector_store.delete_tenant_collection(tenant_id)
    delete_tenant(tenant_id)
    return {"message": f"Tenant '{tenant_id}' deleted successfully"}


@router.get("/stats/overview", response_model=PlatformStats)
async def platform_overview():
    """Get platform-wide statistics."""
    tenants = list_tenants()
    total_docs = 0
    total_chunks = 0
    tenant_responses = []

    for t in tenants:
        stats = vector_store.get_tenant_stats(t["tenant_id"])
        total_docs += stats["document_count"]
        total_chunks += stats["chunk_count"]
        tenant_responses.append(TenantResponse(
            tenant_id=t["tenant_id"],
            name=t["name"],
            description=t["description"],
            document_count=stats["document_count"],
            chunk_count=stats["chunk_count"],
            created_at=t["created_at"],
        ))

    return PlatformStats(
        total_tenants=len(tenants),
        total_documents=total_docs,
        total_chunks=total_chunks,
        tenants=tenant_responses,
    )
