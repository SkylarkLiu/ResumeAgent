---
name: resumeagent-ops
description: Use when the user wants to operate, validate, debug, or maintain the ResumeAgent system, including Docker deployment, health checks, FAISS metadata migration, cache backends, and knowledge-base management.
---

# ResumeAgent Operations

Use this skill when the task is to operate or debug the ResumeAgent system.

This skill is appropriate when the user:

- wants to start or debug Docker deployment
- wants to verify health, checkpointer, metadata store, or expert cache
- wants to inspect ingest, sources, documents, or FAISS compact behavior
- wants to diagnose environment, build, or persistence issues

## Core Runtime Areas

- FastAPI app
- PostgreSQL
- checkpointer backend
- expert cache backend
- FAISS index
- PostgreSQL metadata store

## Operational Workflow

1. Check container and process status.
2. Check `/health` and confirm:
   - `checkpointer_backend`
   - `expert_cache_backend`
   - `metadata_store_backend`
3. For knowledge-base issues, inspect:
   - `/ingest/sources`
   - `/ingest/documents`
   - `/ingest/compact`
4. For persistence issues, verify:
   - PostgreSQL tables
   - `data/faiss_index`
   - `row_map.json`
   - migration status from legacy `metadata.json`
5. For build issues, separate the problem into:
   - Docker Hub
   - Debian mirror
   - pip mirror
   - Python package compatibility

## ResumeAgent-specific Checks

When validating deployment, prefer these checks:

```bash
docker compose ps
docker compose logs --tail=100 app
curl http://localhost:8000/health
```

When validating knowledge-base metadata:

```bash
curl http://localhost:8000/ingest/sources
curl http://localhost:8000/ingest/documents
curl -X POST http://localhost:8000/ingest/compact
```

## Output Guidelines

- Identify the failing layer first: build, runtime, database, cache, or retrieval.
- Prefer concrete commands and expected outputs.
- Keep explanations short and operational.
- If there is a mismatch between intended architecture and live behavior, call that out explicitly.
