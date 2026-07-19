# TechCorp Deployment Runbook

## Pre-Deployment Checklist
1. All CI tests passing (unit, integration, e2e)
2. Code review approved by 2+ reviewers
3. Database migrations tested in staging
4. Performance benchmarks within 5% of baseline
5. Security scan completed (no critical/high findings)
6. Rollback plan documented
7. On-call engineer identified and available

## Deployment Process
### Stage 1: Canary (5% traffic)
Deploy to canary pod. Monitor for 15 minutes: error rate, latency p99, CPU/memory. If metrics degrade > 10%, automatic rollback. Manual approval required to proceed.

### Stage 2: Rolling (25% → 50% → 100%)
Progressive rollout over 30 minutes. Each step monitored for 10 minutes. Automated rollback on: error rate > 1%, latency p99 > 500ms, crash loop detected.

### Stage 3: Verification
Smoke tests run automatically. Dashboard verification by on-call engineer. Customer-impacting changes require product sign-off. Update deployment log and change management ticket.

## Rollback Procedure
1. Trigger rollback via: `kubectl rollout undo deployment/<service>`
2. Verify previous version pods are healthy
3. Check database compatibility (reverse migrations if needed)
4. Notify stakeholders via #deployments Slack channel
5. Post-incident review within 24 hours

## Emergency Hotfix
Skip canary for critical security patches. Requires VP Engineering approval. Deploy to all pods simultaneously. Post-deployment monitoring for 1 hour minimum.