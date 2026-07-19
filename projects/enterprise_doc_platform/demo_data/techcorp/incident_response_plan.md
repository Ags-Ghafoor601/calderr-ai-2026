# TechCorp Incident Response Plan

## Incident Classification
**P1 — Critical**: Complete service outage, data breach, security vulnerability actively exploited. Response: Immediate. All-hands war room.
**P2 — High**: Major feature unavailable, significant performance degradation (> 50%), partial data loss. Response: Within 30 minutes.
**P3 — Medium**: Minor feature impacted, non-critical bug affecting < 10% users. Response: Within 4 hours.
**P4 — Low**: Cosmetic issue, feature request, minor improvement. Response: Next business day.

## Response Procedure

### 1. Detection & Triage (0-5 minutes)
Alert received via PagerDuty. On-call engineer assesses severity. Escalate P1/P2 to engineering manager. Create incident channel in Slack (#incident-YYYY-MM-DD-N).

### 2. Investigation (5-30 minutes)
Check dashboards: Datadog metrics, error logs, deployment history. Identify blast radius (affected users, regions, services). Determine root cause hypothesis. Decide: fix forward or rollback?

### 3. Mitigation (30-60 minutes)
Implement fix or rollback. Communicate status to stakeholders every 15 minutes. Update status page for customer-facing issues. Document actions taken in incident channel.

### 4. Resolution
Confirm service restored to normal. Verify via automated smoke tests and manual checks. Close incident channel summary. Schedule post-incident review within 48 hours.

### 5. Post-Incident Review
Blameless retrospective: What happened? Why? What do we change? Document: timeline, root cause, contributing factors, action items. Action items tracked in Jira with owners and deadlines. Share learnings in weekly engineering all-hands.

## Communication Templates
**Status Page Update**: "We are currently investigating [issue description]. [X]% of users may experience [impact]. We will provide updates every [interval]. Current status: [Investigating/Identified/Monitoring/Resolved]."