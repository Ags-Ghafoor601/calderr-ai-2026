#!/usr/bin/env python3
"""Generate demo data for 3 tenants: Acme Legal, MedCare Clinic, TechCorp."""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent

TENANT_DOCS = {
    "acme_legal": {
        "employment_contract.md": """# Standard Employment Contract

## Article 1 — Parties
This Employment Contract is entered into between Acme Legal Services LLC ("Employer") and the Employee identified in Schedule A.

## Article 2 — Position and Duties
The Employee shall serve as a full-time associate in the capacity specified. Duties include legal research, client consultation, document drafting, and court representation as assigned.

## Article 3 — Compensation
Base salary shall be as specified in Schedule A, paid bi-weekly. Performance bonuses may be awarded quarterly based on billable hours and client satisfaction metrics.

## Article 4 — Benefits
The Employee is entitled to health insurance (medical, dental, vision), 401(k) matching up to 6%, 15 days PTO, and professional development allowance of $5,000 per year.

## Article 5 — Confidentiality
The Employee agrees to maintain strict confidentiality of all client information, case details, and firm proprietary data during and after employment. Violation constitutes grounds for immediate termination.

## Article 6 — Non-Compete
For 12 months following termination, the Employee shall not practice within the same specialization in the same metropolitan area without written consent from Acme Legal.

## Article 7 — Termination
Either party may terminate with 30 days written notice. Termination for cause (misconduct, breach) is effective immediately. Severance of 2 weeks per year of service applies to termination without cause.
""",
        "privacy_policy.md": """# Acme Legal Services — Privacy Policy

## Data Collection
We collect personal information necessary to provide legal services: name, contact details, case-related documents, financial information for billing, and communication records.

## Data Use
Personal data is used exclusively for: providing legal representation, billing and invoicing, regulatory compliance, conflict of interest checks, and case management.

## Data Storage
All client data is stored in encrypted, access-controlled systems. Physical documents are maintained in locked, fire-resistant storage. Digital records use AES-256 encryption at rest and TLS 1.3 in transit.

## Data Retention
Client files are retained for 7 years after case closure per bar association requirements. Financial records are maintained for 10 years per tax regulations. Upon request, non-required data will be deleted within 30 days.

## Third-Party Sharing
Client data is never sold. Sharing occurs only with: opposing counsel (as required by proceedings), court systems, regulatory bodies, and authorized third-party service providers bound by confidentiality agreements.

## Client Rights
Clients may request access to their data, correction of inaccuracies, deletion of non-required records, and data portability in standard formats. Requests are processed within 15 business days.
""",
        "terms_of_service.md": """# Terms of Service — Acme Legal Services

## Scope of Services
Acme Legal provides corporate law, intellectual property, employment law, and regulatory compliance services. Specific scope is defined in each client engagement letter.

## Fees and Billing
Hourly rates range from $250-$600 depending on attorney seniority. Fixed-fee arrangements are available for routine matters. Invoices are issued monthly with net-30 payment terms. Late payments incur 1.5% monthly interest.

## Client Responsibilities
Clients must provide truthful, complete information. Failure to disclose material facts may result in withdrawal of representation. Clients must respond to communications within 5 business days during active matters.

## Limitation of Liability
Acme Legal's liability is limited to the fees paid for the specific matter in question. We are not liable for outcomes, as legal proceedings involve inherent uncertainty. Professional liability insurance coverage is maintained at $10M per occurrence.

## Dispute Resolution
Disputes between Acme Legal and clients shall be resolved through mediation first, then binding arbitration under AAA Commercial Rules. The prevailing party is entitled to reasonable attorneys' fees.

## Governing Law
These terms are governed by the laws of the State of California, without regard to conflict of laws principles.
""",
        "nda_template.md": """# Mutual Non-Disclosure Agreement

## Purpose
This Mutual NDA protects confidential information exchanged between the parties during business discussions, negotiations, and potential collaboration.

## Definition of Confidential Information
Confidential Information includes: trade secrets, business plans, financial data, customer lists, technical specifications, software code, marketing strategies, and any information marked "Confidential."

## Exclusions
Information is not confidential if it: was publicly available before disclosure, becomes public through no fault of the receiving party, was already known to the receiving party, or is independently developed without use of confidential information.

## Obligations
Each party agrees to: protect confidential information with at least the same care as their own confidential information, limit access to employees with a need-to-know, not disclose to third parties without written consent, and use information only for the stated purpose.

## Duration
Confidentiality obligations survive for 3 years from the date of disclosure. Trade secrets remain protected indefinitely under applicable law.

## Remedies
Breach may result in irreparable harm. The disclosing party is entitled to injunctive relief in addition to any other legal remedies. The breaching party shall bear all legal costs.
""",
        "employee_handbook.md": """# Acme Legal — Employee Handbook

## Work Hours
Standard hours are 9:00 AM to 6:00 PM, Monday through Friday. Attorneys are expected to bill a minimum of 1,800 hours annually. Flexible scheduling is available with supervisor approval.

## Code of Conduct
All employees must adhere to the highest ethical standards. The firm follows ABA Model Rules of Professional Conduct. Conflicts of interest must be reported immediately. Pro bono work is encouraged — 50 hours annually minimum.

## Leave Policy
PTO: 15 days for 0-3 years, 20 days for 3-7 years, 25 days for 7+ years. Sick leave: 10 days annually. Parental leave: 12 weeks paid. Bereavement: 5 days. Bar exam preparation: 2 weeks paid.

## Technology Policy
Firm-issued devices must be used for all client work. Personal devices may not store client data. Two-factor authentication is mandatory. Unapproved software installation is prohibited. All communications are subject to monitoring for compliance.

## Professional Development
Annual CLE requirements must be met. The firm provides: $5,000 annual education budget, conference attendance (2 per year), mentorship program for associates, and quarterly skills workshops.

## Anti-Harassment Policy
Acme Legal maintains zero tolerance for harassment, discrimination, and retaliation. Reports can be made to HR, the ethics committee, or via anonymous hotline. All reports are investigated within 5 business days.
""",
    },

    "medcare_clinic": {
        "patient_intake_guidelines.md": """# MedCare Clinic — Patient Intake Guidelines

## Registration Process
New patients complete digital intake forms including: demographic information, insurance details, medical history questionnaire, current medications list, allergy information, and emergency contacts.

## Insurance Verification
Insurance must be verified before the first appointment. We accept: Medicare, Medicaid, Blue Cross/Blue Shield, Aetna, Cigna, United Healthcare, and most PPO plans. Self-pay patients receive a 20% discount with payment at time of service.

## Medical History Review
The intake nurse reviews: past surgeries and hospitalizations, chronic conditions, family medical history (parents, siblings), current medications with dosages, known allergies and adverse reactions, and immunization records.

## Triage Protocol
Patients are triaged by severity: RED (life-threatening, immediate), ORANGE (urgent, within 15 minutes), YELLOW (semi-urgent, within 30 minutes), GREEN (non-urgent, within 60 minutes), BLUE (scheduled routine care).

## Consent Forms
Required consents include: general treatment consent, HIPAA privacy acknowledgment, financial responsibility agreement, authorization for insurance billing, and telehealth consent (if applicable).
""",
        "diabetes_management.md": """# Diabetes Management Protocol — MedCare Clinic

## Type 2 Diabetes Standards of Care

### Diagnosis
Confirmed by: HbA1c ≥ 6.5%, fasting plasma glucose ≥ 126 mg/dL, or 2-hour OGTT ≥ 200 mg/dL. Two abnormal tests required for diagnosis in asymptomatic patients.

### Treatment Goals
Target HbA1c: < 7.0% for most adults. Individualize based on: duration of diabetes, life expectancy, comorbidities, and hypoglycemia risk. Fasting glucose target: 80-130 mg/dL. Post-meal glucose target: < 180 mg/dL.

### First-Line Treatment
Metformin 500mg once daily, titrate to 2000mg daily over 4 weeks. Lifestyle modifications: medical nutrition therapy, 150 minutes/week moderate exercise, weight loss goal of 5-10% body weight.

### Monitoring Schedule
HbA1c every 3 months until stable, then every 6 months. Comprehensive metabolic panel annually. Lipid panel annually. Urine albumin-to-creatinine ratio annually. Dilated eye exam annually. Foot examination every visit.

### Complications Screening
Retinopathy: annual dilated eye exam. Nephropathy: annual urine albumin, GFR. Neuropathy: annual monofilament test. Cardiovascular: annual lipids, blood pressure at every visit. Depression: PHQ-9 screening annually.
""",
        "hypertension_guide.md": """# Hypertension Treatment Guide — MedCare Clinic

## Classification
Normal: < 120/80 mmHg. Elevated: 120-129/< 80 mmHg. Stage 1: 130-139/80-89 mmHg. Stage 2: ≥ 140/≥ 90 mmHg. Hypertensive crisis: > 180/120 mmHg (requires immediate evaluation).

## Initial Assessment
Complete blood count, comprehensive metabolic panel, lipid panel, thyroid function, urinalysis, ECG. Assess for secondary causes if: onset before age 30, resistant to treatment, or sudden worsening.

## Non-Pharmacological Management
DASH diet (fruits, vegetables, low sodium < 2300mg/day). Regular aerobic exercise (150 min/week). Weight management (BMI < 25). Limit alcohol (≤ 2 drinks/day men, ≤ 1 women). Smoking cessation. Stress management techniques.

## Pharmacological Treatment
First-line agents: ACE inhibitors, ARBs, calcium channel blockers, thiazide diuretics. Start with single agent at low dose. If not at goal in 1 month, add second agent from different class. Target: < 130/80 for most patients, < 140/90 for ≥ 65 years.

## Follow-up Protocol
Monthly visits until BP at goal. Then every 3-6 months. Home BP monitoring recommended. Labs every 6-12 months (renal function, electrolytes). Medication adherence assessment at each visit.
""",
        "vaccination_schedule.md": """# MedCare Clinic — Adult Vaccination Schedule

## Routine Adult Vaccines
Influenza: Annually, September-October preferred. COVID-19: Per current CDC guidance, updated formulations. Tdap/Td: Tdap once if not previously received, then Td booster every 10 years.

## Age-Based Recommendations
Ages 19-26: HPV vaccine series (if not completed). Ages 50+: Shingrix (recombinant zoster) — 2 doses, 2-6 months apart. Ages 65+: Pneumococcal vaccines PCV20 or PCV15+PPSV23. Annual influenza with high-dose formulation for 65+.

## Risk-Based Vaccines
Hepatitis B: Healthcare workers, dialysis patients, chronic liver disease. Hepatitis A: Travel to endemic areas, chronic liver disease. Meningococcal: Complement deficiency, asplenia, college students in dorms.

## Pre-Travel Vaccines
Typhoid, Yellow Fever, Japanese Encephalitis, Rabies — based on destination. Recommend travel consultation 4-6 weeks before departure. Malaria prophylaxis as indicated.

## Contraindications and Precautions
Live vaccines contraindicated in immunocompromised patients. Egg allergy: can receive influenza vaccine with observation. Previous anaphylaxis to vaccine component: avoid that vaccine. Pregnancy: avoid live vaccines; influenza and COVID-19 recommended.
""",
        "emergency_procedures.md": """# Emergency Procedures — MedCare Clinic

## Cardiac Arrest (Code Blue)
1. Call 911 immediately. 2. Begin CPR (30:2 ratio). 3. Apply AED within 2 minutes. 4. Continue until EMS arrival. AED locations: front reception, hallway B, procedure room. CPR-certified staff on every shift.

## Anaphylaxis Protocol
Signs: hives, swelling, difficulty breathing, hypotension. Treatment: Epinephrine 0.3mg IM (auto-injector in each exam room). Call 911. Position patient supine with legs elevated. Monitor vitals every 5 minutes. Second epi dose in 5-15 minutes if no improvement.

## Stroke Recognition (BE-FAST)
Balance loss, Eyes (vision changes), Face drooping, Arm weakness, Speech difficulty, Time to call 911. Document onset time. Do not give food/water. Maintain airway. Transport to nearest stroke center.

## Severe Bleeding
Apply direct pressure with sterile gauze. Elevate injured area above heart level. Apply tourniquet if direct pressure fails (note time). Call 911 for uncontrolled bleeding. Do not remove embedded objects.

## Fire Evacuation
RACE: Rescue, Alarm, Contain, Evacuate. Pull fire alarm. Evacuate patients via designated routes (posted on each floor). Assembly point: north parking lot. Account for all patients and staff. Do not use elevators.
""",
    },

    "techcorp": {
        "api_documentation.md": """# TechCorp API Documentation

## Overview
The TechCorp Platform API provides RESTful endpoints for user management, data processing, and analytics. Base URL: `https://api.techcorp.io/v2`. Authentication via Bearer tokens (OAuth 2.0).

## Authentication
POST /auth/token — Obtain access token. Requires client_id and client_secret. Tokens expire after 1 hour. Refresh tokens valid for 30 days. Rate limit: 100 requests/minute per token.

## User Endpoints
GET /users — List users (paginated, max 100). POST /users — Create user (requires admin role). GET /users/{id} — Get user details. PUT /users/{id} — Update user. DELETE /users/{id} — Soft delete (30-day retention).

## Data Processing
POST /jobs — Submit processing job. GET /jobs/{id} — Check job status. GET /jobs/{id}/results — Download results (available 7 days). Supported formats: CSV, JSON, Parquet. Max file size: 500MB per job.

## Analytics
GET /analytics/dashboard — Dashboard metrics. GET /analytics/reports — Generated reports. POST /analytics/custom — Custom query (SQL-like syntax). Results cached for 15 minutes.

## Error Handling
400: Bad Request (validation error). 401: Unauthorized (invalid/expired token). 403: Forbidden (insufficient permissions). 404: Not Found. 429: Rate Limited (retry after header). 500: Internal Server Error (contact support).

## Webhooks
POST /webhooks — Register webhook URL. Events: job.completed, job.failed, user.created, alert.triggered. Retry policy: 3 attempts with exponential backoff (1s, 4s, 16s).
""",
        "system_architecture.md": """# TechCorp System Architecture

## Overview
TechCorp operates a microservices architecture deployed on AWS using Kubernetes (EKS). The system processes 50 million API requests daily with 99.99% uptime SLA.

## Service Components
- **API Gateway** (Kong): Request routing, rate limiting, authentication
- **User Service**: User management, RBAC, SSO integration
- **Data Pipeline**: Apache Kafka → Apache Spark → S3 → Snowflake
- **Analytics Engine**: Real-time dashboards, custom reports, ML predictions
- **Notification Service**: Email, SMS, push, webhooks

## Data Flow
1. Client requests hit API Gateway (Kong on EKS)
2. Gateway authenticates via OAuth 2.0 / JWT validation
3. Requests routed to appropriate microservice
4. Services communicate via gRPC (internal) or REST (external)
5. Event-driven workflows through Apache Kafka
6. Data stored in PostgreSQL (transactional), Redis (cache), S3 (blob storage)
7. Analytics queries run against Snowflake data warehouse

## Infrastructure
- **Compute**: EKS clusters (3 regions: us-east-1, eu-west-1, ap-southeast-1)
- **Database**: Aurora PostgreSQL (multi-AZ), ElastiCache Redis
- **Storage**: S3 with lifecycle policies (hot/warm/cold tiers)
- **CDN**: CloudFront for static assets and API caching
- **Monitoring**: Datadog (metrics), PagerDuty (alerting), Jaeger (tracing)

## Scaling Strategy
Horizontal pod autoscaling based on CPU/memory (70% threshold). Kafka consumer groups auto-scale with partition count. Database read replicas in each region. Circuit breakers prevent cascade failures (Hystrix pattern).
""",
        "deployment_runbook.md": """# TechCorp Deployment Runbook

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
""",
        "security_best_practices.md": """# TechCorp Security Best Practices

## Authentication & Authorization
All services use OAuth 2.0 with PKCE for public clients. JWT tokens signed with RS256. Token lifetime: 1 hour access, 30 days refresh. Role-Based Access Control (RBAC) with principle of least privilege. Multi-factor authentication required for admin roles.

## Data Protection
Encryption at rest: AES-256 for all databases and S3 buckets. Encryption in transit: TLS 1.3 for all communications. PII fields encrypted at application level (field-level encryption). Key management via AWS KMS with annual rotation.

## Secure Development
OWASP Top 10 training for all engineers (annual). Static analysis (SonarQube) in CI pipeline. Dynamic analysis (OWASP ZAP) weekly scans. Dependency vulnerability scanning (Snyk) continuous. Code review required for security-sensitive changes.

## Network Security
VPC with private subnets for all services. Security groups with minimal port exposure. WAF rules for OWASP protection. DDoS mitigation via AWS Shield Advanced. VPN required for production access.

## Incident Response
Severity levels: P1 (data breach, service down), P2 (degraded service), P3 (minor issue), P4 (improvement). P1 response time: 15 minutes. Post-incident review required for P1/P2. All incidents logged in PagerDuty.

## Compliance
SOC 2 Type II certified (annual audit). GDPR compliant for EU data processing. CCPA compliant for California residents. HIPAA BAA available for healthcare customers. Annual penetration testing by third party.
""",
        "incident_response_plan.md": """# TechCorp Incident Response Plan

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
""",
    },
}


def main():
    for tenant, docs in TENANT_DOCS.items():
        tenant_dir = BASE / tenant
        tenant_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in docs.items():
            (tenant_dir / filename).write_text(content.strip(), encoding="utf-8")
        print(f"  Created {len(docs)} documents for tenant: {tenant}")

    print(f"\nTotal: {sum(len(d) for d in TENANT_DOCS.values())} demo documents across {len(TENANT_DOCS)} tenants")


if __name__ == "__main__":
    main()
