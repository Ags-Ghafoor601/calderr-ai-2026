# TechCorp Security Best Practices

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