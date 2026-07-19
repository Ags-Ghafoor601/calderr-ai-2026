# TechCorp API Documentation

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