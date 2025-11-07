# Environment Setup Guide

This guide explains how to set up and run the Zero Factory project in different environments (development and production).

## Overview

The project supports two environments:
- **Development** (dev): For local development with hot-reload, no SSL, direct port access
- **Production** (prod): For deployment with HTTPS, SSL certificates, and production optimizations

## File Structure

```
zero-factory/
├── docker-compose.yml           # Base configuration (common settings)
├── docker-compose.dev.yml       # Development overrides
├── docker-compose.prod.yml      # Production overrides
├── .env                         # Active environment file (auto-generated, gitignored)
├── .env.dev                     # Development environment variables
├── .env.prod                    # Production environment variables
├── .env.example                 # Example environment variables
├── nginx/
│   ├── nginx.conf               # Production nginx config (with HTTPS)
│   └── nginx.dev.conf           # Development nginx config (HTTP only)
├── start-dev.sh                 # Development environment script
└── start-prod.sh                # Production environment script
```

## Quick Start

### Option 1: Using Environment Scripts (Recommended)

```bash
# Development
./start-dev.sh          # Start in dev mode
./start-dev.sh down     # Stop dev services

# Production
./start-prod.sh         # Start in prod mode
./start-prod.sh down    # Stop prod services
```

### Option 2: Manual Docker Compose

You can also use docker compose directly:

```bash
# Development
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Environment Configuration

### Development Environment (.env.dev)

- **API URL**: `http://localhost:3000` (direct access, no nginx proxy)
- **Web URL**: `http://localhost:3001` (direct access)
- **Nginx**: HTTP only on port 80
- **Hot Reload**: Enabled for API and Web services
- **SSL**: Disabled
- **Certbot**: Disabled
- **Ports**: All service ports exposed for direct access
- **Database**: Uses separate volume `postgres-data-dev`

**Access URLs:**
- Main Application: http://localhost
- API (direct): http://localhost:3000
- Web (direct): http://localhost:3001
- Database: localhost:5432
- Label Studio: http://localhost/label-studio

### Production Environment (.env.prod)

- **API URL**: `https://zeromap.store/api` (via nginx proxy)
- **Web URL**: `https://zeromap.store` (via nginx proxy)
- **Nginx**: HTTPS on port 443, HTTP redirects to HTTPS
- **Hot Reload**: Disabled
- **SSL**: Enabled with Let's Encrypt certificates
- **Certbot**: Enabled for certificate renewal
- **Ports**: Only nginx ports exposed (80, 443)
- **Database**: Uses volume `postgres-data`

**Access URLs:**
- Main Application: https://zeromap.store
- API: https://zeromap.store/api
- Label Studio: https://zeromap.store/label-studio

## Environment Variables

### Required Variables

Both `.env.dev` and `.env.prod` require:

```bash
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=zerowaste_dev

# Admin
ADMIN_CODE=your_admin_code

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:3000  # or https://domain.com/api
NEXT_PUBLIC_KAKAO_MAP_KEY=your_kakao_map_key
```

### Customizing Secrets

**IMPORTANT**: Before deploying to production, update these values in `.env.prod`:

1. `POSTGRES_PASSWORD`: Use a strong, unique password
2. `ADMIN_CODE`: Use a secure admin code
3. `NEXT_PUBLIC_KAKAO_MAP_KEY`: Use your production API key (if different)

## Setting Up SSL (Production Only)

For production deployment with SSL:

1. Ensure your domain DNS points to your server
2. Update `.env.prod` with your domain
3. Run the SSL initialization script:

```bash
./init-letsencrypt.sh
```

4. Start production services:

```bash
./start-prod.sh
```

## Common Tasks

### Switching Environments

```bash
# Switch to development
cp .env.dev .env
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Switch to production
cp .env.prod .env
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Viewing Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f web
docker compose logs -f nginx
```

### Rebuilding Services

```bash
# Development
./start-dev.sh up -d --build

# Production
./start-prod.sh up -d --build
```

### Database Access

```bash
# Development (port 5432 exposed)
psql -h localhost -U postgres -d zerowaste_dev

# Production (via docker exec)
docker exec -it postgresdb psql -U postgres -d zerowaste_dev
```

### Cleaning Up

```bash
# Stop and remove containers
./start-dev.sh down

# Stop, remove containers, and delete volumes (WARNING: deletes data)
./start-dev.sh down -v

# Remove unused images
docker image prune -a
```

## Troubleshooting

### Port Conflicts

If ports 80, 443, 3000, 3001, or 5432 are already in use:

1. Stop conflicting services
2. Or modify port mappings in `docker-compose.dev.yml` or `docker-compose.prod.yml`

### SSL Certificate Issues

If SSL certificates fail to generate:

1. Ensure domain DNS is correctly configured
2. Check certbot logs: `docker compose logs certbot`
3. Try staging environment first (edit `init-letsencrypt.sh`)

### Services Not Starting

1. Check logs: `docker compose logs -f`
2. Verify environment file: `cat .env`
3. Ensure correct compose files are used
4. Check for port conflicts

### Hot Reload Not Working

In development, if code changes aren't reflected:

1. Ensure volumes are correctly mounted in `docker-compose.dev.yml`
2. Restart the specific service: `docker compose restart api`
3. Check service logs for errors

## CI/CD Integration

For automated deployments:

```bash
# In your CI/CD pipeline (e.g., GitHub Actions)
./start-prod.sh up -d --build
```

## Additional Resources

- Main README: [../README.md](../README.md)
- Docker Compose Documentation: https://docs.docker.com/compose/
- Nginx Documentation: https://nginx.org/en/docs/
