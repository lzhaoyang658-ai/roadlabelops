SHELL := /bin/bash

PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ENV_FILE ?= $(PROJECT_ROOT)/.env
BACKEND_URL ?= http://127.0.0.1:8100
NEXT_PUBLIC_CVAT_BASE_URL ?= http://localhost:8080
NEXT_PUBLIC_SOURCE_URL ?= https://github.com/lzhaoyang658-ai/roadlabelops
NEXT_PROXY_CLIENT_MAX_BODY_SIZE ?= 2gb
BACKEND_HOST ?= 127.0.0.1
BACKEND_PORT ?= 8100
FRONTEND_HOST ?= 127.0.0.1
FRONTEND_PORT ?= 3100

.PHONY: bootstrap build-frontend production-bootstrap doctor doctor-demo serve-backend serve-frontend

bootstrap:
	cd "$(PROJECT_ROOT)" && ./setup.sh

build-frontend:
	cd "$(PROJECT_ROOT)" && \
		BACKEND_URL="$(BACKEND_URL)" \
		NEXT_PUBLIC_API_BASE_URL="/api/v1" \
		NEXT_PUBLIC_CVAT_BASE_URL="$(NEXT_PUBLIC_CVAT_BASE_URL)" \
		NEXT_PUBLIC_SOURCE_URL="$(NEXT_PUBLIC_SOURCE_URL)" \
		NEXT_PROXY_CLIENT_MAX_BODY_SIZE="$(NEXT_PROXY_CLIENT_MAX_BODY_SIZE)" \
		npm --prefix frontend run build

production-bootstrap: bootstrap build-frontend

doctor:
	cd "$(PROJECT_ROOT)" && \
		ROADLABELOPS_ENV_FILE="$(ENV_FILE)" .venv/bin/roadlabelops doctor

doctor-demo:
	cd "$(PROJECT_ROOT)" && \
		ROADLABELOPS_ENV_FILE="$(ENV_FILE)" .venv/bin/roadlabelops doctor --demo-only

serve-backend:
	cd "$(PROJECT_ROOT)" && \
		ROADLABELOPS_ENV_FILE="$(ENV_FILE)" \
		.venv/bin/uvicorn roadlabelops.api:app \
		--host "$(BACKEND_HOST)" --port "$(BACKEND_PORT)" --workers 1

serve-frontend:
	cd "$(PROJECT_ROOT)" && \
		HOSTNAME="$(FRONTEND_HOST)" PORT="$(FRONTEND_PORT)" npm --prefix frontend run start
