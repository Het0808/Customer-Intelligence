# Azure Deployment Guide — Customer Intelligence Platform

> **Project context:** FastAPI app + Docker + MLflow + FAISS + sentence-transformers.
> This guide deploys the containerised API to **Azure Container Apps** (recommended)
> with fallback instructions for **Azure App Service (Web App for Containers)**.
> Estimated time: 45–60 minutes for a first deployment.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Understand Your Options](#2-understand-your-options)
3. [Prepare the Project](#3-prepare-the-project)
4. [Create Azure Resources](#4-create-azure-resources)
5. [Build and Push the Container Image](#5-build-and-push-the-container-image)
6. [Deploy to Azure Container Apps](#6-deploy-to-azure-container-apps)
7. [Configure Environment Variables and Secrets](#7-configure-environment-variables-and-secrets)
8. [Set Up CI/CD with GitHub Actions](#8-set-up-cicd-with-github-actions)
9. [Verify and Test the Deployment](#9-verify-and-test-the-deployment)
10. [Custom Domain and HTTPS](#10-custom-domain-and-https)
11. [Monitoring and Logs](#11-monitoring-and-logs)
12. [Cost Management](#12-cost-management)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

### Accounts and tools you need before starting

| Requirement | Why | Get it |
|---|---|---|
| Azure account | Hosts everything | [portal.azure.com](https://portal.azure.com) — free tier available |
| Azure CLI | All commands in this guide use it | [Install guide](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) |
| Docker Desktop | Build and test the image locally | [docker.com](https://www.docker.com/products/docker-desktop/) |
| GitHub account | CI/CD pipeline | Already have one (your repo is there) |

### Verify your tools

```bash
# Check Azure CLI version (need 2.50+)
az --version

# Check Docker
docker --version

# Log in to Azure (opens browser)
az login

# Confirm you are logged in and see your subscription
az account show
```

### Set your subscription (if you have more than one)

```bash
# List subscriptions
az account list --output table

# Set the one you want to use
az account set --subscription "YOUR_SUBSCRIPTION_NAME_OR_ID"
```

---

## 2. Understand Your Options

Azure offers several ways to run a containerised application. Here is when to use each:

| Service | Best for | Monthly cost (estimate) | Complexity |
|---|---|---|---|
| **Azure Container Apps** ✅ Recommended | API / microservices, scale-to-zero, no VM management | $0–$20 (scale-to-zero when idle) | Low |
| **Azure App Service (Web App for Containers)** | Simple always-on apps, familiar PaaS | $13–$55 (B1–B2 tier) | Very low |
| **Azure Kubernetes Service (AKS)** | Multi-container production systems | $70+ per node | High |
| **Azure Container Instances (ACI)** | One-off jobs, batch workloads | Pay per second | Low |
| **Azure Virtual Machines** | Full OS control, legacy apps | $15+ | Very high |

**This guide uses Azure Container Apps** because:
- It supports Docker directly
- Scale-to-zero means you pay nothing when the API is idle (good for a capstone demo)
- Built-in HTTPS with a public URL
- No Kubernetes knowledge required

> **If you prefer App Service:** every step is identical up to Section 6.
> Section 6 includes a separate App Service path.

---

## 3. Prepare the Project

### 3.1 Confirm your Dockerfile is production-ready

Your existing `Dockerfile` is already multi-stage and uses a non-root user — that is production-grade. One addition needed for Azure: the container must listen on the port Azure tells it to via the `PORT` environment variable.

Open `Dockerfile` and check the final `CMD` line:

```dockerfile
# Current (hardcoded port 8000 — fine for local, but Azure injects $PORT)
CMD ["python", "-m", "uvicorn", "src.serving.serve:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

Azure Container Apps maps your container to port 8000 by default — you just need to
tell Azure which port your container exposes (you do this in Section 6, not in the Dockerfile).
No change needed to your Dockerfile.

### 3.2 Confirm your .env.example is complete

```bash
# Every variable your app reads must be documented here
cat .env.example
```

Azure does not use `.env` files — you will enter each variable as an **Application Setting** (Section 7).
Make sure `.env.example` lists everything including:
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME`
- `FAISS_INDEX_DIR`
- `EMBEDDING_MODEL`
- `RAG_MIN_SCORE`

### 3.3 Decide how to handle MLflow and the FAISS index

**Problem:** Your trained model lives in `mlruns/` and the FAISS index in `faiss_index/`.
Both are git-ignored (correct). On Azure the container starts fresh with no local files.

**Options (pick one):**

| Option | How | Best for |
|---|---|---|
| **A — Bake artefacts into the image** | `COPY mlruns/ mlruns/` in Dockerfile | Simplest; image is self-contained |
| **B — Azure Blob Storage mount** | Mount a blob container at `/app/mlruns` | Production; artefacts updated without rebuilding |
| **C — Run train + build_index as a startup script** | Entrypoint script runs training on first boot | Works but slow cold starts |

**For a capstone demo, Option A is easiest.**

Add these lines to your Dockerfile, just before the `USER appuser` line:

```dockerfile
# Option A: bake trained artefacts into the image
# Run 'python -m src.training.train --sample' and
# 'python -m src.rag.build_index --sample' LOCALLY first,
# then uncomment these two lines before building for Azure:
# COPY --chown=appuser:appuser mlruns/ mlruns/
# COPY --chown=appuser:appuser faiss_index/ faiss_index/
```

> **Before building the image:** run training and index locally so the artefact
> directories exist, then uncomment those two COPY lines.

```bash
# Generate the artefacts locally first
python -m src.training.train --sample
python -m src.rag.build_index --sample
```

---

## 4. Create Azure Resources

All commands below use the Azure CLI. Run them in your terminal once.

### 4.1 Define variables (set these once, reuse everywhere)

```bash
# ── Edit these values ─────────────────────────────────────────────────────────
RESOURCE_GROUP="customer-intelligence-rg"
LOCATION="eastus"               # or "centralindia", "westeurope", etc.
REGISTRY_NAME="customintelreg"  # must be globally unique, lowercase, 5-50 chars
APP_NAME="customer-intel-api"   # name of your Container App
ENVIRONMENT_NAME="customer-intel-env"
# ─────────────────────────────────────────────────────────────────────────────
```

> **Tip:** run `az account list-locations --output table` to find a location
> close to your users. `centralindia` is a good choice for IIT Gandhinagar.

### 4.2 Create a Resource Group

A Resource Group is a logical container for all Azure resources in this project.
Deleting the Resource Group deletes everything inside it — useful for cleanup.

```bash
az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION
```

Expected output: `"provisioningState": "Succeeded"`

### 4.3 Create an Azure Container Registry (ACR)

ACR stores your Docker image privately in Azure.

```bash
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $REGISTRY_NAME \
  --sku Basic \
  --admin-enabled true

# Get your registry login server URL (you'll need this later)
az acr show --name $REGISTRY_NAME --query loginServer --output tsv
# → customintelreg.azurecr.io
```

### 4.4 Create a Container Apps Environment

The Environment is the shared network/infrastructure that Container Apps run inside.

```bash
az containerapp env create \
  --name $ENVIRONMENT_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION
```

---

## 5. Build and Push the Container Image

### 5.1 Log in to your registry

```bash
az acr login --name $REGISTRY_NAME
```

### 5.2 Build the image

```bash
# From the project root (where Dockerfile lives)
cd "C:\Users\hetdp\OneDrive\Desktop\Customer Intelligence"

docker build \
  --tag $REGISTRY_NAME.azurecr.io/customer-intelligence:latest \
  --tag $REGISTRY_NAME.azurecr.io/customer-intelligence:v1.0 \
  .
```

> **First build takes 5–15 minutes** — it installs PyTorch, FAISS, sentence-transformers.
> Subsequent builds are faster due to Docker layer caching.

> **Common mistake:** forgetting to uncomment the `COPY mlruns/` lines in the
> Dockerfile before building. The container will start but `/health` will return 500
> because no trained model is found.

### 5.3 Test the image locally before pushing

```bash
docker run --rm -p 8000:8000 \
  --env-file .env \
  $REGISTRY_NAME.azurecr.io/customer-intelligence:latest
```

In a second terminal:
```bash
curl http://localhost:8000/health
# Should return: {"status":"ok","is_ready":true,...}
```

If it works locally it will work on Azure.

### 5.4 Push the image to ACR

```bash
docker push $REGISTRY_NAME.azurecr.io/customer-intelligence:latest
docker push $REGISTRY_NAME.azurecr.io/customer-intelligence:v1.0

# Confirm it arrived
az acr repository list --name $REGISTRY_NAME --output table
```

---

## 6. Deploy to Azure Container Apps

### Option A — Azure Container Apps (recommended)

```bash
# Get the ACR admin password (used to pull the image)
ACR_PASSWORD=$(az acr credential show \
  --name $REGISTRY_NAME \
  --query "passwords[0].value" \
  --output tsv)

# Deploy
az containerapp create \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $ENVIRONMENT_NAME \
  --image $REGISTRY_NAME.azurecr.io/customer-intelligence:latest \
  --registry-server $REGISTRY_NAME.azurecr.io \
  --registry-username $REGISTRY_NAME \
  --registry-password $ACR_PASSWORD \
  --target-port 8000 \
  --ingress external \
  --cpu 1.0 \
  --memory 2.0Gi \
  --min-replicas 0 \
  --max-replicas 3 \
  --env-vars \
    MLFLOW_TRACKING_URI=file:/app/mlruns \
    MLFLOW_EXPERIMENT_NAME=customer-intelligence \
    FAISS_INDEX_DIR=faiss_index \
    EMBEDDING_MODEL=all-MiniLM-L6-v2 \
    RAG_MIN_SCORE=0.35
```

> **`--min-replicas 0`** means the app scales to zero when idle — no traffic = no cost.
> Cold start takes ~10–15 seconds after a period of inactivity.

Get your live URL:

```bash
az containerapp show \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn \
  --output tsv
# → customer-intel-api.nicename-abc123.eastus.azurecontainerapps.io
```

Your API is now live at `https://<that-url>`.

---

### Option B — Azure App Service (Web App for Containers)

Use this if you prefer a simpler interface or need always-on behaviour.

```bash
# Create an App Service Plan (B1 = ~$13/month, always on)
az appservice plan create \
  --name customer-intel-plan \
  --resource-group $RESOURCE_GROUP \
  --is-linux \
  --sku B1

# Create the Web App pointing at your ACR image
az webapp create \
  --resource-group $RESOURCE_GROUP \
  --plan customer-intel-plan \
  --name $APP_NAME \
  --deployment-container-image-name \
    $REGISTRY_NAME.azurecr.io/customer-intelligence:latest

# Configure ACR credentials
az webapp config container set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --docker-custom-image-name $REGISTRY_NAME.azurecr.io/customer-intelligence:latest \
  --docker-registry-server-url https://$REGISTRY_NAME.azurecr.io \
  --docker-registry-server-user $REGISTRY_NAME \
  --docker-registry-server-password $ACR_PASSWORD

# Set the port
az webapp config appsettings set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings WEBSITES_PORT=8000
```

Your URL: `https://<APP_NAME>.azurewebsites.net`

---

## 7. Configure Environment Variables and Secrets

Sensitive values (API keys) must NEVER go in the Dockerfile or in a public repo.
Azure stores them encrypted as secrets.

### 7.1 Add the OpenAI API key as a secret (Container Apps)

```bash
# Add the secret
az containerapp secret set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --secrets openai-api-key=sk-YOUR_ACTUAL_KEY_HERE

# Reference the secret as an environment variable
az containerapp update \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --set-env-vars "OPENAI_API_KEY=secretref:openai-api-key"
```

### 7.2 App Service equivalent

```bash
az webapp config appsettings set \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings OPENAI_API_KEY="sk-YOUR_ACTUAL_KEY_HERE"
```

> App Service settings are encrypted at rest. They are equivalent to secrets for most purposes.

### 7.3 Verify variables are set

```bash
# Container Apps
az containerapp show \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.template.containers[0].env

# App Service
az webapp config appsettings list \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --output table
```

---

## 8. Set Up CI/CD with GitHub Actions

This adds automatic deployment: every push to `main` that passes CI will also
rebuild and redeploy the container. Your existing 4-job CI pipeline runs first;
deploy only happens if all jobs pass.

### 8.1 Create a service principal for GitHub to authenticate with Azure

```bash
# Get your subscription ID
SUBSCRIPTION_ID=$(az account show --query id --output tsv)

# Create the service principal with Contributor rights on your resource group
az ad sp create-for-rbac \
  --name "github-customer-intelligence" \
  --role Contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP \
  --sdk-auth
```

This outputs a JSON block. **Copy the entire block** — you need it in the next step.

```json
{
  "clientId": "...",
  "clientSecret": "...",
  "subscriptionId": "...",
  "tenantId": "...",
  ...
}
```

### 8.2 Add secrets to GitHub

Go to: **GitHub repo → Settings → Secrets and variables → Actions → New repository secret**

Add these secrets:

| Secret name | Value |
|---|---|
| `AZURE_CREDENTIALS` | The entire JSON block from step 8.1 |
| `REGISTRY_LOGIN_SERVER` | `customintelreg.azurecr.io` |
| `REGISTRY_USERNAME` | `customintelreg` |
| `REGISTRY_PASSWORD` | (from `az acr credential show --name $REGISTRY_NAME`) |
| `AZURE_RESOURCE_GROUP` | `customer-intelligence-rg` |
| `AZURE_APP_NAME` | `customer-intel-api` |
| `OPENAI_API_KEY` | Your actual key |

### 8.3 Add the deploy job to your CI workflow

Open `.github/workflows/ci.yml` and append this job after `eval-gate`:

```yaml
  # ── Job 5: Deploy to Azure (only on main, after all checks pass) ─────────────
  deploy:
    needs: [unit-tests, data-validation, eval-gate]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'

    steps:
      - uses: actions/checkout@v4

      # Generate artefacts needed inside the image
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: pip
      - run: pip install -r requirements.txt
      - name: Train model and build FAISS index
        env:
          DATA_DIR: ${{ github.workspace }}/data
          MLFLOW_TRACKING_URI: file:${{ github.workspace }}/mlruns
        run: |
          python -m src.training.train --sample
          python -m src.rag.build_index --sample

      # Log in to Azure
      - uses: azure/login@v2
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}

      # Log in to ACR
      - uses: azure/docker-login@v1
        with:
          login-server: ${{ secrets.REGISTRY_LOGIN_SERVER }}
          username: ${{ secrets.REGISTRY_USERNAME }}
          password: ${{ secrets.REGISTRY_PASSWORD }}

      # Build and push image
      - name: Build and push container image
        run: |
          IMAGE=${{ secrets.REGISTRY_LOGIN_SERVER }}/customer-intelligence
          docker build -t $IMAGE:${{ github.sha }} -t $IMAGE:latest .
          docker push $IMAGE:${{ github.sha }}
          docker push $IMAGE:latest

      # Update Container App to use the new image
      - name: Deploy to Azure Container Apps
        uses: azure/container-apps-deploy-action@v1
        with:
          resourceGroup: ${{ secrets.AZURE_RESOURCE_GROUP }}
          containerAppName: ${{ secrets.AZURE_APP_NAME }}
          imageToDeploy: >-
            ${{ secrets.REGISTRY_LOGIN_SERVER }}/customer-intelligence:${{ github.sha }}
```

> **Why train inside CI?** The image must contain the trained model. CI trains on the
> committed 500-row sample and bakes it into the image. For production you would pull
> artefacts from Azure Blob Storage instead.

---

## 9. Verify and Test the Deployment

### 9.1 Get the live URL

```bash
LIVE_URL=$(az containerapp show \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn \
  --output tsv)

echo "https://$LIVE_URL"
```

### 9.2 Run the same smoke tests against the live URL

```bash
# Health check
curl https://$LIVE_URL/health | python -m json.tool

# Prediction
curl -s -X POST https://$LIVE_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 42, "job": "admin.", "marital": "married",
    "education": "university.degree", "default": "no",
    "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "mon", "duration": 300,
    "campaign": 2, "pdays": 999, "previous": 0,
    "poutcome": "nonexistent", "emp.var.rate": -1.8,
    "cons.price.idx": 93.994, "cons.conf.idx": -36.4,
    "euribor3m": 4.857, "nr.employed": 5191.0
  }' | python -m json.tool

# Swagger UI
echo "Open in browser: https://$LIVE_URL/docs"
```

### 9.3 Check container logs if something is wrong

```bash
# Container Apps — stream live logs
az containerapp logs show \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --follow

# App Service
az webapp log tail \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP
```

---

## 10. Custom Domain and HTTPS

Azure Container Apps provides a free `*.azurecontainerapps.io` URL with HTTPS included.
If you want a custom domain (e.g. `api.yourdomain.com`):

```bash
# 1. Add the domain
az containerapp hostname add \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --hostname api.yourdomain.com

# 2. Azure shows a TXT record and CNAME to add in your DNS provider
# 3. After DNS propagates, bind a managed certificate (free)
az containerapp hostname bind \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --hostname api.yourdomain.com \
  --validation-method CNAME
```

HTTPS certificate is issued and renewed automatically — no Let's Encrypt setup needed.

---

## 11. Monitoring and Logs

### 11.1 Enable Application Insights (optional but recommended)

Application Insights gives you request traces, latency graphs, and error rates with zero code changes.

```bash
# Create the resource
az monitor app-insights component create \
  --app customer-intel-insights \
  --location $LOCATION \
  --resource-group $RESOURCE_GROUP \
  --kind web

# Get the instrumentation key
INSTRUMENTATION_KEY=$(az monitor app-insights component show \
  --app customer-intel-insights \
  --resource-group $RESOURCE_GROUP \
  --query instrumentationKey \
  --output tsv)

# Add it to the Container App
az containerapp update \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --set-env-vars "APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=$INSTRUMENTATION_KEY"
```

### 11.2 View metrics in the Azure Portal

Go to: **portal.azure.com → Your Container App → Metrics**

Useful metrics to watch:
- `Requests` — total requests per minute
- `Response time` — p50/p95 latency
- `Replicas` — how many instances are running
- `CPU/Memory usage` — spot runaway memory from model loading

### 11.3 Set up an alert for errors

```bash
az monitor metrics alert create \
  --name "high-error-rate" \
  --resource-group $RESOURCE_GROUP \
  --scopes $(az containerapp show --name $APP_NAME \
    --resource-group $RESOURCE_GROUP --query id --output tsv) \
  --condition "avg Requests > 10 where ResultCode includes 5" \
  --window-size 5m \
  --evaluation-frequency 1m \
  --action email your@email.com
```

---

## 12. Cost Management

### Estimated monthly cost for this project

| Resource | Tier | Est. cost/month |
|---|---|---|
| Container Apps (scale-to-zero) | Consumption | $0–$5 (idle), $10–$30 (active demo) |
| Azure Container Registry | Basic | $5 |
| Application Insights | Free tier (5 GB) | $0 |
| **Total** | | **~$5–$35** |

### Cost-saving tips

**1. Scale to zero when not in use**
```bash
# Already set with --min-replicas 0 — nothing to do
# The app automatically stops when there is no traffic
```

**2. Set spending alerts**
```bash
az consumption budget create \
  --resource-group $RESOURCE_GROUP \
  --budget-name "customer-intel-budget" \
  --amount 30 \
  --time-grain Monthly \
  --start-date $(date +%Y-%m-01) \
  --end-date 2027-01-01 \
  --notifications '[{
    "enabled": true,
    "operator": "GreaterThan",
    "threshold": 80,
    "contactEmails": ["your@email.com"]
  }]'
```

**3. Delete everything when done with the demo**
```bash
# This deletes ALL resources in the group — nothing will keep running or charging
az group delete --name $RESOURCE_GROUP --yes --no-wait
echo "All resources queued for deletion."
```

> ⚠️ Do NOT delete the resource group until after your submission/demo is graded.

---

## 13. Troubleshooting

### `/health` returns 500

**Cause:** The model was not found — `mlruns/` was not baked into the image.

**Fix:**
```bash
# 1. Train locally
python -m src.training.train --sample

# 2. Uncomment COPY lines in Dockerfile
# COPY --chown=appuser:appuser mlruns/ mlruns/
# COPY --chown=appuser:appuser faiss_index/ faiss_index/

# 3. Rebuild and push
docker build -t $REGISTRY_NAME.azurecr.io/customer-intelligence:latest .
docker push $REGISTRY_NAME.azurecr.io/customer-intelligence:latest

# 4. Force a new revision
az containerapp update --name $APP_NAME --resource-group $RESOURCE_GROUP \
  --image $REGISTRY_NAME.azurecr.io/customer-intelligence:latest
```

---

### Container exits immediately (exit code 1)

**Cause:** Crash during startup — usually a missing environment variable.

**Fix:**
```bash
# Stream logs to see the exact error
az containerapp logs show \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --tail 50
```

---

### `az containerapp create` fails with "registry unauthorized"

**Cause:** Wrong ACR username or password.

**Fix:**
```bash
# Re-fetch the password (it rotates)
az acr credential show --name $REGISTRY_NAME
# Use password[0] or password[1]
```

---

### RAG retriever fails (DLL error on Linux)

The Windows DLL error does NOT occur on Linux (Azure runs Linux containers). 
The `sentence_transformers` import-before-`faiss` ordering in `serve.py` is 
harmless on Linux — the RAG endpoint will work correctly on Azure even if it 
fails locally on Windows.

---

### First request is very slow (~15–30 seconds)

**Cause:** Container scaled from zero — cold start. Normal behaviour.

**Fix:** Set `--min-replicas 1` to keep one instance warm (adds ~$10/month):
```bash
az containerapp update \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --min-replicas 1
```

---

## Quick-reference: all variables in one place

```bash
RESOURCE_GROUP="customer-intelligence-rg"
LOCATION="eastus"
REGISTRY_NAME="customintelreg"
APP_NAME="customer-intel-api"
ENVIRONMENT_NAME="customer-intel-env"
LIVE_URL=$(az containerapp show --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn --output tsv)
```

---

*Guide written for Azure CLI 2.50+ · Container Apps 1.0 · GitHub Actions · May 2026*
