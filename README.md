# WJ Analytics API

FastAPI backend for the Weighted Jaccard pairing-family decomposition methodology.

**Live methodology reference:** [innerarchitecturellc.com](https://innerarchitecturellc.com)
**Author:** Drake H. Harbert | Inner Architecture LLC | ORCID 0009-0007-7740-3616

## Endpoints

- `POST /v1/regime` — submit a correlation matrix, get unsigned/signed WJ + binary Jaccard + pairing-family gaps
- `GET /v1/usage` — your account usage this billing period
- `POST /webhooks/stripe` — Stripe subscription lifecycle webhook
- Admin endpoints for API key issuance

## Deploy on Render

1. Click "New Web Service"
2. Connect this repository
3. Render auto-detects `render.yaml` and configures the service
4. Set the secrets `STRIPE_SECRET_KEY` and `STRIPE_WEBHOOK_SECRET` in Render dashboard
5. Deploy

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Free vs Paid tiers

Free tier returns: `wj_unsigned`, `binary_jaccard`
Paid tier returns: full pairing-family decomposition (`wj_signed`, `type1_gap`, `type2_gap`, `sign_inversion_pct`, `regime_classification`)

## License

Proprietary. All rights reserved by Inner Architecture LLC.
