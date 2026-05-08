"""
WJ Analytics SaaS API
=====================
Single-file FastAPI backend exposing the WJ pairing-family methodology
with API key authentication, usage metering, and Stripe-driven
subscription tiers.

Run locally:
    uvicorn main:app --reload --port 8000

Deploy on Render.com:
    See deployment_steps.md
"""
import os
import sqlite3
import secrets
import hashlib
import time
from typing import Optional
from datetime import datetime

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------
# Configuration (override via environment variables)
# ----------------------------------------------------------------------------
DB_PATH = os.environ.get("WJ_DB_PATH", "wj_saas.db")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_API_KEY = os.environ.get("STRIPE_SECRET_KEY") or os.environ.get("STRIPE_API_KEY", "")
ADMIN_TOKEN = os.environ.get("WJ_ADMIN_TOKEN", "change-me-in-production")

try:
    import stripe as _stripe_module
    if STRIPE_API_KEY:
        _stripe_module.api_key = STRIPE_API_KEY
except ImportError:
    pass

TIER_LIMITS = {
    "free": 100,
    "solo": 5_000,
    "team": 50_000,
    "enterprise": float("inf"),
}

# ----------------------------------------------------------------------------
# Database helpers
# ----------------------------------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            tier TEXT NOT NULL DEFAULT 'free',
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            api_key_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            current_month_calls INTEGER DEFAULT 0,
            month_reset_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS api_calls (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            endpoint TEXT NOT NULL,
            tier TEXT NOT NULL,
            ts TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()


def hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


def lookup_user_by_api_key(api_key: str):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "SELECT * FROM users WHERE api_key_hash = ?",
        (hash_key(api_key),),
    )
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def increment_usage(user_id: int, endpoint: str, tier: str):
    now = datetime.utcnow().isoformat()
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO api_calls (user_id, endpoint, tier, ts) VALUES (?, ?, ?, ?)",
        (user_id, endpoint, tier, now),
    )
    c.execute(
        "UPDATE users SET current_month_calls = current_month_calls + 1 WHERE id = ?",
        (user_id,),
    )
    conn.commit()
    conn.close()


def reset_monthly_usage_if_needed(user: dict):
    """Reset current_month_calls at the start of each calendar month."""
    last_reset = datetime.fromisoformat(user["month_reset_at"])
    now = datetime.utcnow()
    if now.year != last_reset.year or now.month != last_reset.month:
        conn = get_db()
        c = conn.cursor()
        c.execute(
            "UPDATE users SET current_month_calls = 0, month_reset_at = ? WHERE id = ?",
            (now.isoformat(), user["id"]),
        )
        conn.commit()
        conn.close()


# ----------------------------------------------------------------------------
# WJ methodology functions (canonical implementations)
# ----------------------------------------------------------------------------
def weighted_jaccard(corr_a: np.ndarray, corr_b: np.ndarray) -> float:
    idx = np.triu_indices(corr_a.shape[0], k=1)
    a = np.abs(corr_a[idx])
    b = np.abs(corr_b[idx])
    num = np.minimum(a, b).sum()
    den = np.maximum(a, b).sum()
    return float(num / den) if den > 0 else 1.0


def signed_weighted_jaccard(corr_a: np.ndarray, corr_b: np.ndarray) -> float:
    idx = np.triu_indices(corr_a.shape[0], k=1)
    a = corr_a[idx] + 1.0
    b = corr_b[idx] + 1.0
    num = np.minimum(a, b).sum()
    den = np.maximum(a, b).sum()
    return float(num / den) if den > 0 else 1.0


def binary_jaccard(corr_a: np.ndarray, corr_b: np.ndarray, threshold: float) -> float:
    idx = np.triu_indices(corr_a.shape[0], k=1)
    a = np.abs(corr_a[idx]) >= threshold
    b = np.abs(corr_b[idx]) >= threshold
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union > 0 else 1.0


def directional_classification(corr_a: np.ndarray, corr_b: np.ndarray) -> str:
    """Convergence if mean |corr| in B exceeds A; divergence otherwise."""
    mean_a = np.abs(corr_a).mean()
    mean_b = np.abs(corr_b).mean()
    if mean_b > mean_a + 0.02:
        return "CONVERGENCE"
    if mean_b < mean_a - 0.02:
        return "DIVERGENCE"
    return "STABLE"


# ----------------------------------------------------------------------------
# Pydantic schemas
# ----------------------------------------------------------------------------
class RegimeRequest(BaseModel):
    data: list = Field(..., description="Correlation matrix as 2D list")
    baseline: Optional[list] = Field(None, description="Baseline correlation matrix")
    binary_threshold: float = Field(0.3, description="Threshold for binary Jaccard")


class RegimeResponse(BaseModel):
    wj_unsigned: float
    wj_signed: float
    binary_jaccard: float
    type1_gap: float
    type2_gap: float
    sign_inversion_pct: float
    magnitude_change_pct: float
    regime_classification: str
    n_pairs: int
    notes: str


# ----------------------------------------------------------------------------
# Auth dependency
# ----------------------------------------------------------------------------
def require_api_key(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing or malformed Authorization header")
    api_key = authorization.split(" ", 1)[1].strip()
    user = lookup_user_by_api_key(api_key)
    if user is None:
        raise HTTPException(403, "Invalid API key")
    reset_monthly_usage_if_needed(user)
    user = lookup_user_by_api_key(api_key)
    limit = TIER_LIMITS.get(user["tier"], 100)
    if user["current_month_calls"] >= limit:
        raise HTTPException(
            429,
            f"Monthly quota exceeded ({limit} calls on {user['tier']} tier). "
            f"Upgrade at https://wjanalytics.com/pricing",
        )
    return user


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
app = FastAPI(
    title="WJ Analytics API",
    description="Pairing-family decomposition for correlation network analysis",
    version="1.0.0",
)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
def root():
    return {
        "service": "WJ Analytics API",
        "version": "1.0.0",
        "docs": "/docs",
        "operator": "Inner Architecture LLC",
    }


@app.post("/v1/regime", response_model=RegimeResponse)
def detect_regime(req: RegimeRequest, user: dict = Depends(require_api_key)):
    """Compute WJ pairing-family decomposition between data and baseline.

    Returns unsigned WJ, signed WJ, binary Jaccard at threshold, Type 1 gap,
    Type 2 gap, sign-inversion percentage, and direction classification.
    """
    try:
        corr_data = np.array(req.data, dtype=np.float64)
        if corr_data.ndim != 2 or corr_data.shape[0] != corr_data.shape[1]:
            raise ValueError("data must be a square correlation matrix")

        if req.baseline is None:
            corr_baseline = np.eye(corr_data.shape[0])
            baseline_note = "No baseline provided; using identity matrix as reference."
        else:
            corr_baseline = np.array(req.baseline, dtype=np.float64)
            baseline_note = "Custom baseline used."

        if corr_baseline.shape != corr_data.shape:
            raise ValueError("baseline shape must match data shape")
    except Exception as e:
        raise HTTPException(400, f"Input error: {e}")

    # Free tier: only unsigned WJ + binary Jaccard
    free_tier = user["tier"] == "free"

    wj_uns = weighted_jaccard(corr_baseline, corr_data)
    bj = binary_jaccard(corr_baseline, corr_data, req.binary_threshold)

    if free_tier:
        wj_sgn = float("nan")
        type1_gap = float("nan")
        type2_gap = float("nan")
        sign_inv_pct = float("nan")
        mag_pct = float("nan")
        notes = (
            "Free tier: unsigned WJ + binary Jaccard returned. "
            "Upgrade to Solo for full pairing-family decomposition."
        )
    else:
        wj_sgn = signed_weighted_jaccard(corr_baseline, corr_data)
        type1_gap = wj_uns - bj
        type2_gap = wj_sgn - wj_uns
        reorg_unsigned = 1 - wj_uns
        if reorg_unsigned > 0:
            sign_inv_pct = (type2_gap / reorg_unsigned) * 100
            mag_pct = 100 - sign_inv_pct
        else:
            sign_inv_pct = 0.0
            mag_pct = 100.0
        notes = baseline_note

    direction = directional_classification(corr_baseline, corr_data)
    n_pairs = corr_data.shape[0] * (corr_data.shape[0] - 1) // 2

    increment_usage(user["id"], "/v1/regime", user["tier"])

    return RegimeResponse(
        wj_unsigned=wj_uns,
        wj_signed=wj_sgn,
        binary_jaccard=bj,
        type1_gap=type1_gap,
        type2_gap=type2_gap,
        sign_inversion_pct=sign_inv_pct,
        magnitude_change_pct=mag_pct,
        regime_classification=direction,
        n_pairs=n_pairs,
        notes=notes,
    )


@app.get("/v1/usage", dependencies=[])
def get_usage(user: dict = Depends(require_api_key)):
    limit = TIER_LIMITS.get(user["tier"], 100)
    return {
        "tier": user["tier"],
        "calls_this_month": user["current_month_calls"],
        "monthly_limit": limit if limit != float("inf") else None,
        "remaining": (
            limit - user["current_month_calls"]
            if limit != float("inf") else "unlimited"
        ),
    }


# ----------------------------------------------------------------------------
# Stripe webhook
# ----------------------------------------------------------------------------
@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """Receive Stripe events: subscription created/updated/canceled.

    Defensive design: any failure inside event handling logs the error and
    returns 200 to Stripe. Returning 500 makes Stripe retry indefinitely
    and pollutes the delivery log. The application's own error log is the
    source of truth for diagnostics.
    """
    try:
        import stripe
    except ImportError:
        print("[WEBHOOK] stripe package not installed")
        return JSONResponse({"received": False, "error": "stripe sdk missing"}, status_code=200)

    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")

    if STRIPE_WEBHOOK_SECRET:
        try:
            event = stripe.Webhook.construct_event(
                payload, sig, STRIPE_WEBHOOK_SECRET
            )
        except Exception as e:
            print(f"[WEBHOOK] signature verification failed: {e}")
            raise HTTPException(400, f"Webhook signature verification failed: {e}")
    else:
        # No secret configured — accept the raw event (development only)
        import json as _json
        try:
            event = _json.loads(payload)
        except Exception as e:
            print(f"[WEBHOOK] payload parse failed: {e}")
            return JSONResponse({"received": False, "error": "bad payload"}, status_code=200)

    obj = event.get("data", {}).get("object", {})
    typ = event.get("type", "")
    print(f"[WEBHOOK] received {typ} | event_id={event.get('id')}")

    try:
        if typ == "checkout.session.completed":
            email = (
                obj.get("customer_email")
                or (obj.get("customer_details") or {}).get("email")
                or "unknown@example.com"
            )
            customer_id = obj.get("customer")
            subscription_id = obj.get("subscription")

            # Extract tier from price metadata if API key is configured.
            # Falls back to "solo" if API call fails or no key is set.
            tier = "solo"
            if STRIPE_API_KEY and obj.get("id"):
                try:
                    line_items = stripe.checkout.Session.list_line_items(obj["id"])
                    for li in (line_items.get("data") or []):
                        price_id = (li.get("price") or {}).get("id")
                        if price_id:
                            price = stripe.Price.retrieve(price_id)
                            tier = (price.get("metadata") or {}).get("tier", "solo")
                            break
                except Exception as e:
                    print(f"[WEBHOOK] line items lookup failed: {e}; defaulting tier=solo")

            api_key = "wjk_" + secrets.token_urlsafe(32)
            api_key_hash = hash_key(api_key)
            now = datetime.utcnow().isoformat()
            conn = get_db()
            c = conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO users "
                "(email, tier, stripe_customer_id, stripe_subscription_id, "
                "api_key_hash, created_at, current_month_calls, month_reset_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                (email, tier, customer_id, subscription_id, api_key_hash, now, now),
            )
            conn.commit()
            conn.close()
            print(f"[NEW USER] {email} | tier={tier} | api_key={api_key}")

        elif typ in ("customer.subscription.updated", "customer.subscription.deleted"):
            customer_id = obj.get("customer")
            new_tier = "free" if typ == "customer.subscription.deleted" else "solo"
            conn = get_db()
            c = conn.cursor()
            c.execute(
                "UPDATE users SET tier = ? WHERE stripe_customer_id = ?",
                (new_tier, customer_id),
            )
            conn.commit()
            conn.close()
            print(f"[SUBSCRIPTION CHANGE] customer={customer_id} | tier={new_tier}")

    except Exception as e:
        print(f"[WEBHOOK] handler error for {typ}: {e}")
        return JSONResponse({"received": True, "warning": str(e)}, status_code=200)

    return JSONResponse({"received": True})


# ----------------------------------------------------------------------------
# Admin endpoints (for Drake to manage users manually if needed)
# ----------------------------------------------------------------------------
@app.post("/admin/create-user")
def admin_create_user(
    email: str,
    tier: str = "free",
    admin_token: str = Header(None, alias="X-Admin-Token"),
):
    """Manual user creation (useful for Enterprise customers, comp accounts)."""
    if admin_token != ADMIN_TOKEN:
        raise HTTPException(401, "Invalid admin token")
    if tier not in TIER_LIMITS:
        raise HTTPException(400, f"Invalid tier: {tier}")
    api_key = "wjk_" + secrets.token_urlsafe(32)
    api_key_hash = hash_key(api_key)
    now = datetime.utcnow().isoformat()
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users "
            "(email, tier, api_key_hash, created_at, "
            "current_month_calls, month_reset_at) "
            "VALUES (?, ?, ?, ?, 0, ?)",
            (email, tier, api_key_hash, now, now),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(409, f"User already exists: {email}")
    finally:
        conn.close()
    return {"email": email, "tier": tier, "api_key": api_key}


@app.get("/admin/stats")
def admin_stats(admin_token: str = Header(None, alias="X-Admin-Token")):
    if admin_token != ADMIN_TOKEN:
        raise HTTPException(401, "Invalid admin token")
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT tier, COUNT(*) FROM users GROUP BY tier")
    by_tier = dict(c.fetchall())
    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]
    c.execute(
        "SELECT COUNT(*) FROM api_calls "
        "WHERE ts > datetime('now', '-1 day')"
    )
    calls_24h = c.fetchone()[0]
    conn.close()
    return {
        "total_users": total_users,
        "users_by_tier": by_tier,
        "api_calls_last_24h": calls_24h,
    }
