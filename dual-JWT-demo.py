import os
import time
import uuid
import jwt
import redis as redis_lib
from typing import Protocol
from fastapi import FastAPI, HTTPException, Request, Response, Depends, Body


# ============================================================
# Configuration and constants: base scheme
# ============================================================

JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "RS256")
JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "api.example.com")
JWT_ISSUER = os.environ.get("JWT_ISSUER", "https://auth.example.com")
JWT_LEEWAY = int(os.environ.get("JWT_LEEWAY", "30"))

JWT_PRIVATE_KEY = os.environ["JWT_PRIVATE_KEY"]
JWT_PUBLIC_KEY = os.environ["JWT_PUBLIC_KEY"]

ACCESS_TTL = 15 * 60
REFRESH_SLIDING_TTL = 30 * 24 * 3600
SESSION_ABSOLUTE_TTL = 90 * 24 * 3600

CLAIM_SUB = "sub"
CLAIM_JTI = "jti"
CLAIM_SID = "sid"
CLAIM_IAT = "iat"
CLAIM_EXP = "exp"
CLAIM_TYP = "typ"
CLAIM_AUD = "aud"
CLAIM_ISS = "iss"
CLAIM_SESSION_EXPIRY = "session_expiry"

TYP_ACCESS = "access"
TYP_REFRESH = "refresh"

REFRESH_COOKIE_NAME = "refresh"
REFRESH_COOKIE_PATH = "/auth/refresh"


# ============================================================
# Configuration and constants: Redis Revocation (Extension)
# ============================================================

ENABLE_REDIS_REVOCATION_EXTENSION = (
    os.environ.get("ENABLE_REDIS_REVOCATION_EXTENSION", "true").lower() == "true"
)

# TTL for RedisRevocationStore keys.
#
# revocation_cutoff is needed while at least one refresh token with
# iat < cutoff is still alive.
# Such a refresh has exp = iat + REFRESH_SLIDING_TTL
#                            <= cutoff + REDIS_REVOCATION_STORE_TTL - 1.
# Hence revocation_ttl = REDIS_REVOCATION_STORE_TTL + 1 (rounding headroom).
#
# NEVER decrease it: otherwise the key gets deleted before the old refresh
# tokens expire, and they would become valid again.
# When REFRESH_SLIDING_TTL is increased — increase REDIS_REVOCATION_STORE_TTL.
# Declared as a separate number, not a copy of REFRESH_SLIDING_TTL, so that an
# accidental decrease of REFRESH_SLIDING_TTL does not lower
# REDIS_REVOCATION_STORE_TTL.
REDIS_REVOCATION_STORE_TTL = 30 * 24 * 3600

USER_REVOCATION_CUTOFF_KEY = "user:{user_id}:revocation_cutoff"


# ============================================================
# Redis client
# ============================================================

redis_client = redis_lib.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", "6379")),
    db=int(os.environ.get("REDIS_DB", "0")),
    decode_responses=True,
)


# ============================================================
# RevocationStore
# ============================================================

class RevocationStore(Protocol):
    """Session revocation store.

    Keeps the revocation_cutoff boundary: all refresh tokens
    with iat < cutoff are considered revoked.
    """

    def revoke_all(self, user_id: str, now: int) -> None:
        ...

    def is_revoked(self, user_id: str, iat: int) -> bool:
        ...


class NopRevocationStore:
    """No-op implementation: revocation is disabled.

    is_revoked — always False (no refresh token is revoked).
    revoke_all — raises RuntimeError so that it does not create
    a false impression that sessions have been revoked.

    Used when ENABLE_REDIS_REVOCATION_EXTENSION=false
    """

    def revoke_all(self, user_id: str, now: int) -> None:
        raise RuntimeError(
            "revoke_all is disabled. "
            "Set ENABLE_REDIS_REVOCATION_EXTENSION=true to enable revocation."
        )

    def is_revoked(self, user_id: str, iat: int) -> bool:
        return False


class RedisRevocationStore:
    def __init__(self, client, revocation_ttl_base: int):
        self._client = client
        self._revocation_ttl_base = revocation_ttl_base

    def revoke_all(self, user_id: str, now: int) -> None:
        # revocation_cutoff — the cutoff boundary.
        # All refresh tokens with iat < cutoff are considered revoked.
        #
        # +1: revoke everything up to the end of the current second
        # (inclusive).
        # iat in a JWT is an integer number of seconds (UTC).
        # If cutoff = now, a refresh issued in that same second has
        # iat == now. The check iat < cutoff gives false, and such a refresh
        # is NOT rejected — that is a hole.
        # +1 shifts the boundary: iat == now < now + 1 → true → rejected.
        cutoff = now + 1

        # revocation_ttl — TTL of the cutoff key.
        # +1 — rounding headroom (see the comment on REDIS_REVOCATION_STORE_TTL).
        revocation_ttl = self._revocation_ttl_base + 1

        self._client.set(
            USER_REVOCATION_CUTOFF_KEY.format(user_id=user_id),
            cutoff,
            ex=revocation_ttl,
        )

    def is_revoked(self, user_id: str, iat: int) -> bool:
        cutoff = self._client.get(
            USER_REVOCATION_CUTOFF_KEY.format(user_id=user_id)
        )
        return bool(cutoff) and iat < int(cutoff)


def build_revocation_store() -> RevocationStore:
    if ENABLE_REDIS_REVOCATION_EXTENSION:
        return RedisRevocationStore(redis_client, REDIS_REVOCATION_STORE_TTL)
    return NopRevocationStore()


revocation_store = build_revocation_store()


# ============================================================
# Time
# ============================================================

def utc_now() -> int:
    """Current UTC time in seconds (int)."""
    return int(time.time())


# ============================================================
# Cookie helpers
# ============================================================

# Refresh-cookie attributes. One dict for both setting and deleting:
# the browser matches a cookie on overwrite by name + path + domain, so the
# cookie used for deletion must carry the same path as the one that was set.
# Secure/HttpOnly/SameSite match as well: a secure cookie can be overwritten
# only by a secure instruction, otherwise some browsers ignore the removal and
# the httpOnly refresh cookie would survive logout.
REFRESH_COOKIE_ATTRS = {
    "httponly": True,
    "secure": True,
    "samesite": "strict",
    "path": REFRESH_COOKIE_PATH,
}


def set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=token,
        max_age=REFRESH_SLIDING_TTL,
        **REFRESH_COOKIE_ATTRS,
    )


def delete_refresh_cookie(response: Response) -> None:
    # Deleting a cookie = setting the same cookie with an empty value and an
    # expired lifetime (Max-Age=0 + Expires in the past).
    #
    # response.delete_cookie() does not reuse the attributes from
    # set_refresh_cookie, and old Starlette versions do not accept
    # secure/httponly at all — removing a secure cookie would not work there.
    # So the expired cookie is set manually via set_cookie with the same
    # attributes (REFRESH_COOKIE_ATTRS) — a guaranteed match.
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value="",
        max_age=0,
        expires=0,
        **REFRESH_COOKIE_ATTRS,
    )


# ============================================================
# Token issuing
# ============================================================

def issue_access(user_id: str, now: int) -> str:
    payload = {
        CLAIM_ISS: JWT_ISSUER,
        CLAIM_AUD: JWT_AUDIENCE,
        CLAIM_SUB: user_id,
        CLAIM_JTI: str(uuid.uuid4()),        # audit only
        CLAIM_IAT: now,
        CLAIM_EXP: now + ACCESS_TTL,
        CLAIM_TYP: TYP_ACCESS,
    }
    return jwt.encode(payload, JWT_PRIVATE_KEY, algorithm=JWT_ALGORITHM)


def issue_refresh(
    user_id: str,
    sid: str,
    now: int,
    session_expiry: int,
) -> str:
    payload = {
        CLAIM_ISS: JWT_ISSUER,
        CLAIM_AUD: JWT_AUDIENCE,
        CLAIM_SUB: user_id,
        CLAIM_JTI: str(uuid.uuid4()),        # audit only
        CLAIM_SID: sid,
        CLAIM_IAT: now,
        CLAIM_EXP: now + REFRESH_SLIDING_TTL,
        CLAIM_SESSION_EXPIRY: session_expiry,
        CLAIM_TYP: TYP_REFRESH,
    }
    return jwt.encode(payload, JWT_PRIVATE_KEY, algorithm=JWT_ALGORITHM)


# ============================================================
# Token decoding
# ============================================================

def decode_access(token: str) -> dict:
    payload = jwt.decode(
        token,
        JWT_PUBLIC_KEY,
        algorithms=[JWT_ALGORITHM],
        audience=JWT_AUDIENCE,
        issuer=JWT_ISSUER,
        leeway=JWT_LEEWAY,
        options={
            "require": [
                CLAIM_EXP, CLAIM_IAT, CLAIM_SUB, CLAIM_JTI, CLAIM_TYP,
            ]
        },
    )
    if payload.get(CLAIM_TYP) != TYP_ACCESS:
        raise jwt.InvalidTokenError("not an access token")
    return payload


def decode_refresh(token: str) -> dict:
    payload = jwt.decode(
        token,
        JWT_PUBLIC_KEY,
        algorithms=[JWT_ALGORITHM],
        audience=JWT_AUDIENCE,
        issuer=JWT_ISSUER,
        leeway=JWT_LEEWAY,
        options={
            "require": [
                CLAIM_EXP, CLAIM_IAT, CLAIM_SUB,
                CLAIM_JTI, CLAIM_SID, CLAIM_SESSION_EXPIRY, CLAIM_TYP,
            ]
        },
    )
    if payload.get(CLAIM_TYP) != TYP_REFRESH:
        raise jwt.InvalidTokenError("not a refresh token")
    return payload


# ============================================================
# Login
# ============================================================

def issue_tokens_for_login(user_id: str, response: Response) -> dict:
    """Issues access + refresh for a standard login.

    - access → return value
    - refresh → set httpOnly cookie

    Called after the user has been authenticated successfully.
    """
    now = utc_now()
    sid = str(uuid.uuid4())
    session_expiry = now + SESSION_ABSOLUTE_TTL

    access = issue_access(user_id, now)
    refresh = issue_refresh(user_id, sid, now, session_expiry)

    set_refresh_cookie(response, refresh)

    return {"access": access, "expires_in": ACCESS_TTL}


# ============================================================
# Refresh
# ============================================================

def refresh(request: Request, response: Response) -> dict:
    token = request.cookies.get(REFRESH_COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="No refresh token")

    try:
        payload = decode_refresh(token)
    except jwt.ExpiredSignatureError:
        delete_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        delete_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = payload[CLAIM_SUB]
    sid = payload[CLAIM_SID]
    iat = payload[CLAIM_IAT]
    session_expiry = payload[CLAIM_SESSION_EXPIRY]

    now = utc_now()

    # `<=`, not `<`: session_expiry is the absolute session limit, and the
    # moment of equality means "session expired". With `<`, a refresh used in
    # the very second the session has already expired would stay valid.
    if session_expiry <= now:
        delete_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="Session max lifetime reached")

    if revocation_store.is_revoked(user_id, iat):
        delete_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="Session revoked")

    new_access = issue_access(user_id, now)

    new_refresh = issue_refresh(user_id, sid, now, session_expiry)
    set_refresh_cookie(response, new_refresh)

    return {"access": new_access, "expires_in": ACCESS_TTL}


# ============================================================
# Logout
# ============================================================

def logout(response: Response) -> dict:
    delete_refresh_cookie(response)
    return {"ok": True}


# ============================================================
# Logout all
# ============================================================

def logout_all(user_id: str, response: Response) -> dict:
    now = utc_now()
    revocation_store.revoke_all(user_id, now)
    delete_refresh_cookie(response)
    return {"ok": True}


# ============================================================
# Access verification
# ============================================================

def verify_access(token: str) -> dict:
    try:
        payload = decode_access(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Access token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid access token")

    # Access is stateless — Redis is not touched.
    # Logout all → revocation_cutoff applies to refresh tokens.
    # Access expires after ACCESS_TTL.
    return payload


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(title="Dual JWT Demo")


# --- Dependencies ---

def extract_access_token(request: Request) -> str | None:
    """Extract Bearer token from the Authorization header."""
    auth = request.headers.get("Authorization")
    if auth and auth.startswith("Bearer "):
        return auth[7:]
    return None


def get_current_user(request: Request) -> dict:
    """Dependency: verify access token and return payload (sub, jti, etc.)."""
    token = extract_access_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    return verify_access(token)


# --- Helpers (in-memory user store for demo only) ---

USERS_DB = {"user@example.com": {"password": "secret", "user_id": "u-42"}}


# --- Endpoints ---

@app.post("/auth/login")
def login(
    response: Response,
    email: str = Body(..., examples=["user@example.com"]),
    password: str = Body(..., examples=["secret"]),
):
    user = USERS_DB.get(email)
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    tokens = issue_tokens_for_login(user["user_id"], response)
    return tokens


@app.get("/echo")
def echo(user: dict = Depends(get_current_user)):
    """Secured endpoint — returns user identity from the access token."""
    return {"user_id": user[CLAIM_SUB], "jti": user[CLAIM_JTI]}


@app.post("/auth/refresh")
def auth_refresh(
    request: Request,
    response: Response,
):
    return refresh(request, response)


@app.post("/auth/logout")
def auth_logout(response: Response):
    return logout(response)


@app.post("/auth/logout-all")
def auth_logout_all(user: dict = Depends(get_current_user), response: Response):
    """Revoke all sessions for the current user."""
    return logout_all(user[CLAIM_SUB], response)
