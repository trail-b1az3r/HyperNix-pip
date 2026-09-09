"""t1api.schemas — Pydantic v2 request/response models for the T1 API.

Requires ``fastapi``/``pydantic`` (the ``hypernix[t1api]`` extra). Import
this module lazily from route handlers, not from ``t1api.registry`` /
``t1api.usage`` / ``t1api.storage``, which must stay importable without
these dependencies.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail
    request_id: str


# ---------------------------------------------------------------------------
# Health / status
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    request_id: str


class StatusResponse(BaseModel):
    status: str = "ok"
    environment: str
    #: This deployment's own name. Discovery (``waiter -F <name>``) has
    #: nothing to match on without it, which made name lookup silently
    #: never succeed.
    server_name: str = ""
    host_id: str = ""
    server_id: str = ""
    #: The short spelling, ``1.0.26.8.0.1``. Kept as the field clients
    #: already read; the long spelling and the parsed components arrive
    #: alongside it rather than replacing it.
    t1_api_version: str
    t1_api_version_long: str = ""
    t1_version: dict[str, Any] = Field(default_factory=dict)
    hypernix_version: str
    #: Historically the beta number ("beta4"). Now the release line —
    #: "t1-1.0" — because the betas ended at T1 v1.0.26.8.0.1. Still
    #: present, and still a string, so a Beta 3 client does not break.
    beta: str
    lmstudio_bridge_enabled: bool = False
    lmstudio_configured: bool = False
    hyperlink_enabled: bool = False
    model_count: int
    storage_backend: str
    request_id: str
    # Beta 3: enough for an operator (or `waiter status`) to see which
    # protections are actually on, without exposing a single secret's
    # value — `secrets_configured` reports set/unset booleans only.
    tls_enabled: bool = False
    mtls_mode: str = "off"
    rate_limit_enabled: bool = True
    audit_enabled: bool = True
    network_policy_enabled: bool = True
    allow_unlisted_clients: bool = True
    remote_deployment_enabled: bool = False
    secrets_configured: dict[str, bool] = Field(default_factory=dict)
    production_ready: bool = True
    production_warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class ValidateKeyRequest(BaseModel):
    key: str = Field(..., min_length=4, description="Raw T1_ key string.")


class ValidateKeyResponse(BaseModel):
    key_id: str
    key_type: str
    scopes: list[str]
    active: bool
    is_admin: bool
    expires_at: float | None
    request_id: str


class TokenRequest(BaseModel):
    key: str = Field(..., min_length=4, description="Raw T1_ key string to exchange.")
    ttl_seconds: int | None = Field(default=None, ge=1, le=86400)
    scopes: list[str] | None = Field(
        default=None, description="Subset of the key's scopes to grant. Omit for all scopes."
    )


class TokenResponse(BaseModel):
    token: str
    key_id: str
    scopes: list[str]
    expires_at: float
    request_id: str


class RotateResponse(BaseModel):
    key_id: str
    key: str
    key_type: str
    scopes: list[str]
    rotated_from: str | None
    request_id: str


class AdminRotateRequest(BaseModel):
    target_key_id: str
    promote_to_admin: bool = False


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ModelSummary(BaseModel):
    model_id: str
    display_name: str
    version: str
    architecture: str
    status: str
    availability: str
    minimum_plan: str
    free_tier_available: bool
    routing_priority: int
    # Beta 3: the limits belong in the summary, not just the detail.
    # "display model limits" is a waiter TUI requirement, and a client
    # rendering a model list should not have to make one extra request
    # per model to fill in three columns. Additive, so existing clients
    # are unaffected.
    context_limit: int = 0
    input_token_limit: int = 0
    output_token_limit: int = 0
    tool_call_limit: int | None = None


class ModelDetail(ModelSummary):
    total_parameters: float
    active_parameters: float | None
    supported_tasks: list[str]
    api_available: bool
    local_available: bool
    remote_available: bool
    pricing: dict[str, Any]
    fallback_model: str | None
    license: str
    is_example_entry: bool
    notes: str


class ModelListResponse(BaseModel):
    models: list[ModelSummary]
    count: int
    request_id: str


class ModelDetailResponse(BaseModel):
    model: ModelDetail
    request_id: str


class ModelAvailabilityResponse(BaseModel):
    model_id: str
    available: bool
    reason: str | None
    minimum_plan: str
    request_id: str


class ModelUsageResponse(BaseModel):
    model_id: str
    input_tokens_used: int
    output_tokens_used: int
    requests: int
    input_token_limit: int
    output_token_limit: int
    input_remaining: int
    output_remaining: int
    is_exhausted: bool
    request_id: str


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class UsageCurrentResponse(BaseModel):
    key_id: str
    window_seconds: float
    current_window: dict[str, int]
    all_time: dict[str, int]
    by_model: list[dict[str, Any]]
    request_id: str


class UsageRemainingResponse(BaseModel):
    model_id: str
    input_remaining: int
    output_remaining: int
    is_exhausted: bool
    request_id: str


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ConfigResponse(BaseModel):
    config: dict[str, Any]
    request_id: str


# ---------------------------------------------------------------------------
# Beta 2: routing
# ---------------------------------------------------------------------------


class RouteRequest(BaseModel):
    # Beta 3: no longer required, and no longer trusted when supplied.
    # The plan is a property of the key's server-side assignment
    # (POST /keys/assign); passing one here is only an assertion, and a
    # mismatch against the assigned plan is refused with
    # AUTH_INSUFFICIENT_SCOPE rather than honoured. See t1api.keys —
    # "the client must NEVER be trusted to decide what it can access".
    plan: str | None = Field(
        default=None,
        description="Optional assertion of the caller's plan. The server resolves the real "
        "plan from the key's assignment; a mismatch is rejected, never silently accepted.",
    )
    model_id: str | None = Field(
        default=None, description="Manual model selection. Omit for automatic routing."
    )
    input_tokens: int = Field(default=0, ge=0)
    automatic_fallback: bool = Field(
        default=False,
        description="Only meaningful with model_id set: fall through the cascade if the "
        "requested model is exhausted, instead of raising MODEL_QUOTA_EXHAUSTED.",
    )


class RouteResponse(BaseModel):
    model_id: str
    reason: str
    cascade_position: int
    policy_name: str
    considered: list[dict[str, Any]]
    request_id: str
    resolved_plan: str = Field(
        default="", description="The plan the server actually applied, from the key's assignment."
    )


# ---------------------------------------------------------------------------
# Beta 2: servers
# ---------------------------------------------------------------------------


class ServerRegisterRequest(BaseModel):
    name: str
    address: str
    capabilities: list[str] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    allow_private_address: bool = Field(
        default=False, description="Set True for local/Tailscale addresses."
    )


class ServerUpdateRequest(BaseModel):
    name: str | None = None
    status: str | None = None
    trust_level: str | None = None
    capabilities: list[str] | None = None
    tags: dict[str, str] | None = None


class ServerItem(BaseModel):
    server_id: str
    name: str
    address: str
    trust_level: str
    status: str
    capabilities: list[str]
    tags: dict[str, str]
    registered_by: str
    created_at: float
    updated_at: float
    last_seen: float | None


class ServerListResponse(BaseModel):
    servers: list[ServerItem]
    count: int
    request_id: str


class ServerDetailResponse(BaseModel):
    server: ServerItem
    request_id: str


# ---------------------------------------------------------------------------
# Beta 2: modules
# ---------------------------------------------------------------------------


class ModuleCreateRequest(BaseModel):
    name: str
    version: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModuleUpdateRequest(BaseModel):
    metadata: dict[str, Any] | None = None
    status: str | None = None


class ModuleUploadRemoteRequest(BaseModel):
    source_url: str
    allow_private: bool = False


class ModuleSyncRequest(BaseModel):
    server_id: str


class ModuleItem(BaseModel):
    module_id: str
    name: str
    version: str
    owner_key_id: str
    status: str
    source_type: str
    source_url: str | None
    checksum: str | None
    size_bytes: int | None
    deployed_servers: list[str]
    metadata: dict[str, Any]
    created_at: float
    updated_at: float


class ModuleListResponse(BaseModel):
    modules: list[ModuleItem]
    count: int
    request_id: str


class ModuleDetailResponse(BaseModel):
    module: ModuleItem
    request_id: str


class ModuleSyncResponse(BaseModel):
    job_id: str
    status: str
    request_id: str


# ---------------------------------------------------------------------------
# Beta 2: jobs
# ---------------------------------------------------------------------------


class JobCreateRequest(BaseModel):
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)


class JobItem(BaseModel):
    job_id: str
    kind: str
    payload: dict[str, Any]
    status: str
    result: dict[str, Any] | None
    error: str | None
    created_by: str
    created_at: float
    started_at: float | None
    finished_at: float | None


class JobListResponse(BaseModel):
    jobs: list[JobItem]
    count: int
    request_id: str


class JobDetailResponse(BaseModel):
    job: JobItem
    request_id: str


# ---------------------------------------------------------------------------
# Beta 2: events
# ---------------------------------------------------------------------------


class EventItem(BaseModel):
    event_id: str
    type: str
    data: dict[str, Any]
    source: str
    ts: float


class EventListResponse(BaseModel):
    events: list[EventItem]
    count: int
    request_id: str


# ---------------------------------------------------------------------------
# Beta 2: billing
# ---------------------------------------------------------------------------


class BillingBalanceResponse(BaseModel):
    account_type: str
    account_id: str
    balance: float
    request_id: str


class TransactionItem(BaseModel):
    transaction_id: str
    account_type: str
    account_id: str
    amount: float
    kind: str
    balance_after: float
    note: str
    created_by: str
    created_at: float


class TransactionListResponse(BaseModel):
    transactions: list[TransactionItem]
    count: int
    request_id: str


class PaymentTokenMintRequest(BaseModel):
    amount: float = Field(..., gt=0)
    currency: str = "USD"
    ttl_seconds: int | None = Field(default=None, ge=1)


class PaymentTokenMintResponse(BaseModel):
    token: str = Field(..., description="Raw token — shown exactly once, never retrievable again.")
    payment_token_id: str
    amount: float
    currency: str
    request_id: str


class RedeemRequest(BaseModel):
    token: str
    account_type: str = "user"
    account_id: str | None = Field(
        default=None, description="Defaults to the authenticated caller's key_id if omitted."
    )


class AddBalanceRequest(BaseModel):
    account_type: str
    account_id: str
    amount: float = Field(..., gt=0)
    note: str = ""


__all__ = [
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "StatusResponse",
    "ValidateKeyRequest",
    "ValidateKeyResponse",
    "TokenRequest",
    "TokenResponse",
    "RotateResponse",
    "AdminRotateRequest",
    "ModelSummary",
    "ModelDetail",
    "ModelListResponse",
    "ModelDetailResponse",
    "ModelAvailabilityResponse",
    "ModelUsageResponse",
    "UsageCurrentResponse",
    "UsageRemainingResponse",
    "ConfigResponse",
    "RouteRequest",
    "RouteResponse",
    "ServerRegisterRequest",
    "ServerUpdateRequest",
    "ServerItem",
    "ServerListResponse",
    "ServerDetailResponse",
    "ModuleCreateRequest",
    "ModuleUpdateRequest",
    "ModuleUploadRemoteRequest",
    "ModuleSyncRequest",
    "ModuleItem",
    "ModuleListResponse",
    "ModuleDetailResponse",
    "ModuleSyncResponse",
    "JobCreateRequest",
    "JobItem",
    "JobListResponse",
    "JobDetailResponse",
    "EventItem",
    "EventListResponse",
    "BillingBalanceResponse",
    "TransactionItem",
    "TransactionListResponse",
    "PaymentTokenMintRequest",
    "PaymentTokenMintResponse",
    "RedeemRequest",
    "AddBalanceRequest",
    # Beta 3
    "UsageHistoryItem",
    "UsageHistoryResponse",
    "UsageAggregateRow",
    "UsageAggregateResponse",
    "CostLineItem",
    "ForecastItem",
    "UsageCostResponse",
    "UsageEstimateRequest",
    "UsageEstimateResponse",
    "KeyAssignmentItem",
    "KeyItem",
    "KeyListResponse",
    "KeyImportRequest",
    "KeyImportResponse",
    "KeyAssignRequest",
    "KeyAssignResponse",
    "AuditItem",
    "AuditListResponse",
    "NetworkPolicyEntryItem",
    "NetworkPolicyResponse",
    "NetworkPolicyEntryRequest",
    "ForcedLimitItem",
    "ForcedLimitRequest",
    "ForcedLimitListResponse",
    "ForcedLimitResponse",
    "RateLimitStatusResponse",
    "ModuleDeployRequest",
    "ModuleDeployResponse",
    "ModuleFetchResponse",
    "ModuleReceiveResponse",
    "GenericOkResponse",
]


# ---------------------------------------------------------------------------
# Beta 3 — usage history / cost / estimate
# ---------------------------------------------------------------------------


class UsageHistoryItem(BaseModel):
    key_id: str
    model_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    requests: int
    endpoint: str
    server_id: str
    module_id: str
    user_id: str
    account_id: str
    ts: float


class UsageHistoryResponse(BaseModel):
    events: list[UsageHistoryItem]
    count: int
    since: float | None = None
    until: float | None = None
    limit: int
    offset: int
    request_id: str


class UsageAggregateRow(BaseModel):
    group_by: str
    value: str
    requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int


class UsageAggregateResponse(BaseModel):
    group_by: str
    rows: list[UsageAggregateRow]
    count: int
    since: float | None = None
    until: float | None = None
    request_id: str


class CostLineItem(BaseModel):
    dimension: str
    value: str
    requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str


class ForecastItem(BaseModel):
    projected_cost: float
    horizon_seconds: float
    observed_seconds: float
    observed_cost: float
    rate_per_day: float
    confidence: str
    currency: str
    basis: str


class UsageCostResponse(BaseModel):
    total_cost: float
    currency: str
    since: float | None = None
    until: float | None = None
    lines: list[CostLineItem]
    unpriced_models: list[str] = Field(
        default_factory=list,
        description="Models with recorded usage but no current registry entry. Their usage is "
        "reported and priced at zero rather than dropped, so the gap is visible.",
    )
    forecast: ForecastItem | None = None
    request_id: str


# A single report is one model call, so these ceilings are far above any
# honest value. They exist so a typo (or a malicious client) can't record
# 10**18 tokens and permanently exhaust a key's window in one request.
MAX_REPORTED_TOKENS = 10_000_000
MAX_REPORTED_REQUESTS = 1_000


class UsageEstimateRequest(BaseModel):
    model_id: str = Field(..., description="Must be a registered model_id.")
    input_tokens: int = Field(..., ge=0)
    output_tokens: int | None = Field(
        default=None,
        description="Omit for an upper-bound estimate using the model's output_token_limit.",
    )
    requests: int = Field(default=1, ge=1)


class UsageReportRequest(BaseModel):
    """Tokens a client actually consumed running a model the server routed it to.

    There is deliberately no ``key_id`` field. Usage is always recorded
    against the caller's own key: a body-supplied key_id would let any
    holder of a valid key burn somebody else's quota, which is the same
    class of mistake as Beta 2's client-supplied ``plan``.
    """

    model_id: str = Field(..., description="Must be a registered model_id.")
    input_tokens: int = Field(default=0, ge=0, le=MAX_REPORTED_TOKENS)
    output_tokens: int = Field(default=0, ge=0, le=MAX_REPORTED_TOKENS)
    requests: int = Field(default=1, ge=1, le=MAX_REPORTED_REQUESTS)
    endpoint: str = Field(default="", max_length=128)
    server_id: str = Field(default="", max_length=128)
    module_id: str = Field(default="", max_length=128)


class UsageReportResponse(BaseModel):
    recorded: bool
    model_id: str
    input_tokens: int
    output_tokens: int
    requests: int
    # The post-write quota state, so a client never has to make a second
    # call to find out whether that report exhausted the model.
    input_remaining: int
    output_remaining: int
    exhausted: bool
    request_id: str


class UsageEstimateResponse(BaseModel):
    model_id: str
    requests: int
    input_tokens: int
    output_tokens: int
    output_tokens_assumed: bool
    input_cost: float
    output_cost: float
    estimated_cost: float
    currency: str
    exceeds_input_token_limit: bool
    exceeds_context_limit: bool
    input_token_limit: int
    context_limit: int
    note: str
    request_id: str


# ---------------------------------------------------------------------------
# Beta 3 — keys
# ---------------------------------------------------------------------------


class KeyAssignmentItem(BaseModel):
    key_id: str
    plan: str
    account_id: str
    user_id: str
    server_ids: list[str]
    allowed_models: list[str]
    note: str
    assigned_at: float


class KeyItem(BaseModel):
    """Credential-free view of a key. No endpoint ever returns the raw
    key string for an existing key — only key rotation, which mints a
    new one, hands a secret back."""

    key_id: str
    key_type: str
    scopes: list[str]
    active: bool
    created_at: float
    expires_at: float | None
    prefix: str
    server_id: str
    usage_count: int
    request_count: int
    usage_cap: int | None
    request_limit: int | None
    rotated_from: str | None
    assignment: KeyAssignmentItem | None = None


class KeyListResponse(BaseModel):
    keys: list[KeyItem]
    count: int
    scope: str = Field(
        default="own",
        description="'all' for an admin caller, 'own' otherwise — a non-admin sees only their key.",
    )
    request_id: str


class KeyImportRequest(BaseModel):
    payload: dict[str, Any] = Field(
        ...,
        description="A Keymaster export object: {\"keys\": [ ... ]}. Contains raw key material, "
        "so it is never logged and never echoed back — the response carries masked ids only.",
    )


class KeyImportResponse(BaseModel):
    imported: int
    skipped: int
    imported_key_ids: list[str]
    skipped_key_ids: list[str]
    request_id: str


class KeyAssignRequest(BaseModel):
    key_id: str
    plan: str | None = None
    account_id: str | None = None
    user_id: str | None = None
    server_ids: list[str] | None = None
    allowed_models: list[str] | None = Field(
        default=None,
        description="Narrows this key to a subset of registered models. Every entry is validated "
        "against the model registry; an unregistered id is rejected with MODEL_NOT_SUPPORTED.",
    )
    note: str | None = None


class KeyAssignResponse(BaseModel):
    assignment: KeyAssignmentItem
    request_id: str


# ---------------------------------------------------------------------------
# Beta 3 — audit
# ---------------------------------------------------------------------------


class AuditItem(BaseModel):
    audit_id: str
    ts: float
    category: str
    action: str
    outcome: str
    actor_key_id: str
    actor_is_admin: bool
    resource_type: str
    resource_id: str
    client_ip: str
    request_id: str
    details: dict[str, Any]


class AuditListResponse(BaseModel):
    events: list[AuditItem]
    count: int
    total: int
    limit: int
    offset: int
    request_id: str


# ---------------------------------------------------------------------------
# Beta 3 — network policy / forced limits (waiter -B/-W/-a/-r)
# ---------------------------------------------------------------------------


class NetworkPolicyEntryItem(BaseModel):
    entry_id: str
    cidr: str
    kind: str
    reason: str
    created_by: str
    created_at: float
    expires_at: float | None = None
    expired: bool = False


class NetworkPolicyResponse(BaseModel):
    entries: list[NetworkPolicyEntryItem]
    count: int
    allow_unlisted_clients: bool
    enabled: bool
    request_id: str


class NetworkPolicyEntryRequest(BaseModel):
    cidr: str = Field(..., description="A single IP or a CIDR range, e.g. 10.0.0.0/8.")
    reason: str = ""
    ttl_seconds: float | None = Field(
        default=None, description="Optional expiry, for a temporary block or a time-boxed grant."
    )


class ForcedLimitItem(BaseModel):
    subject_type: str
    subject_id: str
    requests_per_window: int | None
    tokens_per_window: int | None
    window_seconds: float
    reason: str
    created_at: float


class ForcedLimitRequest(BaseModel):
    subject_type: str = Field(..., description="'key' or 'server'.")
    subject_id: str
    requests_per_window: int | None = Field(default=None, ge=1)
    tokens_per_window: int | None = Field(default=None, ge=1)
    window_seconds: float = Field(default=60.0, gt=0)
    reason: str = ""


class ForcedLimitListResponse(BaseModel):
    limits: list[ForcedLimitItem]
    count: int
    request_id: str


class ForcedLimitResponse(BaseModel):
    limit: ForcedLimitItem
    request_id: str


class RateLimitStatusResponse(BaseModel):
    rules: list[dict[str, Any]]
    enabled: bool
    request_id: str


# ---------------------------------------------------------------------------
# Beta 3 — remote deployment
# ---------------------------------------------------------------------------


class ModuleDeployRequest(BaseModel):
    server_ids: list[str] = Field(
        ..., min_length=1, description="Registered, trusted server_ids to push this module to."
    )


class ModuleDeployResponse(BaseModel):
    job_id: str
    status: str
    server_ids: list[str]
    request_id: str


class ModuleFetchResponse(BaseModel):
    job_id: str
    status: str
    request_id: str


class ModuleReceiveResponse(BaseModel):
    """Response to an inbound server-to-server push. The pusher compares
    ``checksum`` against what it sent — a mismatch means the bytes changed
    in flight, and it fails the deployment rather than trusting the 200."""

    module_id: str
    checksum: str
    size_bytes: int
    status: str
    request_id: str


class GenericOkResponse(BaseModel):
    ok: bool = True
    detail: str = ""
    request_id: str


# ---------------------------------------------------------------------------
# T1 v1.0.26.8.0.1 — LM Studio bridge
# ---------------------------------------------------------------------------


class BridgeModel(BaseModel):
    model_id: str
    kind: str = "unknown"
    loaded: bool = False
    architecture: str = ""
    quantization: str = ""
    context_length: int = 0
    max_context_length: int = 0
    publisher: str = ""
    supports_vision: bool = False
    supports_tools: bool = False


class BridgeProbe(BaseModel):
    base_url: str
    reachable: bool
    usable: bool = False
    latency_ms: float = 0.0
    model_count: int = 0
    loaded_count: int = 0
    native_api: bool = False
    cors_enabled: bool | None = None
    cors_allow_origin: str = ""
    error: str = ""
    error_code: str = ""


class BridgeStatusResponse(BaseModel):
    enabled: bool
    configured_url: str
    probe: BridgeProbe | None = None
    discovered: list[BridgeProbe] = Field(default_factory=list)
    request_id: str


class BridgeModelsResponse(BaseModel):
    base_url: str
    models: list[BridgeModel]
    count: int
    loaded_count: int
    request_id: str


class BridgeChatMessage(BaseModel):
    role: str
    # ``content`` is str for a plain message and a list of parts for a
    # multimodal one (OpenAI's own shape). Typed as Any because
    # constraining it here would reject a valid vision request that LM
    # Studio itself accepts, and the far side is the real validator.
    content: Any


class BridgeChatRequest(BaseModel):
    messages: list[BridgeChatMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    stream: bool = False
    base_url: str | None = None


class BridgeChatResponse(BaseModel):
    model: str
    content: str
    finish_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    base_url: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)
    request_id: str


# ---------------------------------------------------------------------------
# T1 v1.0.26.8.0.1 — HyperLink
# ---------------------------------------------------------------------------


class HyperLinkEndpoint(BaseModel):
    url: str
    kind: str
    priority: int
    note: str = ""


class HyperLinkEndpointsResponse(BaseModel):
    server_name: str
    t1_version: str
    endpoints: list[HyperLinkEndpoint]
    tailscale: bool
    reachable_off_lan: bool
    #: This installation's stable identifier (0.72.4). A client pins it
    #: at pairing time and re-checks it on every reconnection: a name can
    #: be claimed by any machine on the network, this cannot. See
    #: ``hypernix.hyperlink.identity``.
    server_fingerprint: str = ""
    #: Whether this server accepts keyless connections from trusted
    #: networks at all, and whether the *caller's own* origin qualifies.
    #: The second is what lets an app say "this network can connect
    #: without a key" instead of making the user find out by failing.
    trusted_network: bool = False
    keyless_available_here: bool = False
    #: How this caller was classified: loopback | tailnet | lan | public.
    origin_trust: str = ""
    request_id: str


class HyperLinkPeer(BaseModel):
    """One machine on the tailnet. A candidate, never a trusted server.

    ``verified`` is always ``false`` and is present to say so in the
    payload rather than only in the documentation: nothing in peer
    discovery authenticates anything, and ``name``/``server_name`` are
    what a peer calls itself, which any machine can claim.
    """

    name: str
    address: str
    url: str
    online: bool = False
    reachable: bool = False
    t1_version: str = ""
    server_name: str = ""
    detail: str = ""
    os: str = ""
    verified: bool = False


class HyperLinkPeersResponse(BaseModel):
    peers: list[HyperLinkPeer] = Field(default_factory=list)
    count: int = 0
    reachable: int = 0
    tailscale: bool = False
    detail: str = ""
    request_id: str


class PairingCreateRequest(BaseModel):
    label: str = ""
    scopes: list[str] | None = None
    ttl_seconds: float | None = None


class PairingCodeResponse(BaseModel):
    code: str
    expires_at: float
    seconds_remaining: float
    scopes: list[str]
    label: str = ""
    qr_payload: dict[str, Any] = Field(default_factory=dict)
    endpoints: list[HyperLinkEndpoint] = Field(default_factory=list)
    request_id: str


class PairingRedeemRequest(BaseModel):
    code: str
    device_name: str
    platform: str = "ios"
    app_version: str = ""


class PairingRedeemResponse(BaseModel):
    """The one and only time the device token is transmitted."""

    device_id: str
    device_token: str
    name: str
    scopes: list[str]
    server_name: str = ""
    t1_version: str = ""
    request_id: str


class DeviceSummary(BaseModel):
    device_id: str
    name: str
    platform: str
    app_version: str
    scopes: list[str]
    paired_by: str
    created_at: float
    last_seen: float | None = None
    last_address: str = ""
    revoked: bool = False
    revoked_at: float | None = None


class DeviceListResponse(BaseModel):
    devices: list[DeviceSummary]
    count: int
    request_id: str


class DeviceResponse(BaseModel):
    device: DeviceSummary
    request_id: str


class SessionCreateRequest(BaseModel):
    title: str = ""
    model_id: str = ""
    backend: str = ""
    system_prompt: str = ""


class SessionUpdateRequest(BaseModel):
    title: str | None = None
    model_id: str | None = None
    backend: str | None = None
    system_prompt: str | None = None
    archived: bool | None = None


class SessionSummary(BaseModel):
    session_id: str
    title: str
    owner: str
    device_id: str = ""
    model_id: str = ""
    backend: str = ""
    system_prompt: str = ""
    created_at: float
    updated_at: float
    archived: bool = False
    message_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]
    count: int
    request_id: str


class SessionResponse(BaseModel):
    session: SessionSummary
    request_id: str


class MessageSummary(BaseModel):
    message_id: str
    session_id: str
    seq: int
    role: str
    content: str
    model_id: str = ""
    attachment_ids: list[str] = Field(default_factory=list)
    created_at: float
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageListResponse(BaseModel):
    session_id: str
    messages: list[MessageSummary]
    count: int
    request_id: str


class HyperLinkChatRequest(BaseModel):
    content: str = ""
    attachment_ids: list[str] = Field(default_factory=list)
    model_id: str | None = None
    backend: str | None = None          # "lmstudio" (default) | "registry"
    temperature: float | None = None
    max_tokens: int | None = None
    token_budget: int = 8000
    stream: bool = False


class HyperLinkChatResponse(BaseModel):
    session_id: str
    user_message: MessageSummary
    assistant_message: MessageSummary
    model_id: str
    backend: str
    request_id: str


class AttachmentSummary(BaseModel):
    file_id: str
    sha256: str
    filename: str
    content_type: str
    size_bytes: int
    owner: str
    device_id: str = ""
    session_id: str = ""
    created_at: float
    is_image: bool = False
    is_text: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class AttachmentResponse(BaseModel):
    file: AttachmentSummary
    request_id: str


class AttachmentListResponse(BaseModel):
    files: list[AttachmentSummary]
    count: int
    total_bytes: int
    request_id: str


class HFResolveRequest(BaseModel):
    """Either link, or both. That is the whole point of the endpoint."""

    page_url: str = ""
    file_url: str = ""
    prefer: str = "strict"              # strict | file | page
    include_vision: bool = True
    offline: bool = False


class HFFile(BaseModel):
    filename: str
    url: str
    size_bytes: int = 0
    role: str = "weights"
    part_index: int = 0
    part_total: int = 0
    sha256: str = ""


class HFResolveResponse(BaseModel):
    repo_id: str
    revision: str
    host: str
    quantization: str = ""
    gated: bool = False
    private: bool = False
    license: str = ""
    total_bytes: int = 0
    total_size_human: str = ""
    file_count: int = 0
    is_split: bool = False
    has_vision: bool = False
    primary_file: str = ""
    files: list[HFFile] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    metadata_from_api: bool = False
    request_id: str


# ---------------------------------------------------------------------------
# T1 v1.0.26.8.1.0 — auth undo/redo and backups
# ---------------------------------------------------------------------------


class AuthUndoResponse(BaseModel):
    direction: str
    entry: dict[str, Any]
    can_undo: bool
    can_redo: bool
    request_id: str


class AuthHistoryResponse(BaseModel):
    entries: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    can_undo: bool = False
    can_redo: bool = False
    next_undo: dict[str, Any] | None = None
    next_redo: dict[str, Any] | None = None
    request_id: str


class BackupSummary(BaseModel):
    backup_id: str
    created_at: float
    format_version: int
    t1_version: str = ""
    hypernix_version: str = ""
    sections: dict[str, int] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)
    attachment_count: int = 0
    label: str = ""
    created_by: str = ""
    excluded: dict[str, str] = Field(default_factory=dict)
    size_bytes: int = 0
    filename: str = ""


class BackupListResponse(BaseModel):
    backups: list[BackupSummary]
    count: int
    sections_captured: list[str] = Field(default_factory=list)
    request_id: str


class BackupCreateRequest(BaseModel):
    label: str = ""


class BackupRestoreRequest(BaseModel):
    backup_id: str
    #: ``None`` means "decide from ?confirm" — a restore without an
    #: explicit confirmation is a dry run.
    dry_run: bool | None = None
    sections: list[str] = Field(default_factory=list)


class BackupRestoreResponse(BaseModel):
    backup_id: str
    dry_run: bool
    created_at: float
    label: str = ""
    sections: list[dict[str, Any]] = Field(default_factory=list)
    excluded: dict[str, str] = Field(default_factory=dict)
    note: str = ""
    request_id: str


# ---------------------------------------------------------------------------
# 0.72.1 — HyperLink model downloads
# ---------------------------------------------------------------------------


class HFDownloadRequest(BaseModel):
    """Either a ``repo_id``, or links to resolve into one."""

    repo_id: str = ""
    revision: str = "main"
    filenames: list[str] = Field(default_factory=list)
    kind: str = "auto"                  # auto | gguf | pytorch
    page_url: str = ""
    file_url: str = ""
    prefer: str = "strict"


class HFDownloadResponse(BaseModel):
    job_id: str
    repo_id: str
    revision: str
    filenames: list[str] = Field(default_factory=list)
    kind: str = "auto"
    request_id: str


class DownloadedModelsResponse(BaseModel):
    models: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    directory: str = ""
    request_id: str


# ---------------------------------------------------------------------------
# Inference (T1 v1.0.26.9.2.1)
#
# The governed inference surface. /bridge/lmstudio/* is a pass-through: it
# hands the caller's model string straight to LM Studio, so the registry,
# the routing cascade and the quota never see the request. These do the
# opposite — the server resolves the model, applies the plan's cascade,
# refuses an exhausted key before any work happens, and meters what was
# actually spent — which is the design principle the rest of the API is
# built on, applied to the one path that had escaped it.
# ---------------------------------------------------------------------------


class InferenceMessage(BaseModel):
    role: str
    content: str


class InferenceChatRequest(BaseModel):
    model: str
    messages: list[InferenceMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    #: Let the routing engine substitute when the named model is
    #: exhausted or unavailable. Off by default: a silent substitution is
    #: the thing the spec forbids, so asking for it has to be explicit.
    allow_fallback: bool = False


class InferenceCompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    allow_fallback: bool = False


class InferenceResponse(BaseModel):
    #: What actually ran. Differs from the requested model when the
    #: cascade substituted, which is why it is reported rather than
    #: echoed back.
    model: str
    requested_model: str
    content: str
    finish_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    currency: str = "USD"
    backend: str = ""
    substituted: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)
    request_id: str


class InferenceEmbeddingsRequest(BaseModel):
    model: str
    input: list[str]


class InferenceEmbeddingsResponse(BaseModel):
    model: str
    embeddings: list[list[float]] = Field(default_factory=list)
    dimensions: int = 0
    input_tokens: int = 0
    backend: str = ""
    request_id: str


class InferenceTokenCountRequest(BaseModel):
    model: str
    #: Exactly one of these. Counting what you are about to send is the
    #: point, so both shapes a caller might send are accepted.
    text: str | None = None
    messages: list[InferenceMessage] | None = None


class InferenceTokenCountResponse(BaseModel):
    model: str
    tokens: int
    #: What this many input tokens would cost on this model, before the
    #: reply. A caller sizing a prompt against a spend cap needs this
    #: before it commits, not after.
    estimated_input_cost: float = 0.0
    currency: str = "USD"
    remaining_input_tokens: int | None = None
    method: str = ""
    request_id: str


class InferenceBackend(BaseModel):
    name: str
    kind: str
    reachable: bool
    detail: str = ""
    address: str = ""


class InferenceBackendsResponse(BaseModel):
    backends: list[InferenceBackend] = Field(default_factory=list)
    default: str = ""
    request_id: str


# ---------------------------------------------------------------------------
# 0.72.4 — training administration
# ---------------------------------------------------------------------------


class TrainingRunItem(BaseModel):
    """One run, as ``training.monitor`` reconciled it.

    ``percent`` and ``eta_seconds`` are ``None`` rather than ``0`` when
    the trainer never declared a schedule — a dashboard that renders 0%
    for a job two hours in states something false with confidence.
    """

    run_id: str
    name: str = ""
    state: str = ""
    epoch: int = 0
    total_epochs: int = 0
    step: int = 0
    total_steps: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    loss_history: list[float] = Field(default_factory=list)
    checkpoints: list[str] = Field(default_factory=list)
    model: str = ""
    started_at: float = 0.0
    updated_at: float = 0.0
    finished_at: float | None = None
    error: str = ""
    job_id: str = ""
    pid: int = 0
    log_path: str = ""
    percent: float | None = None
    eta_seconds: float | None = None
    is_active: bool = False


class TrainingRunListResponse(BaseModel):
    runs: list[TrainingRunItem] = Field(default_factory=list)
    count: int = 0
    active: int = 0
    root: str = ""
    request_id: str


class TrainingRunResponse(BaseModel):
    run: TrainingRunItem
    request_id: str


class TrainingLogResponse(BaseModel):
    run_id: str
    #: Empty with a populated ``detail`` when the run has no log — a
    #: run started outside the launcher has nowhere to read from, and
    #: that is not an error worth a 404 for the whole run.
    log: str = ""
    lines: int = 0
    path: str = ""
    detail: str = ""
    request_id: str


class TrainingCheckpointItem(BaseModel):
    path: str
    exists: bool = False
    size_bytes: int = 0
    modified_at: float | None = None


class TrainingCheckpointListResponse(BaseModel):
    run_id: str
    checkpoints: list[TrainingCheckpointItem] = Field(default_factory=list)
    count: int = 0
    request_id: str


class TrainingResourcesResponse(BaseModel):
    gpus: list[dict[str, Any]] = Field(default_factory=list)
    cpu_percent: float | None = None
    ram_total_mb: int | None = None
    ram_used_mb: int | None = None
    ram_percent: float | None = None
    request_id: str


class TrainingControlResponse(BaseModel):
    run: TrainingRunItem
    action: str
    note: str = ""
    request_id: str
