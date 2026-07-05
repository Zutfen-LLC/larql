// ── Public error type ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum RemoteMoeError {
    /// Could not reach the shard server (connection refused, DNS failure, etc.).
    Unreachable { url: String, cause: String },
    /// The server responded with a non-2xx status.
    ServerError { status: u16, body: String },
    /// Response body could not be parsed.
    BadResponse(String),
    /// No shard owns a required expert ID.
    NoShard { expert_id: usize },
    /// HTTP client construction failed.
    Client(String),
}

impl std::fmt::Display for RemoteMoeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unreachable { url, cause } => {
                write!(f, "expert shard unreachable: {url} ({cause})")
            }
            Self::ServerError { status, body } => {
                write!(f, "expert shard returned {status}: {body}")
            }
            Self::BadResponse(msg) => write!(f, "bad expert response: {msg}"),
            Self::NoShard { expert_id } => write!(f, "no shard owns expert {expert_id}"),
            Self::Client(msg) => write!(f, "HTTP client error: {msg}"),
        }
    }
}

impl RemoteMoeError {
    /// GPU-3002 — classify whether this error is worth a retry against a
    /// different replica (or re-dispatch to the same shard after backoff).
    ///
    /// **Retryable** (transient, likely to succeed on another replica):
    ///   - [`RemoteMoeError::Unreachable`] — connection refused, DNS failure,
    ///     network partition, or read/connect timeout. The replica is down or
    ///     unreachable from us.
    ///   - [`RemoteMoeError::ServerError`] with status `503` (saturation /
    ///     overload), `500`/`502`/`504` (gateway), or `429` (rate limit). All
    ///     indicate the shard is alive but temporarily unwilling/able.
    ///
    /// **Fatal** (will fail on every replica identically):
    ///   - [`RemoteMoeError::NoShard`] — no shard in the *map* owns this
    ///     expert. Retrying against other replicas cannot help because the
    ///     candidate set is already exhausted by the routing step.
    ///   - [`RemoteMoeError::ServerError`] with status `400` — the request
    ///     shape itself is wrong (e.g. no-shard-configured on the server).
    ///   - [`RemoteMoeError::BadResponse`] — the shard returned a
    ///     syntactically/semantically invalid response; a re-dispatch would
    ///     hit the same bug.
    ///   - [`RemoteMoeError::Client`] — local HTTP client construction
    ///     failure; not network-related.
    ///
    /// Timeouts surface as `Unreachable` (reqwest maps connect/read timeouts
    /// to errors) or as `ServerError` with a 504 gateway status; both are
    /// retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Unreachable { .. } => true,
            Self::ServerError { status, .. } => matches!(status, 429 | 500 | 502 | 503 | 504),
            Self::NoShard { .. } | Self::BadResponse(_) | Self::Client(_) => false,
        }
    }
}

impl std::error::Error for RemoteMoeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unreachable_is_retryable() {
        let e = RemoteMoeError::Unreachable {
            url: "http://x".into(),
            cause: "refused".into(),
        };
        assert!(e.is_retryable());
    }

    #[test]
    fn server_error_503_is_retryable() {
        let e = RemoteMoeError::ServerError {
            status: 503,
            body: "saturated".into(),
        };
        assert!(e.is_retryable());
    }

    #[test]
    fn server_error_400_is_fatal() {
        let e = RemoteMoeError::ServerError {
            status: 400,
            body: "no shard configured".into(),
        };
        assert!(!e.is_retryable());
    }

    #[test]
    fn no_shard_is_fatal() {
        let e = RemoteMoeError::NoShard { expert_id: 7 };
        assert!(!e.is_retryable());
    }

    #[test]
    fn bad_response_is_fatal() {
        let e = RemoteMoeError::BadResponse("shape mismatch".into());
        assert!(!e.is_retryable());
    }

    #[test]
    fn client_error_is_fatal() {
        let e = RemoteMoeError::Client("tls".into());
        assert!(!e.is_retryable());
    }
}
