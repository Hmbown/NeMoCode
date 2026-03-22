# Security Policy

## Supported Versions

Only the latest release of NeMoCode receives security updates.

## Reporting a Vulnerability

To report a security vulnerability, please open a [GitHub Security Advisory](https://github.com/Hmbown/NeMoCode/security/advisories/new). Do not open a public issue for security vulnerabilities.

## Threat Model

NeMoCode is a **local CLI tool** that executes shell commands and edits files based on LLM output. Understanding the trust boundaries is important for safe use.

### Trust Boundaries

| Component | Trust Level | Notes |
|-----------|-------------|-------|
| User | Trusted | The user initiates all actions and approves tool calls |
| NVIDIA NIM API | Trusted | TLS-encrypted, NVIDIA-operated |
| LLM Output | Untrusted | Model output drives tool execution — review before approving |
| Local filesystem | No sandbox | NeMoCode can read/write any file accessible to the user |
| Shell commands | No sandbox | `bash_exec` runs with the user's full shell permissions |

### What NeMoCode Does NOT Provide

- **Sandboxing:** Shell commands and file edits run with the user's full permissions
- **Network isolation:** NeMoCode makes outbound HTTPS requests to configured endpoints
- **Server mode authentication:** NeMoCode is designed for single-user local use

### API Key Handling

- API keys are stored in the system keyring (via `keyring` package) or environment variables
- Keys are never logged, printed, or included in error messages
- Keys are never sent to third-party endpoints (only to configured NVIDIA NIM endpoints)

### `.env` File Handling

- Reading `.env` files requires explicit user permission via the permission engine
- `.env` contents are not included in context or prompts by default

## Recommendations for Users

- Review tool calls before approving them, especially `bash_exec` and file writes
- Use the permission engine to restrict tool access (e.g., deny `bash_exec` for certain commands)
- Do not run NeMoCode with elevated privileges (sudo, root)
- Keep your API key secure — use the system keyring rather than environment variables when possible
- Use `nemo doctor` to verify your configuration
