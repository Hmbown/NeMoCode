# NVIDIA API Authentication

> How to authenticate with NVIDIA API endpoints.

## Getting an API Key

1. Go to `https://build.nvidia.com`
2. Sign in with your NVIDIA account (or create one)
3. Navigate to any model page
4. Click "Get API Key"
5. Copy and store securely

## Using the API Key

### HTTP Header
```bash
curl -H "Authorization: Bearer $NVIDIA_API_KEY" \
  https://integrate.api.nvidia.com/v1/chat/completions
```

### Environment Variable
```bash
export NVIDIA_API_KEY="nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Python SDK
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)
```

## Rate Limits

- Free tier: Rate-limited serverless endpoints for development
- Enterprise: Higher rate limits via NVIDIA AI Enterprise subscription
- Self-hosted NIMs: No API rate limits (limited by your hardware)

## API Key Format

NVIDIA API keys follow the pattern: `nvapi-` followed by an alphanumeric string.

## NGC API Keys (for container pulls)

Separate from build.nvidia.com keys. Used to pull NIM containers from NGC:

1. Go to `https://ngc.nvidia.com`
2. Navigate to Setup > API Key
3. Generate key
4. Use with `docker login nvcr.io`

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>
```
