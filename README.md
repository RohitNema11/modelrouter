# ModelRouter

A FastAPI service that routes queries across fast, general, and reasoning LLM tiers. It uses ML classification and rule-based fallback to select a model, with retries and compatible model failover for provider errors.

## Features
- Query routing by tier and capability
- Chat API with token, latency, and estimated-cost metadata
- Retry and fallback handling
- Evaluation runner for comparing routed and fixed-model requests