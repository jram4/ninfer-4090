# Security policy

## Supported version

Security fixes target the latest source release of the RTX 4090/Qwen3.6-27B port.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use **Security → Report a
vulnerability** on the GitHub repository to submit a private report. Include affected revision,
reproduction steps, impact, and any proposed mitigation.

The maintainer will acknowledge a complete report as soon as practical, coordinate validation,
and publish a fix and advisory when appropriate. Please avoid accessing data that is not yours,
disrupting third-party systems, or publicly disclosing the issue before remediation is available.

## Scope notes

Model artifacts are downloaded separately and must be checksum-verified before use. The server
accepts an optional API key, but deployments remain responsible for network isolation, TLS,
credential rotation, and least-privilege service configuration.
