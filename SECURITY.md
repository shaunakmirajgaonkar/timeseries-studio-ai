# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ |

## Reporting a Vulnerability

TimeSeries Studio AI runs entirely locally and does not transmit data to any
external service by default, which limits (but does not eliminate) its attack
surface. Relevant concerns include:

- Arbitrary file read/write via the data loader (e.g. path traversal in
  uploaded file names)
- Dependency vulnerabilities in third-party packages (pandas, statsmodels,
  Prophet, XGBoost, PyTorch, Streamlit, Plotly)
- Unsafe deserialization if forecast/state files are loaded from untrusted
  sources

If you discover a security vulnerability, please **do not open a public
issue**. Instead, email:

**mirajgaonkarshaunak@gmail.com**

Include:
- A description of the vulnerability and its potential impact
- Steps to reproduce
- Any suggested fix, if you have one

We aim to acknowledge reports within 5 business days and will credit
reporters in the changelog unless anonymity is requested.

## Scope

This policy covers the code in this repository. It does not cover
vulnerabilities in upstream dependencies — please report those to the
respective maintainers, though we're happy to be notified so we can pin a
patched version.
