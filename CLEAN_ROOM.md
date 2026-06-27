# Clean Room Policy

Babel is maintained as an original, open-source French correction system. The repository may use public and open resources, but it must not contain restricted payloads, recovered binary strings, proprietary rule sets, or private local paths.

## Allowed

- Original glue code written for Babel.
- Public resource names when used for attribution.
- Public data sources with their licenses and notices.

## Not Allowed

- Commercial product names used as source material or branding.
- Vendor-specific private paths, project IDs, or local machine paths.
- Decoded binary filenames or recovered symbol names.
- Proprietary payloads, rules, or weights.

## Review Rule

Before merging or publishing, run the banned-token scan and confirm it passes against code, docs, manifests, and configuration files.

If a source must be mentioned for attribution, prefer the source's public name and a short license note rather than copying source-like implementation details.
