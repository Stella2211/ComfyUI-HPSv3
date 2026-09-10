# Publishing to Comfy Registry

The publisher is `stella`, configured in `pyproject.toml`. Store the publishing API key only in the GitHub Actions repository secret `REGISTRY_ACCESS_TOKEN`.

Update `project.version` to an unpublished version and push `pyproject.toml` to `main` to run the `Publish to Comfy Registry` workflow. It can also be started manually from the Actions page. Use a new version for each release.

Validate and inspect the archive before publishing:

```sh
uvx --from comfy-cli==1.20.0 comfy node validate
uvx --from comfy-cli==1.20.0 comfy node pack
```

Inspect `node.zip`. It must not contain `third_party/`, model weights, credentials, or local environments. The installer retrieves pinned dependencies directly from upstream. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the unresolved upstream implementation licensing; publishing this extension does not resolve that issue.
