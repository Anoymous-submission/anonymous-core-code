# Release verification

`RELEASE_MANIFEST.json` records the source file list and SHA-256 hashes.
To verify a freshly unpacked release, run:

```bash
python -B scripts/audit_release.py
```

Run this before installing packages or generating outputs in the source directory.
The audit rejects unlisted files, including environment folders, editable-install
metadata and test caches. Six visually reviewed paper figures are included as
PNG files containing only image headers and pixel data. The audit checks their
format, rejects metadata and trailing data, and verifies their manifest hashes.
It also checks for symlinks, other binary artifacts, private
paths, credential patterns and unexpected Git author metadata.

Use a separate pristine copy when checking an archive after development.
Generated results are local artifacts and are ignored by Git. Changes to the
source require updating the manifest and re-auditing the release archive.

For experiment scope and required data, see coverage.
