# Public release audit

## Scope

The tracked tree and every reachable commit were checked for private keys,
credential formats, authenticated URLs, environment files, personal email
addresses, IP addresses, hostnames, absolute user paths, and identifying binary
metadata. Translator archives were also inspected as strings. The numerical
artifacts contain hashes, token IDs, logits, timings, and fitted maps rather
than raw WikiText passages.

## Finding and repair

No credential, private key, personal email address, IP address, or private
hostname was found. The release did contain machine-local checkout paths in 16
manifests, 10 audits, and 15 editable-install records. Internal agent operating
instructions were also present in the root and source snapshots.

The public release removes those instructions and machine-local fields. Run
manifests now retain only repository-relative config paths, auditors emit
repository-relative artifact paths, and environment capture excludes the local
editable-install path. Artifact manifests, audits, reproduction bindings, and
the experiment ledger were rehashed after this metadata-only transformation.
Raw records, summaries, timings, data indices, and translator tensors were not
rewritten. Migrated audit records carry `public_metadata_only: true` so the
transformation is explicit.

Internal automation workflow files and review checklists were removed from both
the root tree and historical source snapshots. The public repository keeps the
scientific protocol, trial cards, experiment ledger, report, and independent
audit code.

The public branch history was replaced after the repair so the removed paths do
not remain in an earlier reachable commit. Copies made while the old revision
was public are outside this repository's control.

## Information intentionally retained

GPU model, memory, CUDA, Python, framework versions, model revisions, dataset
revision, UTC run times, and the full dependency inventory remain public. They
describe the experimental environment and do not identify the machine or its
operator.

## Preventive controls

- New manifests do not record the Python executable, working directory, or
  absolute config path.
- New audits record artifact paths relative to the repository.
- Bootstrap dependency capture excludes editable installations.
- Common key, certificate, and environment filenames are ignored.
- The experiment ledger still verifies every retained artifact and audit hash.
