# Full NAMO Training-Geometry Signature Artifact Design

## Goal

Let the `tdn39` Full NAMO population builder prove complete-room geometry disjointness from HY5U's private training corpus without copying the 1.3-million-row training H5, exposing its room paths, or granting broad access to roughly 198,000 training XML files.

## Scope

This change adds one compact, immutable training-geometry signature artifact and teaches the existing held-out population builder to consume it. It does not change the room identity definition, structural population rules, generator, Full NAMO protocol, success denominator, or statistical analysis.

The canonical identity remains `verify_geom_disjoint.py::geom_sig`: a full-room MD5 over sorted walls plus initial movable-obstacle geometry, with robot start and goal excluded. Wall-only signatures remain descriptive floorplan-overlap metadata and are not leak exclusions by themselves.

## Exporter

Add `scripts/pipeline/export_geom_signatures.py`. It accepts one or more repeated `--train-xmls` H5/TXT/JSON references, resolves them with the existing `load_xmls`, canonicalizes and deduplicates paths, computes geometry through the existing `geom_sig`/`sig_map` helpers, and writes one new JSON file through `--out`.

The exporter refuses to overwrite an existing artifact and fails without writing anything if any source reference is unreadable, any referenced XML is unparseable, no training rooms are found, or the final signature sets are empty. A partial export must never be accepted as a complete training reference.

The deterministic artifact schema is:

```json
{
  "schema": "namo-room-geometry-signatures-v1",
  "sources": [
    {
      "path": "/absolute/path/to/training.h5",
      "sha256": "<source-file-sha256>"
    }
  ],
  "counts": {
    "xml_paths": 1302659,
    "unique_xml_paths": 198267,
    "unique_room_signatures": 198267,
    "unique_floorplan_signatures": 10
  },
  "full_signatures": ["<sorted-md5>"] ,
  "wall_signatures": ["<sorted-md5>"]
}
```

Exact counts above illustrate the fields; the exporter records measured values rather than assuming them from experiment prose. Source SHA-256 values bind the compact artifact to the exact private inputs, while the artifact's own SHA-256 binds what crosses accounts.

The artifact contains no XML paths, labels, images, actions, or H5 rows beyond aggregate counts and source provenance.

## Builder integration

Extend `scripts/pipeline/build_full_namo_population.py` with repeated `--train-signatures` inputs. The builder may consume direct `--train-xmls` references, compact `--train-signatures` artifacts, or both, but the CLI requires at least one training reference across the two forms.

For every signature artifact, the builder requires the exact schema identifier, required source and count fields, lowercase 32-character hexadecimal MD5 entries, sorted unique full and wall signature lists, nonempty signature sets, and count values consistent with list lengths. Any malformed or internally inconsistent artifact aborts the build before output creation.

The builder unions direct-reference and compact-artifact full signatures for exact-scene leak rejection and unions wall signatures for descriptive floorplan overlap. The existing candidate manifest/probe equality, zero-simulation structural gates, deterministic ordering, output refusal, and no-empty-population behavior remain unchanged.

`population_audit.json` records every direct training reference, every compact artifact path and SHA-256, each artifact's embedded source provenance, and merged training signature counts. This makes the final freeze reviewable without granting the evaluation account access to private training data.

## Cross-account data flow

1. On the data-owning CS account, check out the committed exporter revision and run it against `/common/users/dm1487/scratch_namo/aquaman/round0/hybrid_train_v1.h5`.
2. Verify the exporter reports zero unparseable rooms and review its measured counts.
3. Copy the JSON into the existing read-only NAMO shared-artifact tree for `tdn39` and record its SHA-256 in the experiment card.
4. Transfer that exact JSON to `/scratch/tdn39/full_namo_heldout_v1/artifacts/` and verify the SHA-256 again.
5. Run the held-out population builder with `--train-signatures` and no private H5 dependency.

The exported signatures are a deterministic derivative of the private source and may be regenerated from the recorded source hash and committed exporter revision. They are not a substitute for registering the underlying HY5U training lineage.

## Testing

Exporter tests use minimal XML fixtures and a manifest reference to verify deterministic sorted output, source hashing, exact full/wall signatures, overwrite refusal, empty-input rejection, and fail-closed handling of an unparseable referenced XML.

Builder tests verify that a compact signature artifact excludes an exact training-geometry leak, preserves wall-only overlap as nonleaking metadata, records artifact provenance, accepts mixed direct and compact references, rejects malformed schema/signatures/counts, and requires at least one training reference at the CLI.

The existing direct H5/TXT/JSON behavior remains covered, and the full `full_namo_sim_exp` regression suite must remain green before the export is run or generation is launched.

## Documentation and launch record

Update the NAMO data-pipeline inventory, `full_namo_sim_exp/README.md`, and the live held-out experiment card with the exact export and builder commands. Record the exporter commit, private source H5 SHA-256, compact artifact SHA-256, measured signature counts, transfer verification, and final population audit.

