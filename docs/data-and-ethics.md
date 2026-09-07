# Data, model provenance and responsible use

## Inventory

| Asset | Status |
| --- | --- |
| `data/best.pt` | Pre-existing detector checkpoint, 5835140 bytes |
| Report source/figures | Imported from the report repository at the recorded SHA |
| `refs/*.pdf` | Pre-existing third-party reference publications |
| Classroom videos/CVAT images | Not supplied as a reproducible dataset in this tree |
| Original synthetic suites/results | Not supplied as a complete benchmark archive |
| Smoke VAR data | Generated locally without personal data or external APIs |

Detector SHA-256: `bfabb7854c90318c520a485ba2b0c9b6194d85191e8b09462918d8840697fa6f`.
Training corpus, annotation agreement, architecture provenance, license for the
weights, subgroup performance and calibration have not been established by this
integration. Keeping the checkpoint is not a claim that it is validated or
cleared for redistribution. Only load trusted checkpoints: deserialization and
custom model code can execute code. Obtain maintainer provenance before reuse.

## Research safeguards

Obtain appropriate institutional approval, informed consent/assent and permission
for classroom collection and retention. Use session/subject-separated evaluation
where relevant. Remove identifying footage and metadata from shared artifacts.
Report illustrations can contain classroom imagery; review permissions and
identifiability before a public release or wider distribution. No new consent or
license is inferred from an asset already being present in a repository.

Behavior-share measurements are not psychological diagnoses, reliable measures
of motivation, or evidence that a particular student caused an outcome. Do not
use this prototype for automated discipline, grading, eligibility or other
high-stakes decisions. Keep a human review step and describe detection error,
confounding and uncertainty with any graph.

LLM proposals can transmit behavior context and graph information to DeepSeek;
synthetic generation calls Gemini; enabled W&B logging can transmit telemetry.
Review provider/data-processing terms and minimize payloads. There is no network
in the smoke path, and CI requires no API credentials. Never put raw videos,
student identifiers, provider responses containing personal data, or real keys
in issues or CI artifacts.

## Release checklist still requiring a maintainer

Confirm detector/figure/dataset rights and attribution; provide approved benchmark
access instructions and checksums; rerun protocol-v2 experiments; review report
claims; select a release commit/tag and archive it with a persistent identifier.
A DOI is intentionally not invented. Branch protections and required CI checks
are repository settings and must be enabled separately by an administrator.
