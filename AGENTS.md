# Repository editing rules

Read README.md, docs/reproducibility.md and THIRD_PARTY_NOTICES.md first.
Preserve public module paths. Never silently alter a research objective, graph
orientation, data split, ontology, prompt or historical report claim. Add tests
and a protocol note for result-changing fixes. Do not fabricate benchmark scores,
DOIs, data rights, model provenance or successful test/CI outcomes.

Run `make lint`, `make test`, a new output-directory smoke run and `make report`
when applicable. Use the locked CPU environment. Never call paid providers or
load arbitrary external checkpoints as part of tests. Never print or commit
credentials, private footage or student identifiers. Existing historical source
and attribution are not disposable generated artifacts.
