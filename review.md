## Summary

Uncommitted publication-status WIP (`git diff HEAD` vs `79e97a1`): six metadata/docs files retitle the work as Song et al. (2026) Nat. Commun. and add paper DOI `10.1038/s41467-026-78922-x`. `CITATION.cff` keeps `cff-version: 1.2.0`, `type: software`, and concept DOI `10.5281/zenodo.23223234`. README still shows paper and Zenodo `@software` as separate BibTeX. User-facing `docs/` citation blocks do not.

## Strengths

- [S1] `.zenodo.json` stays `upload_type: software` and links the article via `related_identifiers` / `isSupplementTo` rather than collapsing records.
- [S2] README keeps a distinct Zenodo `@software` block and still states GitHub Cite reads `CITATION.cff`.
- [S3] No personal absolute paths; CLAUDE.md remains English agent docs; Python/uv rules untouched.

## Weaknesses

- [W1] **MAJOR (documented, hard):** `docs/index.md` and `docs/index.zh.md` replace the only citation with `@article{song2026archival,...}` and drop `@software`. **README.md** rule: keep Zenodo software citation distinct from the paper. Same surfaces are English/Chinese user docs.
- [W2] **MAJOR (documented, hard):** `docs/index.zh.md` adds English “Song et al. (2026), *Nature Communications*” outside the BibTeX fence. **User/repo docs convention:** `docs/index.zh.md` stays Chinese except BibTeX. (The English paper title in quotes was already present.)
- [W3] **MINOR (judgement, Duplicated Code):** identical `@article{song2026archival,...}` appears in `README.md`, `docs/index.md`, and `docs/index.zh.md`; author list is restated in `CITATION.cff` `preferred-citation`. Canonical CFF + README pair already exist.

Shotgun Surgery across six files is **suppressed**: these are the documented citation surfaces; the repo layout requires touching them together.

No Mysterious Name / Feature Envy / code-shape smells (docs-only diff). Skip tooling-enforced YAML/CFF schema.

## Questions for Authors

- [Q1] Should `docs/` citation sections mirror README (article + Zenodo software), or is paper-only intentional for MkDocs?
- [Q2] For `index.zh.md`, prefer 「宋爽等（2026）」/「《自然·通讯》」 vs leaving Latin bibliographic tokens?

## Verdict

Revision priority: restore distinct software citation on docs landing pages; sinicize the zh lead-in. Evidence quality high (diff is the artifact). Confidence: high on W1–W2; medium on whether docs were meant to follow README’s dual-cite rule.

## Revision Plan

1. **W1:** In both `docs/index*.md` Citation sections, keep `@article` and restore a Zenodo `@software` block matching README (concept vs version DOI as README already does).
2. **W2:** Rewrite the zh intro in Chinese; leave only the BibTeX (and maybe the DOI URL) in Latin script.
3. **W3 (optional):** Treat `CITATION.cff` + README as sources of truth; don’t add a third BibTeX variant.

---

## Inline annotations

> `docs/index.md` Citation: `@article{song2026archival,` … (former `@software{shifting_baseline,` removed)
**[W1] HARD:** User-docs citation is paper-only. **README.md:** keep Zenodo software citation distinct from the paper.

> `docs/index.zh.md` Citation: same `@article{song2026archival,` replacement
**[W1] HARD:** Chinese counterpart drops software cite. Same README rule.

> `docs/index.zh.md`: `本仓库对应 Song et al. (2026), *Nature Communications*`
**[W2] HARD:** English bibliographic prose outside BibTeX. **Convention:** zh file stays Chinese except BibTeX.

> README / both docs: `@article{song2026archival,` with the same 11-author list
**[W3] JUDGEMENT — Duplicated Code:** same BibTeX shape in three hunks.

> `CITATION.cff`: `type: software` / `doi: 10.5281/zenodo.23223234` unchanged; `preferred-citation:` article added
**Compliant:** documented CFF software record preserved; GitHub Cite still reads CFF.

> `.zenodo.json` `related_identifiers` → `10.1038/s41467-026-78922-x` `isSupplementTo`
**Compliant:** software record remains; paper is related, not substituted.

## Sources

- Diff: `git diff HEAD` (working tree vs `79e97a1`)
- Standards: `CLAUDE.md`, `README.md` Citation section, `CITATION.cff` header (`cff-version` / `type` / concept DOI)
