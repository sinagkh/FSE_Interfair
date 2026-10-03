# Third-party implementation notices

The authors' implementation is accompanied by the upstream files needed for its adapters. Original copyright and license notices are retained alongside the licensed code. References and pinned source records are in `SOURCE_RECORDS.json` and `fetch_manifest.json`.

- HIFI: the released interaction regularizer; MIT notice at `interfair/vendor/HIFI/LICENSE.txt`.
- Fair-SMOTE: the released sample generator and its license in `interfair/vendor/fair_smote/`.
- FAIRER / DRAlign: the native FAIRER loss/training routine and its license in `interfair/vendor/fairer/`.
- RTDL: FT-Transformer implementation and license in `interfair/vendor/rtdl/`.
- ScottKnottESD 2.0.3 and effsize: tested pure-R packages under `environment/R_library/`, with upstream source tarballs and their source records under this directory. Their DESCRIPTION files preserve authorship and license metadata; GPL texts are included.
- LTDD, CoT-Phi, and MirrorFair: pinned source retrieval rather than embedding source without an included redistribution license. `scripts/fetch_upstream.py` verifies file hashes and, for archive downloads, archive and member hashes.

Third-party author names and source repository URLs are acknowledgments to other work, not artifact authorship metadata. The package does not relicense upstream code.
