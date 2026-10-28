# Peer-review audit

**Supported claim:** smooth additive background correction substantially improves pixel ranking over uncorrected intensity clustering in this explicitly specified synthetic population.

**Unsupported claim:** the full intuitionistic model is a superior astronomical detector. The Gaussian-filter comparator wins on AP, Dice, background error, and runtime in the executed primary study.

The original 18% / 24.6% Dice values are user-reported, not reproduced. The new study has a new simulator and explicitly defined labels; its scores cannot be used as a direct before/after comparison with those values.

## Mathematical corrections integrated

1. Additive photon-counting observation model replaces a literal transfer of MRI bias physics.
2. Lowest-centroid background anchoring resolves the additive constant ambiguity.
3. Every implemented spatial and regression objective is stated exactly; the alternating scheme is described as a heuristic rather than endowed with an unproved convergence theorem.
4. Sugeno lambda is constrained to nonnegative values and membership/nonmembership/hesitation are tested for valid decomposition.
5. Bright-tail FCM membership reversal is excluded from the ranking score. The resulting affine rank invariance is disclosed, including identical AP for uncorrected Base and Spatial FCM.
6. AP supplements, rather than erases, threshold-dependent Dice. Reference labels remain explicit and label-dependent.
7. Validation chooses thresholds; test data never select thresholds or model parameters.
8. Confidence intervals resample independent images instead of correlated pixels.
9. A simpler background-aware comparator and a hesitation-free corrected ablation are included.
10. Hubble is qualitative. There is no invented Hubble segmentation truth, and a space-based RGB image is not claimed to validate terrestrial skyglow.

## Evidence limits relevant to submission

The deliverable is a complete reproducible computational study, not evidence of IEEE acceptance or a certified production pipeline. Broader claims would require ground-based calibrated FITS images, independent injection recovery, object-level deblending and catalog metrics, photometric error, out-of-family backgrounds and PSFs, and tuned comparisons with established extractors. The primary five-row ablation changes spatial and hesitation jointly; the separate 81-run factorial sensitivity grid is provided to expose these terms independently.

Author name and institutional affiliation are deliberately anonymous rather than fabricated. Replace them with the actual authors before submission. The external-validation table is explicitly marked unrun and must remain so unless independently measured.
