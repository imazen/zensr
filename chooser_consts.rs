// baked 2026-07-26 from chooser_features.tsv (lianli), sklearn L1-select+L2-refit
// eval (pinned, never fit): see benchmarks/chooser_fit_2026-07-26.txt
const CHOOSER_FEATURES: &[(&str, f32, f32, f32)] = &[  // (name, median, iqr, weight)
    ("variance", 2558.88, 3411.01, 0.68966),
    ("colourfulness", 15.3235, 26.3934, -1.28115),
    ("laplacian_variance", 0.926529, 2.57378, -2.0077),
    ("variance_spread", 1.06317, 0.548007, -0.96721),
    ("distinct_color_bins", 659.0, 1833.0, 0.244856),
    ("palette_density", 0.020111, 0.0559385, 0.24484),
    ("cr_peak_sharpness", 0.0, 2.0, -0.176542),
    ("luma_histogram_entropy", 2.17591, 1.70967, -2.15955),
    ("skin_tone_fraction", 0.005394, 0.154432, 0.228646),
    ("quant_survival_y", 0.071506, 0.088433, 1.2906),
    ("aq_map_p90", 5.11083, 1.26966, 0.51535),
    ("noise_floor_y_p25", 0.0, 0.118233, -0.526132),
    ("noise_floor_y_p90", 1.0, 0.004217, 0.0330069),
    ("noise_floor_uv_p50", 0.0, 0.029115, 0.0293905),
    ("noise_floor_uv_p75", 0.026731, 0.087091, 0.214907),
    ("noise_floor_uv_p90", 0.082923, 0.233621, 0.204648),
    ("luma_kurtosis", 10.7992, 19.2993, 0.35128),
    ("chroma_luma_covariance_cb", -0.164535, 0.515691, -0.601435),
    ("chroma_luma_covariance_cr", 0.034303, 0.380783, 0.229072),
    ("orientation_energy_ratio", 1.14894, 0.0731275, -0.381605),
    ("xyb444_color_loss", 0.692348, 0.030016, -0.39849),
];
const CHOOSER_BIAS: f32 = 2.62081;
const CHOOSER_THRESHOLD: f32 = 0.8; // p(graphics) gate, precision-biased
