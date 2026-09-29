import os


def _configured_path(env_name, default):
    """Resolve a per-machine data root without editing tracked source files."""
    configured = os.environ.get(env_name, "").strip()
    return configured if configured else default


path_mapper = {
    'faced_de': r'D:/大学/脑机接口/FACED/',
    'seed_de_lds': _configured_path('SGDA_SEED_DATA_ROOT', r'D:/大学/脑机接口/SEED/'),
    'seediv_de_lds': _configured_path('SGDA_SEEDIV_DATA_ROOT', r'D:/大学/脑机接口/SEED_IV/'),
    'dreamer': r'D:/大学/脑机接口/DREAMER/DE_processed_1s.npy',
    'deap': r'D:/大学/脑机接口/DEAP/data_preprocessed_python/data_preprocessed_python',
    # read_seedv_feature expects {subject}_data.npy and {subject}_label.npy.
    # The repository's prepared files are under this extracted feature folder.
    'seedv_de_lds': _configured_path(
        'SGDA_SEEDV_DATA_ROOT', r'D:/大学/脑机接口/SEED_V/unzipped_DE_features/'
    ),
}
