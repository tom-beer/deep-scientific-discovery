



rr_feature_names = ['rri_min', 'rri_max', 'rri_median', 'rri_rms', 'rri_std',
                    'rri_multiscale_entropy', 'pnn20', 'pnn50']
p_wave_feature_names = ['p_wave_amp', 'p_wave_amp_std', 'p_wave_eng', 'p_wave_corr_coeff_median',
                        'p_wave_higuchi_fractal_median', 'p_wave_corr_coeff_std', 'p_wave_time']
full_waveform_feature_names = ['full_waveform_std', 'full_waveform_kurtosis']
all_feature_names = rr_feature_names + p_wave_feature_names + full_waveform_feature_names

feature_names = {'rr': rr_feature_names,
                 'p_wave': p_wave_feature_names,
                 'all': all_feature_names}


def feature_subset_to_names(feature_subset):
    feature_names = None
    extracted_rep_file_name = None

    rr_feature_names = ['rri_min', 'rri_max', 'rri_median', 'rri_rms', 'rri_std',
                        'rri_multiscale_entropy', 'pnn20', 'pnn50']
    p_wave_feature_names = ['p_wave_amp', 'p_wave_amp_std', 'p_wave_eng', 'p_wave_corr_coeff_median',
                            'p_wave_higuchi_fractal_median', 'p_wave_corr_coeff_std', 'p_wave_time']
    full_waveform_feature_names = ['full_waveform_std', 'full_waveform_kurtosis']
    all_feature_names = rr_feature_names + p_wave_feature_names + full_waveform_feature_names

    if 'rr' in feature_subset:
        feature_names = rr_feature_names
        extracted_rep_file_name = 'rr_ext_rep_features.pkl'
    if 'all' in feature_subset:
        feature_names = all_feature_names
        extracted_rep_file_name = 'all_ext_rep_features.pkl'
    elif 'full_waveform' in feature_subset:
        feature_names = full_waveform_feature_names
        extracted_rep_file_name = 'full_waveform_rep_not_extracted_yet.pkl'
    elif 'p_wave' in feature_subset:
        feature_names = p_wave_feature_names
        extracted_rep_file_name = 'p_wave_ext_rep_features.pkl'

    feature_len = len(feature_names)

    return feature_names, extracted_rep_file_name, feature_len


def load_features():
    return


def update_rep_dict(sig_names, rep, rep_dict):
    for i in range(len(sig_names)):
        rep_dict[sig_names[i]] = rep[i].cpu().detach().numpy()
    return rep_dict
