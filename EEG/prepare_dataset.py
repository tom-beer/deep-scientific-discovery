import os
import pickle
import glob
import re
import pyedflib
import numpy as np
import scipy.signal as ssignal
from sklearn import preprocessing


def read_edf_and_annot(edf_file, hyp_file, channels=['EEG', 'EEG(sec)'], shhs='1'):

    rawsignal = read_edfrecord(edf_file, channels, shhs)
    stages = read_annot_regex(hyp_file)
    # check that they have the same length
    if shhs == '1':
        if 'EOG(R)' in channels:
            # todo: add check for 50Hz of EOG
            pass
        else:
            assert np.size(rawsignal) % (30 * 125) == 0.
            assert int(np.size(rawsignal) / (len(channels) * 30 * 125)) == len(stages)
    else:
        assert np.size(rawsignal) % (30 * 128) == 0.
        assert int(np.size(rawsignal) / (len(channels) * 30 * 128)) == len(stages)
    return rawsignal, np.array(stages)


def read_annot_regex(filename):
    with open(filename, 'r') as f:
        content = f.read()
    # Check that there is only one 'Start time' and that it is 0
    patterns_start = re.findall(
        r'<EventConcept>Recording Start Time</EventConcept>\n<Start>0</Start>',
        content)
    assert len(patterns_start) == 1
    # Now decode file: find all occurences of EventTypes marking sleep stage annotations
    patterns_stages = re.findall(
        r'<EventType>Stages.Stages</EventType>\n' +
        r'<EventConcept>.+</EventConcept>\n' +
        r'<Start>[0-9\.]+</Start>\n' +
        r'<Duration>[0-9\.]+</Duration>',
        content)
    # print(patterns_stages[-1])
    stages = []
    starts = []
    durations = []
    for pattern in patterns_stages:
        lines = pattern.splitlines()
        stageline = lines[1]
        stage = int(stageline[-16])
        startline = lines[2]
        start = float(startline[7:-8])
        durationline = lines[3]
        duration = float(durationline[10:-11])
        assert duration % 30 == 0.
        epochs_duration = int(duration) // 30

        stages += [stage] * epochs_duration
        starts += [start]
        durations += [duration]
    # last 'start' and 'duration' are still in mem
    # verify that we have not missed stuff..
    assert int((start + duration) / 30) == len(stages)
    return stages


def read_edfrecord(filename, channels, shhs='1'):
    # SHHS1 is at 125 Hz, SHHS2 at 128Hz

    f = pyedflib.EdfReader(filename)
    # print("Startdatetime: ", f.getStartdatetime())
    signal_labels = f.getSignalLabels()

    sigbufs = []
    for channel in channels:
        # assert channel == 'EEG' or channel == 'EEG(sec)', "Channel must be EEG or EEG(sec)"
        assert channel in signal_labels
        idx_chan = [i for i, x in enumerate(signal_labels) if x == channel][0]
        sigbuf = np.zeros((f.getNSamples()[idx_chan]))
        sigbuf[:] = f.readSignal(idx_chan)
        samplingrate = len(sigbuf) / f.file_duration

        sigbufs.append(sigbuf)
    sigbufs = np.vstack(sigbufs)

    return sigbufs


def extract_dataset(prop_sampled=1.0, preprocessed_dir=None, nb_patients=None, ds=False, num_slices=4):
    np.random.seed(42)
    shhs = '1'
    fs = 125  # original sampling frequency [Hz]
    fs_ds = 80  # downsampled sampling frequency [Hz]
    shhs_base_dir = os.path.join(os.getcwd(), 'shhs')
    assert num_slices == 4, 'This code supports 4 slices only for now'
    # if filter, in both cases it will be at 0.5-25Hz
    # + subsampling / 2: 64 or 62.5Hz, i.e. 1920 or 1875 points per 30s segment.

    # read and preprocessed data will be saved in
    # preprocessed_dir
    # dataset_dir = 'filtered' if ds else 'nofilter'
    dataset_dir = 'ALL'
    dataset_dir = dataset_dir + str(prop_sampled)

    if preprocessed_dir is None:
        preprocessed_dir = os.path.join(shhs_base_dir, 'preprocessed', 'shhs' + shhs, dataset_dir)

    edf_dir = os.path.join(shhs_base_dir, 'polysomnography', 'edfs', 'shhs' + shhs)
    print(edf_dir)
    edf_names = glob.glob(os.path.join(edf_dir, '*.edf'))

    hyp_dir = os.path.join(shhs_base_dir, 'polysomnography', 'annotations-events-nsrr', 'shhs' + shhs)
    print(hyp_dir)
    hyp_names = glob.glob(os.path.join(hyp_dir, 'shhs*.xml'))

    print("number of records: ", len(edf_names))
    print("number of hypnograms", len(hyp_names))

    if nb_patients is None:
        nb_patients = len(edf_names)
    print('Number of patients: ', nb_patients)

    # loop on records
    counter_nostage, counter_nosig = 0, 0
    for i in range(nb_patients):
        name = os.path.basename(edf_names[i])
        name = name[:-4]
        hyp_try_name = os.path.join(hyp_dir, name + '-nsrr.xml')
        save_file = os.path.join(preprocessed_dir, name + '.p')
        if not os.path.exists(save_file):
            try:
                print('#####################################')
                print('loading data and annotations for patient %d: ' % i),
                print(edf_names[i])
                try:
                    rawsignal, stages = read_edf_and_annot(
                        edf_names[i], hyp_try_name, channels=['EEG', 'EEG(sec)'], shhs=shhs)

                    raw_eog, stages = read_edf_and_annot(
                        edf_names[i], hyp_try_name, channels=['EOG(R)', 'EOG(L)'], shhs=shhs)

                    raw_emg, stages = read_edf_and_annot(
                        edf_names[i], hyp_try_name, channels=['EMG'], shhs=shhs)

                except FileNotFoundError:
                    counter_nosig += 1
                    continue
                # verify that there is no other stage
                labels, counts = np.unique(stages, return_counts=True)
                assert np.max(labels) <= 5

                samplesper30s = 30 * fs
                if ds:
                    ds_ratio = fs / fs_ds
                    lowcut = 0.5
                    highcut = 25
                    nyquist_freq = fs / ds_ratio
                    low = lowcut / nyquist_freq
                    high = highcut / nyquist_freq
                    b, a = ssignal.butter(3, [low, high], btype='band')
                    rawsignal = ssignal.filtfilt(b, a, rawsignal, axis=1)
                    # subsample (with average)
                    rawsignal = ssignal.resample(rawsignal, int(rawsignal.shape[1] // ds_ratio), axis=1)
                    samplesper30s /= ds_ratio

                # and now rearrange into nparray of training samples
                numberofintervals = rawsignal.shape[1] / samplesper30s
                examples_patient_i = np.array(
                    np.split(rawsignal, numberofintervals, axis=1))

                examples_patient_i_eog = np.array(
                    np.split(raw_eog, numberofintervals, axis=1))

                examples_patient_i_emg = np.array(
                    np.split(raw_emg, numberofintervals, axis=1))


                # Remove excess 'pre-sleep' and 'post-sleep' wake
                # so that total 'out of night' wake is at most equal
                # to the biggest other class
                if stages[np.argmax(counts)] == 0:
                    # print('Wake is the biggest class. Trimming it..')
                    second_biggest_count = np.max(
                        np.delete(counts, np.argmax(counts)))
                    occurencesW = np.where(stages == 0)[0]
                    last_W_evening_index = min(
                        np.where(occurencesW[1:] - occurencesW[0:-1] != 1)[0])
                    nb_evening_Wepochs = last_W_evening_index + 1
                    first_W_morning_index = len(stages) - min(np.where(
                        (occurencesW[1:] - occurencesW[0:-1])[::-1] != 1)[0]) - 1
                    nb_morning_Wepochs = len(stages) - first_W_morning_index
                    nb_pre_post_sleep_wake_eps = \
                        nb_evening_Wepochs + nb_morning_Wepochs
                    if nb_pre_post_sleep_wake_eps > second_biggest_count:
                        total_Weps_to_remove = nb_pre_post_sleep_wake_eps - second_biggest_count
                        print('removing %i wake epochs' % total_Weps_to_remove)
                        if nb_evening_Wepochs > total_Weps_to_remove:
                            stages = stages[total_Weps_to_remove:]
                            examples_patient_i = examples_patient_i[total_Weps_to_remove:, :]
                            examples_patient_i_eog = examples_patient_i_eog[total_Weps_to_remove:, :]
                            examples_patient_i_emg = examples_patient_i_emg[total_Weps_to_remove:, :]

                        else:
                            evening_Weps_to_remove = nb_evening_Wepochs
                            morning_Weps_to_remove = total_Weps_to_remove - evening_Weps_to_remove
                            stages = stages[evening_Weps_to_remove:-morning_Weps_to_remove]
                            examples_patient_i = examples_patient_i[evening_Weps_to_remove:-morning_Weps_to_remove, :]
                            examples_patient_i_eog = examples_patient_i_eog[evening_Weps_to_remove:-morning_Weps_to_remove, :]
                            examples_patient_i_emg = examples_patient_i_emg[evening_Weps_to_remove:-morning_Weps_to_remove, :]

                else:
                    pass

                # merge labels 3 and 4
                indices = np.where(stages == 4)
                stages[indices] = 3
                # now use label 4 for REM
                le = preprocessing.LabelEncoder()
                le.fit([0, 1, 2, 3, 5])
                stages = le.transform(stages)

                # show counts
                cl, cnts = np.unique(stages, return_counts=True)

                # Exclude patients without an epoch of every stage
                if cl.shape[0] < 5:
                    counter_nostage += 1
                    continue

                # sample partial dataset
                if prop_sampled < 1:
                    selected_epochs_num = int(prop_sampled * examples_patient_i.shape[0])
                    if selected_epochs_num < 1:
                        continue
                    selected_idxs = np.random.permutation(np.arange(start=2, stop=examples_patient_i.shape[0] - 1))[
                                    :selected_epochs_num].astype(int)
                    sampled_examples_mat = np.zeros((selected_epochs_num, examples_patient_i.shape[1],
                                                     examples_patient_i.shape[2]))

                    sampled_examples_mat_eog = np.zeros((selected_epochs_num, examples_patient_i_eog.shape[1],
                                                     examples_patient_i_eog.shape[2]))
                    sampled_examples_mat_emg = np.zeros((selected_epochs_num, examples_patient_i_emg.shape[1],
                                                     examples_patient_i_emg.shape[2]))

                    for new_i, orig_i in enumerate(selected_idxs):

                        sampled_examples_mat[new_i, :, :] = examples_patient_i[orig_i, :, :]
                        sampled_examples_mat_eog[new_i, :, :] = examples_patient_i_eog[orig_i, :, :]
                        sampled_examples_mat_emg[new_i, :, :] = examples_patient_i_emg[orig_i, :, :]


                    stages_patient_i = stages[selected_idxs]
                    examples_patient_i = sampled_examples_mat
                    examples_patient_i_eog = sampled_examples_mat_eog
                    examples_patient_i_emg = sampled_examples_mat_emg

                # pickle this patient data to disk
                data = examples_patient_i, examples_patient_i_eog, examples_patient_i_emg, stages_patient_i
                if not os.path.exists(os.path.dirname(save_file)):
                    os.makedirs(os.path.dirname(save_file))
                with open(save_file, 'wb') as fp:
                    pickle.dump(data, fp)

            except FileNotFoundError:
                print('File not found.')
            except AssertionError:
                print('AssertionError. This patient probably has more than 0 to 5 labels...')
                print('Skipping this patient.')
            finally:
                pass
    print(f'counter_nostage: {counter_nostage}, counter_nosig: {counter_nosig}')
    return preprocessed_dir


def train_val_test_split_names(preprocessed_dir, ds, nb_patients):
    np.random.seed(42)
    # dataset_dir = 'filtered0.05' if ds else 'nofilter0.05'
    dataset_dir ='ALL' + str(0.05)
    preprocessed_names = glob.glob(os.path.join(
        preprocessed_dir, '*.p'))
    preprocessed_names = preprocessed_names[:int(nb_patients)]
    # shuffle
    r = np.arange(len(preprocessed_names))
    np.random.shuffle(r)
    preprocessed_names = [preprocessed_names[i] for i in r]

    tvt_proportions = (0.5, 0.2, 0.3)
    n_train = int(tvt_proportions[0] * len(preprocessed_names))
    print('n_train: ', n_train)
    n_valid = int(tvt_proportions[1] * len(preprocessed_names))
    print('n_valid: ', n_valid)
    names_train = preprocessed_names[0:n_train]
    names_valid = preprocessed_names[n_train:n_train + n_valid]
    names_test = preprocessed_names[n_train + n_valid:]
    exp_dir = os.path.join('shhs', 'results', dataset_dir)

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    with open(os.path.join(exp_dir, "names_train.pkl"), 'wb') as fp:
        pickle.dump([name.split('-')[-1].split('.')[0] for name in names_train], fp)
    with open(os.path.join(exp_dir, "names_valid.pkl"), 'wb') as fp:
        pickle.dump([name.split('-')[-1].split('.')[0] for name in names_valid], fp)
    with open(os.path.join(exp_dir, "names_test.pkl"), 'wb') as fp:
        pickle.dump([name.split('-')[-1].split('.')[0] for name in names_test], fp)

    return


if __name__ == '__main__':
    nb_patients = int(1e8)
    ds = True
    dataset_dir = 'ALL'
    dataset_dir = dataset_dir + str(0.05)
    shhs_base_dir = os.path.join(os.getcwd(), 'shhs')

    preprocessed_dir = os.path.join(shhs_base_dir, 'preprocessed', 'shhs' + '1', dataset_dir)
    preprocessed_dir = extract_dataset(prop_sampled=0.05, nb_patients=nb_patients, ds=ds, num_slices=4)
    train_val_test_split_names(preprocessed_dir, ds=ds, nb_patients=nb_patients)
