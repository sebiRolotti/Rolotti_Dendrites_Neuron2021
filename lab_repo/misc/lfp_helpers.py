# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import zscore
from scipy.signal import argrelextrema

import xml.etree.ElementTree as ET
import os
from shutil import copyfile
import re
import warnings
import glob
import h5py


def loadEVT(filepath, evt):

    f = open(filepath, 'rb')
    data = f.read()

    if evt == 'noise':
        delims = ['tstart', 'tend']
    elif evt == 'ripple':
        delims = ['tstart', 'tpeak', 'tend']

    n_evt_types = len(delims)

    regex = '\\' + '\\n|\\'.join(delims) + '\\n'

    linearized = re.split(regex, data)[:-1]
    linearized = [float(x) for x in linearized]

    evt_lists = [linearized[i::n_evt_types] for i in xrange(n_evt_types)]

    return {d: evt_list for d, evt_list in zip(delims, evt_lists)}

def writeEVT(filepath, evt_list, evt_names):
    # evt_list is a list of lists, each list containing each time, in ms, of an event
    # e.g., tstart, tpeak, tend

    format_str = ' {:0.1f}\t{}\n' * len(evt_names)
    for evt in evt_list:
        interleaved_list = [x for t in zip(evt, evt_names) for x in t]

        write_str = ''.join(format_str.format(*interleaved_list))

        with open(filepath, 'a') as f:
            f.write(write_str)

def loadEEG(eegBaseName, channels=0):

    eegTree = ET.parse(eegBaseName + '.xml')
    eegRoot = eegTree.getroot()

    nChanElem = eegRoot.findall('.//nChannels')
    nChan = int(nChanElem[0].text)

    sampFreqElem = eegRoot.findall('.//samplingRate')
    sampFreq = float(sampFreqElem[0].text)

    EEG = np.fromfile(eegBaseName + '.eeg', dtype=np.int16)
    EEG = np.reshape(EEG, (-1, nChan))
    EEG = np.transpose(EEG, (1, 0))

    if channels != 0:
        EEG = EEG[channels]
    else:
        channels = range(nChan)

    tEEG = np.arange(EEG.shape[1]) / sampFreq
    eegObj = {'EEG': EEG, 'tEEG': tEEG, 'sampeFreq': sampFreq,
              'channels': channels, 'nChannels': nChan,
              'fileBase': eegBaseName, 'filePath': os.getcwd()}

    return eegObj


def SaveBinary1(arrayIn, fileName):

    # voltageRange = 20
    # scalingConstant = (2**15) / 10
    scalingConstant = 1
    arrayOut = arrayIn * scalingConstant
    arrayOut = arrayOut.round()
    arrayOut = arrayOut.transpose()
    arrayOut = arrayOut.ravel()
    arrayOut = arrayOut.astype('int16')

    with open(fileName, 'wb') as f:
        f.write(arrayOut)


def ConvertFromRHD(rhdFullPath, destDir=None):
    
    try:
        from rhd import load_intan_rhd_format
    except ImportError:
        warnings.warn("rhd package not found, aborting conversion")
        return
    
    xmlBase = '/analysis/LFP_utils/LFPToIntanBase1.xml'

    assert os.path.isfile(rhdFullPath)

    pathParts = os.path.split(rhdFullPath)
    rhdDir = pathParts[0]
    rhdBaseName = os.path.splitext(pathParts[1])[0]

    if destDir is None:
        destDir = rhdDir
        destPath = rhdFullPath.replace('.rhd', '.eeg')
    else:
        if not os.path.isdir(destDir):
            os.mkdir(destDir)
        destPath = os.path.join(destDir, rhdBaseName + '.eeg')

    result = load_intan_rhd_format.read_data(rhdFullPath)

    rhdSigFS = result['frequency_parameters']['amplifier_sample_rate']
    rhdADCFS = result['frequency_parameters']['board_adc_sample_rate']
    # rhdSigNSamples = len(result['amplifier_data'][0])
    outFS = 1250
    # sigNewNSamples = int(np.round(rhdSigNSamples*(outFS/rhdSigFS)))
    sigDownRatio = int(np.round(rhdSigFS / outFS))
    adcDownRatio = int(np.round(rhdADCFS / outFS))
    sigConstant = 1.
    adcConstant = 1000.

    eegOut = []
    for chan in result['amplifier_data']:
        eegOut.append(sigConstant * fastDownSample(chan, sigDownRatio))

    for adcChan in result['board_adc_data']:
        eegOut.append(adcConstant * fastDownSample(adcChan, adcDownRatio))

    eegOut = np.array(eegOut)

    # Use pulses to find frame times, clip off beginning of LFP to effectively
    # sync LFP and imaging, and then save frame times
    frame_times = np.where(zscore(np.diff(eegOut[-1, :])) > 2)[0] + 1
    eegOut = eegOut[:, frame_times[0]:]

    SaveBinary1(eegOut, destPath)


def find_frame_times(eegFile, signal_idx=-1, min_interval=40, every_n=1):
    """Find imaging frame times in LFP data using the pockels blanking signal.
    Due to inconsistencies in the fame signal, we look for local maxima. This
    avoids an arbitrary threshold that misses small spikes or includes two
    nearby time points that are part of the same frame pulse.

    Parameters
    ----------
    eegFile : str
        Path to eeg data file

    signal_idx : int
        Index of the pockels signal, e.g. eeg[signal_idx, :], default -1

    min_interval : int
        Minimum radius around local maxima to enforce, default 40

    every_n : int
        Return every nth frame time, useful for multiplane data, default 1

    Returns
    -------
    frame times : array, shape (n_frame_times, )
    """
    pc_signal = loadEEG(eegFile.replace('.eeg', ''))['EEG'][signal_idx, :]

    # break ties for local maxima by increasing first point by 1
    same_idx = np.where(np.diff(pc_signal) == 0)[0]
    pc_signal[same_idx] += 1
    pc_signal = np.abs(np.diff(pc_signal))

    frame_times = argrelextrema(pc_signal, np.greater, order=min_interval)[0]
    return frame_times[::every_n]

def closest_idx(array, values):
    # for each value in values return index of element
    # in array that is closest to that value

    # make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = ((idxs == len(array))|(np.fabs(values - array[np.maximum(idxs-1, 0)]) < np.fabs(values - array[np.minimum(idxs, len(array)-1)])))
    idxs[prev_idx_is_less] -= 1

    return idxs

def fastDownSample(chanIn, downRatio):
    padN = downRatio - np.mod(len(chanIn), downRatio)
    padN = np.mod(padN, downRatio)
    chanIn = np.concatenate((chanIn, np.tile(np.nan, (padN))))
    chanOut = chanIn.reshape(-1, downRatio)
    chanOut = np.nanmean(chanOut, 1)
    return chanOut

# Experiment object methods

def get_ripple_channel(expt):

    rip_file = glob.glob(os.path.join(expt.LFPFilePath(), '*rip.evt'))[0]
    return int(rip_file.split('.rip.evt')[0][-1]) - 1

def get_wav(expt):

    return glob.glob(os.path.join(expt.LFPFilePath(), '*Wav.mat'))[0]

def get_spectro(expt):

    with h5py.File(get_wav(expt), 'r') as f:

        if 'eegWav' in f.keys():
            try:
                ref = f['eegWav']['Wavs'][get_ripple_channel(expt), 0]
                data = f[ref][:]
                freqs = f['eegWav']['freqs'][:]
            except KeyError:
                ref = f['eegWav'][get_ripple_channel(expt), 0]
                data = f[ref][:]
                freqs = f['freqs'][:]
        else:
            ref = f['Wavs'][get_ripple_channel(expt), 0]
            data = f[ref][:]
            freqs = f['freqs'][:]

    return data, freqs
