import numpy as np
from scipy.io.wavfile import write
from scipy import signal
from scipy.signal import freqresp
import matplotlib.pyplot as plt
import pandas as pd

sf = 20000
t = 300

# Generating uniform distribution white noise
sig = np.random.uniform(low=-1.0, high=1.0, size=t*sf)
sig = sig / max(abs(sig)) * 32767 * 0.7

# write to wav
write('exported unfiltered_white_noise.wav', sf, sig.astype(np.int16))

def bp_filtering(sig, sf, n, bp_freq, output_filename):
    # Filtering
    sos = signal.butter(n, bp_freq, btype='bandpass', fs=sf, output='sos')
    f_sig = signal.sosfilt(sos, sig)    # filtered signal
    # write to wav
    write(output_filename+'.wav', sf, f_sig.astype(np.int16))
    # write to csv
    df = pd.DataFrame({
        'time': np.linspace(0, len(sig)/sf, len(sig)),
        'amplitude': f_sig
    })
    df.to_csv(output_filename+'.csv', index=False)

    # Potting PSD
    freqs, Pxx = signal.welch(f_sig, sf, nperseg=2**16, noverlap=2**12*.75)
    print('Freqeuncy resolution = {}'.format(freqs[1]-freqs[0]))
    plt.figure()
    plt.plot(np.linspace(0, t, t*sf), f_sig)
    plt.xlabel('time, s')
    plt.ylabel('amplitude, 16bit PCM')
    plt.tight_layout()

    fig, ax = plt.subplots()
    ax.semilogx(freqs, Pxx, 'b')
    ax.grid(which='major')
    ax.grid(which='minor', linestyle=':')
    ax.set_xlabel(r'Frequency [Hz]')
    ax.set_ylabel('PSD [V**2/Hz]')
    plt.xlim(11, 25000)
    ax.set_xticks([16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    ax.set_xticklabels(['16', '31.5', '63', '125', '250',
                      '500', '1k', '2k', '4k', '8k', '16k'])
    fig.tight_layout()
    fig.savefig(output_filename+'.png', dpi=200)
    return

# 200Hz to 4000Hz signal
bp_filtering(sig, sf, 4, [100, 8000],'exported_200to4kHz_white_noise')