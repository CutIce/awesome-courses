# No Imports Allowed!


def backwards(sound):
    rate = sound['rate']
    left = sound['left']
    right = sound['right']

    new_left = left.copy()
    new_right = right.copy()
    new_left.reverse()
    new_right.reverse()
    new_sound = {'rate': rate, 'left': new_left, 'right': new_right}
    write_wav(new_sound, 'sounds/new_sound.wav')
    return new_sound

def mix(sound1, sound2, p):
    rate1 = sound1['rate']
    rate2 = sound2['rate']
    l1 = sound1['left']
    l2 = sound2['left']
    r1 = sound1['right']
    r2 = sound2['right']
    length = 0
    if rate1 != rate2:
        return None
    assert len(l1) == len(r1), "sound 1 not precise"
    assert len(l2) == len(r2), "sound 2 not precise"
    length = min(len(l1), len(l2))
    n_l = []
    n_r = []

    n_l = [p * l1[i] + (1-p) * l2[i] for i in range(min(len(l1), len(l2)))]
    n_r = [p * r1[i] + (1-p) * r2[i] for i in range(min(len(r1), len(r2)))]
    new_sound = {'rate': rate1, 'left': n_l, "right": n_r}
    return new_sound


def echo(sound, num_echos, delay, scale):
    rate = sound['rate']
    l = sound['left']
    r = sound['right']

    size = len(l)
    sample_delay = round(rate * delay)
    nl = [0] * (size + num_echos * sample_delay)
    nr = [0] * (size + num_echos * sample_delay)

    for i in range(size):
        for k in range(num_echos+1):
            nl[i + k * sample_delay] += l[i] * (scale ** k)
            nr[i + k * sample_delay] += r[i] * (scale ** k)

    new = {'rate': rate, 'left': nl, 'right': nr}
    return new



def pan(sound):
    ra = sound['rate']
    l = sound['left']
    r = sound['right']

    n = len(l)
    assert len(l) == len(r), "Size Not Matches"

    nl = l.copy()
    nr = r.copy()
    for i in range(n):
        nl[i] *= (n-1-i) / (n-1)
        nr[i] *= i / (n-1)

    new = {'rate': ra, 'left': nl, 'right': nr}
    return new


def remove_vocals(sound):
    rate = sound['rate']
    l = sound['left']
    r = sound['right']

    assert len(l) == len(r), 'Size doesn\' match'

    nl = list(map(lambda x, y: x-y, l, r))
    nr = nl.copy()

    new = {'rate': rate, 'left': nl, 'right': nr}
    return new

# below are helper functions for converting back-and-forth between WAV files
# and our internal dictionary representation for sounds

import io
import wave
import struct

def load_wav(filename):
    """
    Given the filename of a WAV file, load the data from that file and return a
    Python dictionary representing that sound
    """
    f = wave.open(filename, 'r')
    chan, bd, sr, count, _, _ = f.getparams()

    assert bd == 2, "only 16-bit WAV files are supported"

    left = []
    right = []
    for i in range(count):
        frame = f.readframes(1)
        if chan == 2:
            left.append(struct.unpack('<h', frame[:2])[0])
            right.append(struct.unpack('<h', frame[2:])[0])
        else:
            datum = struct.unpack('<h', frame)[0]
            left.append(datum)
            right.append(datum)

    left = [i/(2**15) for i in left]
    right = [i/(2**15) for i in right]

    return {'rate': sr, 'left': left, 'right': right}


def write_wav(sound, filename):
    """
    Given a dictionary representing a sound, and a filename, convert the given
    sound into WAV format and save it as a file with the given filename (which
    can then be opened by most audio players)
    """
    outfile = wave.open(filename, 'w')
    outfile.setparams((2, 2, sound['rate'], 0, 'NONE', 'not compressed'))

    out = []
    for l, r in zip(sound['left'], sound['right']):
        l = int(max(-1, min(1, l)) * (2**15-1))
        r = int(max(-1, min(1, r)) * (2**15-1))
        out.append(l)
        out.append(r)

    outfile.writeframes(b''.join(struct.pack('<h', frame) for frame in out))
    outfile.close()


if __name__ == '__main__':
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place to put your
    # code for generating and saving sounds, or any other code you write for
    # testing, etc.

    # here is an example of loading a file (note that this is specified as
    # sounds/hello.wav, rather than just as hello.wav, to account for the
    # sound files being in a different directory than this file)
    hello = load_wav('sounds/mystery.wav')
    s1 = load_wav('sounds/mystery.wav')
    s2 = load_wav('sounds/crash.wav')

    mix(s1, s2, 0.5)
    # write_wav(backwards(hello), 'hello_reversed.wav')
