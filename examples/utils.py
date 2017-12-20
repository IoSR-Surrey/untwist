from untwist import data
from collections import namedtuple
import os


Stems = namedtuple('Stems', 'drums bass other vocals accompaniment mixture')


def get_stems(song_idx=0,
              path_to_dsd100_subset='/scratch/DSD100subset',
              mono=False):
    '''
    Load drums, bass, 'other', vocals, accompaniment and the mixture from four
    songs of the Demixing Secret Database 100 (DSD100).

    song_idx: int (0, 1, 2, or 3)
              index of the song to load
    path_to_dsd100_subset: string
              path to https://www.loria.fr/~aliutkus/DSD100subset.zip
    mono: bool
          Converts wavs to mono if true.
    '''

    if song_idx == 0:
        path = "Sources/Test/005 - Angela Thomas Wade - Milk Cow Blues"
    elif song_idx == 1:
        path = "Sources/Test/049 - Young Griffo - Facade"
    elif song_idx == 2:
        path = "Sources/Dev/055 - Angels In Amplifiers - I'm Alright"
    elif song_idx == 3:
        path = "Sources/Dev/081 - Patrick Talbot - Set Me Free"

    path = os.path.join(path_to_dsd100_subset,
                        path)

    if mono:
        def _func(x):
            return x.to_mono()
    else:
        def _func(x):
            return x

    stems = [_func(data.audio.Wave.read(os.path.join(path, _ + '.wav')))
             for _ in ['drums', 'bass', 'other', 'vocals']
             ]

    stems = Stems(
        drums=stems[0],
        bass=stems[1],
        other=stems[2],
        vocals=stems[3],
        accompaniment=sum(stems[:3]),
        mixture=sum(stems),
    )

    return stems
