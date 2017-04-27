import os
import pandas as pd
import yaml


class DatasetCreator:
    '''
    Class for creating yaml files for indexing common multitrack audio
    datasets.
    '''

    def __init__(self):

        template = """
            dataset: name of dataset
            base_path: path to the dataset
            songs:
              - artist: artist name
                title: song name
                style: style of the song
                mixture: relative path to the mixture
                stems:
                  stem: relative path to a stem
            """
        self.template = yaml.load(template)
        self.doc = self.template.copy()
        self.doc['songs'] = []

    def write_to_yaml(self, filename):
        filename, ext = os.path.splitext(filename)
        with open(filename + '.yml', 'w') as f:
            yaml.dump(self.doc, f, default_flow_style=False)

    def add_song(self, **kw):
        for key in self.template['songs'][0].keys():
            if key not in kw.keys():
                raise ValueError(""""Template specification not met
                                 (missing key: '{}')".format(key)""")
        self.doc['songs'].append(kw)

    def dsd100(self, base_path='/vol/vssp/datasets/audio/DSD100'):

        base_path = os.path.abspath(base_path)
        excel = pd.read_excel(os.path.join(base_path, 'dsd100.xlsx'), 'Sheet1')

        self.doc['dataset'] = 'DSD100'
        self.doc['base_path'] = base_path

        # Fix typo in xlsx file
        excel.ix[excel.Name == 'Patrick Talbot - Set Free Me', 'Name'] = (
            'Patrick Talbot - Set Me Free')

        # relative paths to each song
        mix_paths = ['Mixtures/Dev/' + _
                     for _ in sorted(os.listdir(
                         os.path.join(base_path, 'Mixtures/Dev')))]

        mix_paths += ['Mixtures/Test/' + _
                      for _ in sorted(os.listdir(
                          os.path.join(base_path, 'Mixtures/Test')))]

        source_paths = [_.replace('Mixtures', 'Sources') for _ in mix_paths]
        test_set = [0] * 50 + [1] * 50

        for row in excel.iterrows():

            artist, title = row[1]['Name'].split(' - ')
            style = row[1]['Style']
            idx = [i for i, _ in enumerate(mix_paths) if title in _][0]
            mixture = mix_paths[idx] + '/mixture.wav'
            print(mixture, test_set[idx])

            stems = {}
            for key in ['bass', 'drums', 'vocal', 'other']:
                stems[key] = '{0}/{1}.wav'.format(source_paths[idx], key)

            self.add_song(artist=artist,
                          title=title,
                          style=style,
                          mixture=mixture,
                          stems=stems,
                          test_set=test_set[idx])
