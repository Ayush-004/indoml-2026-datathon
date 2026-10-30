"""Resolve the two documented archive expansion layouts without file search."""
from bisect import bisect_right
from pathlib import Path


class InputLayout:
    def __init__(self,root,contract,training,inference):
        self.root=Path(root).resolve()
        self.shards=[];cursor=0
        for entry in contract['archives']:
            if 'training_clips' not in entry:continue
            count=entry['training_clips'];stem=Path(entry['path']).stem
            prefix=self._choose_prefix(training[cursor]['audio_path'],stem)
            self.shards.append({'first_row':cursor,'end_row':cursor+count,'archive_stem':stem,'prefix':prefix})
            cursor+=count
        if cursor!=len(training) or len(self.shards)!=5:raise ValueError('Training archive population changed')
        organizer=[entry for entry in contract['archives'] if 'organizer_noisy_clips' in entry]
        if len(organizer)!=1 or organizer[0]['organizer_noisy_clips']!=len(inference):raise ValueError('Organizer archive population changed')
        self.organizer_stem=Path(organizer[0]['path']).stem
        self.organizer_prefix=self._choose_prefix(inference[0]['audio_path'],self.organizer_stem)
        self.ends=[item['end_row'] for item in self.shards]

    def _path(self,relative,prefix):
        relative=Path(relative)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Input path must stay relative')
        path=(self.root/prefix/relative).resolve()
        if self.root not in path.parents:raise ValueError('Resolved input escaped its dataset root')
        return path

    def _choose_prefix(self,relative,stem):
        choices=[prefix for prefix in ('',stem) if self._path(relative,prefix).is_file()]
        if len(choices)!=1:
            raise ValueError(f'Expected one archive-layout sentinel for {relative}; checked flat and {stem} prefixes, found {len(choices)}')
        return choices[0]

    def training_path(self,index,row):
        shard=bisect_right(self.ends,index)
        if not 0<=shard<len(self.shards):raise ValueError('Training row index is outside archive coverage')
        return self._path(row['audio_path'],self.shards[shard]['prefix'])

    def organizer_path(self,row):
        return self._path(row['audio_path'],self.organizer_prefix)

    def evidence(self):
        return {'training_archives':self.shards,'organizer_archive_stem':self.organizer_stem,
            'organizer_prefix':self.organizer_prefix,'layout_sentinels_passed':6,
            'mapping':'Explicit archive stems and manifest row ranges; no recursive filename matching or audio alteration'}
