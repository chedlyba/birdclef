import os
from tqdm import tqdm
import librosa
import pandas as pd
import json
import warnings
import torchaudio
import math, random
import torch
from torchaudio import transforms
from datetime import datetime
warnings.filterwarnings('ignore')

SOUND_PATH = '/datadrive/datasets/birdclef/train_soundscapes'
SOUND_METADATA = '/datadrive/datasets/birdclef/train_soundscape_labels.csv'
SOUND_JSON_PATH = '/datadrive/datasets/birdclef/soundscape/data'

BIRD_PATH = '/datadrive/datasets/birdclef/train_short_audio'
BIRD_METADATA = '/datadrive/datasets/birdclef/train_metadata.csv'
BIRD_JSON_PATH = '/datadrive/datasets/birdclef/bird/data_m'


class AudioUtil:

    def __init__(self, sr=44100, n_fft=1024, hop_len=512, n_mels=64, top_db=80):
        
        self.melspectrogram = transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
        self.amplitude_to_db = transforms.AmplitudeToDB(top_db=top_db)
        self.spectrogram = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_len)
    
    @staticmethod
    def open(audio_file):
            sig, sr = torchaudio.load(audio_file)
            return sig,sr
        
    @staticmethod
    def rechannel(aud, sr, new_channel):
        sig = aud
        
        if (sig.shape[0] == new_channel):
            return aud
        
        if (new_channel == 1):
            resig = sig[:1,:]
        else:
            resig = torch.cat([sig,sig])
            
        return ((resig))
    
    @staticmethod
    def resample(aud, sr, newsr):
        sig = aud
        
        if (sr == newsr):
            return aud
        
        num_channels = sig.shape[0]
        
        resig = torchaudio.transforms.Resample(sr,newsr)(sig[:1,:])
        if (num_channels>1):
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
            resig = torch.cat([resig,retwo])
            
        return (resig)
    
    @staticmethod
    def pad_trunc(aud, sr, max_ms):
        sig = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000*max_ms
        
        if (sig_len > max_len):
            sig = sig[:,:max_len]
            
        elif (sig_len < max_len):
            pad_begin_len = random.randint(0,max_len - sig_len)
            pad_end_len = max_len - sig_len -pad_begin_len
            
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows,pad_end_len))
            
            sig = torch.cat((pad_begin, sig, pad_end), 1)
            
        return (sig)
    
    @staticmethod
    def time_shift(aud, shift_limit):
        sig = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt))
    
    
    def spectro_gram(self, aud):
        sig = aud
        spec = self.melspectrogram(sig)
        spec = self.amplitude_to_db(spec)
        
        return (spec)
    
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec
        
        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
            
        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
            
        return aug_spec
    
    def preprocess(self, aud, sr=44100, channel=1, max_ms=5000, newsr=44100):
        
        reaud = self.resample(aud, sr, newsr)
        rechan = self.rechannel(reaud, newsr, channel)
        
        return rechan
    
    
    def augment(self, aud, sr=44100, channel=1, max_ms=5000, shift_pct=0.2, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1, newsr=44100):
        
        shift_aud = self.time_shift(aud, shift_pct)
        sgram = self.spectro_gram(shift_aud)
        #aug_sgram = self.spectro_augment(sgram, max_mask_pct=max_mask_pct, n_freq_masks=n_freq_masks, n_time_masks=n_time_masks)
        return aug_sgram

def prepare_soundscape_dataset(path, meta_path, json_path, n_fft=1024, hop_length=512, n_mels=64, top_db=80, max_iter=100):
    audio_util = AudioUtil(sr=32000, n_fft=n_fft, hop_len=hop_length, n_mels=n_mels, top_db=top_db)
    train_sound_meta = pd.read_csv(meta_path)
    data = {
        'id' : [],
        'site' : [],
        'birds' : [],
        'spectrogram' : [],
        'seconds' : []
    }
    file_count = 0
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path)):
        if dirpath is not dir:
            j = 0
            for file in tqdm(filenames):
                j+=1
                id, site = file.split('_')[0], file.split('_')[1]
                file = '/'.join([dirpath, file])
                aud, sr = audio_util.open(file)
                sec = 5
                l = sec * sr
                
                while l <= aud.shape[1] :
                    audio_segment = aud[:, l - (5 * sr):l]
                    
                    row_id = '_'.join([str(id), str(site), str(sec)])
                    birds = train_sound_meta.loc[train_sound_meta['row_id'] == row_id]['birds'].values[0]
                    audio = audio_util.preprocess(audio_segment, sr=sr, newsr=32000)
                    if birds == ['nocall']:
                        j+=1
                        data['id'].append(id)
                        data['site'].append(site)
                        spectrogram = audio_util.spectro_gram(audio, n_mels=n_mels, n_fft=n_fft, hop_len=hop_len).T.tolist()
                        print(f'Processed: {str(id)}_{site}_{sec} : {birds} ')
                    else:
                        for k in range(max_iter):
                            j=j+1
                            spectrogram = audio_util.augment(aud=audio, sr=sr, newsr=32000).T.tolist()
                            data['id'].append(f'{str(id)}.{k}')
                            data['site'].append(site)
                            data['birds'].append(birds)
                            data['spectrogram'].append(spectrogram)
                            data['seconds'].append(sec)
                            print(f'Processed {j}: {str(id)}.{k}_{site}_{sec} : {birds} ')

                            if (j % 32 == 0) & (j != 0):
                                with open(f'{json_path}_{j/32}.json', 'w') as fp:
                                    json.dump(data, fp, indent=4) 
                                    data = {
                                        'id' : [],
                                        'site' : [],
                                        'birds' : [],
                                        'spectrogram' : [],
                                        'seconds' : []
                                    }
                                    fp.close()
                                    file_count+=1
                    
                    sec = sec + 5
                    l = sec * sr
                if j % 32 == 0 & j != 0:
                    with open(f'{json_path}_{j/32}.json', 'w') as fp:
                        json.dump(data, fp, indent=4) 
                        data = {
                            'id' : [],
                            'site' : [],
                            'birds' : [],
                            'spectrogram' : [],
                            'seconds' : []
                        }
                        fp.close()
                        file_count+=1
               
    print(f'{j} items processed.')
    with open(f'{json_path}_final', 'w') as fp:
        json.dump(data, fp, indent=4) 
        fp.close()
        file_count+=1
    print(f'{file_count} files created.')

def prepare_bird_dataset(path, meta_path, json_path, n_fft=1024, hop_length=512, n_mels=64, top_db=80, newsr=32000, max_iter=100):
    audio_util = AudioUtil(sr=newsr, n_fft=n_fft, hop_len=hop_length, n_mels=n_mels, top_db=top_db)
    train_bird_meta = pd.read_csv(meta_path)
    data = {
        'latitude' : [],
        'longitude' : [],
        'bird' : [],
        'spectrogram' : [],
        'seconds' : []
    }
    file_count = 0
    j = 0
    for i, (dirpath, dirnames, filenames) in enumerate((os.walk(path))):
        if dirpath is not dir:
            curr = ''
            count = 0
            for file in (filenames):
                start = datetime.now()
                bird = dirpath.split('/')[-1]
                if bird == curr:
                    count+=1
                else:
                    curr = bird
                if count < 50:
                    latitude = train_bird_meta.loc[train_bird_meta['filename'] == file]['latitude'].values[0]
                    longitude = train_bird_meta.loc[train_bird_meta['filename'] == file]['longitude'].values[0]
                    file = os.path.join(dirpath, file)
                    aud, sr = audio_util.open(file)
                    sec = 5
                    l = sec * newsr
                    print( f"Preprocessed {bird}-{file.split('/')[-1]} in {datetime.now()-start} sec" )
                    audio = audio_util.preprocess(aud, sr=sr, newsr=newsr)
                    length = audio.shape[1]
                    audio = audio_util.pad_trunc(audio, sr=newsr, max_ms=30000)
                    length = newsr * 30

                    while l < length:
                        j+=1
                        audio_segment = audio[:, l - (5 * sr):l]
                        
                        start = datetime.now()
                        spec = audio_util.spectro_gram(aud=audio_segment).T.tolist()
                        data['spectrogram'].append(spec)
                        data['seconds'].append(sec)
                        data['bird'].append(bird)
                        data['longitude'].append(longitude)
                        data['latitude'].append(latitude)
                        print(f'Processed {j} : {bird} in {datetime.now()-start} sec')
                        sec +=5
                        l = sec * newsr
                        if (j % 32 == 0) and (j != 0):
                            start = datetime.now()
                            with open(f'{json_path}_{j/32}.json', 'w') as fp:
                                json.dump(data, fp, indent=4) 
                                data = {
                                    'latitude' : [],
                                    'longitude' : [],
                                    'bird' : [],
                                    'spectrogram' : [],
                                    'seconds' : []
                                }
                                fp.close()
                            file_count+=1 
                            print(f'Wrote data_{j/32}.json in {datetime.now()-start}')
                else:
                    break        

    print(f'{j} items processed.')
    with open(f'{json_path}_final', 'w') as fp:
        json.dump(data, fp, indent=4) 
        fp.close()
        file_count+=1
    print(f'{file_count} files created.')
    
                     
if __name__ == '__main__':

    start = datetime.now()
    #prepare_soundscape_dataset(SOUND_PATH, SOUND_METADATA, SOUND_JSON_PATH, max_iter=10)
    prepare_bird_dataset(BIRD_PATH, BIRD_METADATA, BIRD_JSON_PATH, max_iter=10)
    print(f'Processe ended in {(datetime.now()-start)}')