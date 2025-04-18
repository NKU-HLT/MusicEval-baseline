"""
Largely adapt codes from:
https://github.com/nii-yamagishilab/mos-finetune-ssl
"""
import os
import argparse
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import laion_clap
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModel
from tqdm import tqdm
import random
random.seed(1984)
from utils import *
    
class MosPredictor(nn.Module):
    def __init__(self, up_model, up_out_dim):
        super(MosPredictor, self).__init__()
        self.upstream_model = up_model
        self.upstream_feat_dim = up_out_dim
        self.overall_mlp_layer1 = nn.Linear(in_features = self.upstream_feat_dim, out_features = 256)
        self.overall_mlp_layer2 = nn.Linear(in_features = 256, out_features = 64)
        self.overall_mlp_layer3 = nn.Linear(in_features = 64, out_features = 1)
        self.textual_mlp_layer1 = nn.Linear(in_features = self.upstream_feat_dim*2, out_features = 256)
        self.textual_mlp_layer2 = nn.Linear(in_features = 256, out_features = 64)
        self.textual_mlp_layer3 = nn.Linear(in_features = 64, out_features = 1)


    def forward(self, wavs, texts):
        wav_embed = self.upstream_model.get_audio_embedding_from_data(wavs, use_tensor = True).to(device)
        text_embed = self.upstream_model.get_text_embedding(texts,  use_tensor = True).to(device)

        combine_embed=torch.cat((wav_embed,text_embed),dim=1) # bs*1024        
        
        hidden1 = self.overall_mlp_layer1(wav_embed)
        hidden1_2 = self.overall_mlp_layer2(hidden1)
        out1 = self.overall_mlp_layer3(hidden1_2)
        hidden2 = self.textual_mlp_layer1(combine_embed)
        hidden2_2 = self.textual_mlp_layer2(hidden2)
        out2 = self.textual_mlp_layer3(hidden2_2)
        return out1, out2
    
class MyDataset(Dataset):
    def __init__(self, wavdir, mos_list):
        self.mos_overall_lookup = { }
        self.mos_coherence_lookup = { }
        f = open(mos_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]  # 'audiomos2025-track1-S002_P044.wav'
            mos_overall = float(parts[1])
            mos_coherence = float(parts[2])
            self.mos_overall_lookup[wavname] = mos_overall
            self.mos_coherence_lookup[wavname] = mos_coherence

        self.wavdir = wavdir
        self.wavnames = sorted(self.mos_overall_lookup.keys())

        
    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        if wav.size(1) > 480000:    # 16khz*30s
            wav = wav[:,:480000]
        overall_score = self.mos_overall_lookup[wavname]
        coherence_score = self.mos_coherence_lookup[wavname]
        return wav, overall_score, coherence_score, wavname
    

    def __len__(self):
        return len(self.wavnames)


    def collate_fn(self, batch):
        wavs, overall_scores, coherence_scores, wavnames = zip(*batch)
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)
        output_wavs = torch.stack(output_wavs, dim=0)
        overall_scores  = torch.stack([torch.tensor(x) for x in list(overall_scores)], dim=0)
        coherence_scores  = torch.stack([torch.tensor(x) for x in list(coherence_scores)], dim=0)
        
        return output_wavs, overall_scores, coherence_scores, wavnames


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: ' + str(device))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default="../data/MusicEval-phase1", required=False, help='Path of musiceval dataset')
    parser.add_argument('--expname', type=str, required=False, default='exp1', help='ckpt will be saved in track1_ckpt/EXPNAME')
    args = parser.parse_args()

    DATA_DIR = args.datadir
    UPSTREAM_MODEL = 'CLAP-music'
    EXP_NAME = args.expname
    CKPT_DIR = '../track1_ckpt/' + EXP_NAME # checkpoint will be saved here
   
    if not os.path.exists(CKPT_DIR):
        os.system('mkdir -p ' + CKPT_DIR)    

    wavdir = os.path.join(DATA_DIR, 'wav')
    trainlist = os.path.join(DATA_DIR, 'sets/train_mos_list.txt')
    validlist = os.path.join(DATA_DIR, 'sets/dev_mos_list.txt')

    if UPSTREAM_MODEL == 'CLAP-music':
        UPSTREAM_OUT_DIM= 512 
        model = laion_clap.CLAP_Module(enable_fusion=False,  amodel= 'HTSAT-base')
        model.load_ckpt('../upstream/music_audioset_epoch_15_esc_90.14.pt')
    else:
        print('*** ERROR *** Model type ' + UPSTREAM_MODEL + ' not supported.')
        exit()

    trainset = MyDataset(wavdir, trainlist)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)

    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=8, shuffle=True, num_workers=2, collate_fn=validset.collate_fn)

    net = MosPredictor(model, UPSTREAM_OUT_DIM).to(device)
    
    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), lr=5e-4, momentum=0.9)    

    PREV_VAL_LOSS = 9999999999
    orig_patience=20
    patience=orig_patience
    BEST_EPOCH = 0
    BEST_PATH = os.path.join(CKPT_DIR, 'best_ckpt')

    for epoch in range(1,1001):
        STEPS=0
        net.train()
        train_epoch_loss = 0.0
        train_epoch_loss1 = 0.0
        train_epoch_loss2 = 0.0
        for i, data in enumerate(tqdm(trainloader, desc="Training Progress", ncols=100), 0):
            STEPS += 1
            wavs, labels1, labels2, filenames = data  
            wavs = wavs.squeeze(1)  # tensor(batch,T)
            texts=get_texts_from_filename(filenames)    # list
        
            labels1 = labels1.unsqueeze(1).to(device)
            labels2 = labels2.unsqueeze(1).to(device)

            optimizer.zero_grad()
            output1,output2 = net(wavs,texts)
            loss1 = criterion(output1, labels1)
            loss2 = criterion(output2, labels2)
            train_loss = (loss1+loss2) / 2
            loss1.backward(retain_graph=True)
            loss2.backward()

            optimizer.step() 

            train_epoch_loss += train_loss.item()
            train_epoch_loss1 += loss1.item()
            train_epoch_loss2 += loss2.item()
        print('EPOCH:' + str(epoch) + ', AVG EPOCH TRAIN LOSS: ' + str(train_epoch_loss / STEPS))
        
        # clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        # validation
        VALSTEPS=0
        net.eval()
        valid_epoch_loss = 0.0
        valid_epoch_loss1 = 0.0
        valid_epoch_loss2 = 0.0
        for i, data in enumerate(tqdm(validloader, desc="Validating Progress", ncols=100), 0):
            VALSTEPS+=1
            wavs, labels1, labels2, filenames = data
            wavs = wavs.squeeze(1)
            texts=get_texts_from_filename(filenames)

            labels1 = labels1.unsqueeze(1).to(device)
            labels2 = labels2.unsqueeze(1).to(device)
            with torch.no_grad():
                output1,output2 = net(wavs, texts)

            loss1 = criterion(output1, labels1)
            loss2 = criterion(output2, labels2)
            valid_loss = (loss1+loss2) / 2
            
            valid_epoch_loss1 += loss1.item()
            valid_epoch_loss2 += loss2.item()
            valid_epoch_loss += valid_loss.item()
        avg_val_loss=valid_epoch_loss / VALSTEPS    
        print('EPOCH VAL LOSS: ' + str(avg_val_loss))
        
        if avg_val_loss < PREV_VAL_LOSS:    # Loss has decreased
            torch.save(net.state_dict(), BEST_PATH)
            BEST_EPOCH = epoch
            PREV_VAL_LOSS = avg_val_loss
            patience = orig_patience
        else:
            patience-=1
            if patience == 0:
                print('loss has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                break
    os.rename(BEST_PATH, os.path.join(CKPT_DIR, 'best_ckpt_'+str(BEST_EPOCH)))
    print('Finished Training, best epoch:', BEST_EPOCH)

if __name__ == '__main__':
    main()
