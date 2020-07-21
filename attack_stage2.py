## attack.py -- the last stage for generating audio adversarial examples
##
## 

import argparse
import errno
import json
import os

import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from tqdm import tqdm
from warpctc_pytorch import CTCLoss

from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
from data.data_loader import load_audio
from decoder import GreedyDecoder
from model import WaveToLetter
import Levenshtein as Lev
import matplotlib.pyplot as plt

import torch.optim as optim
import torch.nn.functional as F
import torchaudio

from data import pytorch_mfcc
import scipy.signal

parser = argparse.ArgumentParser(description='Wav2Letter attacks')
parser.add_argument('--max-iterations', default=10000, type=int, help='maximum number of iteration for attacks')
parser.add_argument('--target', default=None, type=str, help='target sentence for attacks')
parser.add_argument('--audio-path', default=None, type=str, help='input audio path for attacks')#0.8 3029reload
parser.add_argument('--orig-path', default=None, type=str, help='original audio path for attacks')
parser.add_argument('--target-path', default=None, type=str, help='original txt path for attacks')
parser.add_argument('--model-path', default='./model_libri/pretrained_wav2Letter.pth.tar',
                    help='Path to model file created by training')
parser.add_argument('--cuda', default=True, action="store_true", help='Use cuda to test model')
parser.add_argument('--lr', '--learning-rate to attack', default=8e-5, type=float, help='initial learning rate')
parser.add_argument('--learning-anneal', default=5, type=float, help="Annealing applied to learning rate every epoch")
parser.add_argument('--decoder', default="beam", choices=["greedy", "beam", "none"], type=str, help="Decoder to use")
parser.add_argument('--bandwidth', default=0.20, type=float, help='Bandwidth for noise allowed')
# audio_conf
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')#.02
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')#.01
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.9, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.1,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.7,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--printSilence', default=False, help='No print except setted to True')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--mixPrec',dest='mixPrec',default=False,action='store_true', help='use mix precision for training')
# decode_conf
parser.add_argument('--top-paths', default=1, type=int, help='number of beams to return')
parser.add_argument('--beam-width', default=300, type=int, help='Beam width to use')
parser.add_argument('--lm-path', default="./4-gram.arpa", type=str,
                       help='Path to an (optional) kenlm language model for use with beam search (req\'d with trie)')
parser.add_argument('--alpha', default=0.75, type=float, help='Language model weight')
parser.add_argument('--beta', default=1.0, type=float, help='Language model word bonus (all words)')
parser.add_argument('--cutoff-top-n', default=40, type=int,
                       help='Cutoff number in pruning, only top cutoff_top_n characters with highest probs in '
                            'vocabulary will be used in beam search, default 40.')
parser.add_argument('--cutoff-prob', default=1.0, type=float,
                       help='Cutoff probability in pruning,default 1.0, no pruning.')
parser.add_argument('--lm-workers', default=2, type=int, help='Number of LM processes to use')
args = parser.parse_args()

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}
class Attacker(object):
    def __init__(self, audio_path, orig_path, audio_conf, labels, criterion=CTCLoss()):
        super(Attacker, self).__init__()
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.noiseInjector = NoiseInjection(audio_conf['noise_dir'],
                                            audio_conf['noise_levels']) if audio_conf.get(
            'noise_dir') is not None else None
        self.noise_prob = audio_conf.get('noise_prob')

        self.audio_path = audio_path
        self.orig_path = orig_path
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.criterion = criterion

    def get_signal(self, path):
        signal = load_audio(path)
        signal = Variable(signal, requires_grad=True)
        return signal

    def normalize(self, mfccs):
        mean = mfccs.mean()
        std = mfccs.std()
        mfccs = torch.add(mfccs, -mean)
        mfccs = mfccs / std 
        return mfccs   
    
    def mfccs_to_inputs(self, mfccs):   
        inputs = torch.zeros(1, mfccs.shape[0], mfccs.shape[1])
        inputs = inputs.cuda()
        inputs[0] = mfccs
        inputs = inputs.transpose(1, 2)
        return inputs

    def save(self, signal, target_path, orig_path, lr, iterations, i, bandwidth):
        outprefix = "generated_stage2"
        if not os.path.exists(outprefix):
            os.makedirs(outprefix)
        signal = signal.unsqueeze(0)
        id_original = self.audio_path.split("/")[-1].split('_')[0]
        id_target = target_path.split("/")[-1].split('.')[0]
        save_path = os.path.join(outprefix, id_original + '_to_' + id_target + "_final" + ".wav")
        torchaudio.save(save_path, signal, sample_rate=16000, precision=32)
        print("Adversarial audio has been saved to " + save_path + "!")
        return save_path

    def attention(self, signal):
        f = open("./data/"+self.audio_path.split("/")[-1].split('.')[0]+".txt", 'w')
        for i in range(signal.shape[0]):
            f.write(str(signal[i]))
            f.write("\n")
        index_max = int(torch.max(signal, 0)[1])
        index_min = int(torch.min(signal, 0)[1])
        return index_max, index_min

    def save_figure(self, audio, save_path):
        n = np.arange(len(audio))
        plt.plot(n, audio.data.numpy())
        folder_figure = save_path[:-4]
        if not os.path.exists(folder_figure):
            os.makedirs(folder_figure)
        plt.savefig(os.path.join(folder_figure, 'audio.png'))

        audio = audio.data.numpy() / np.max(audio.data.numpy())#标准化
        length_signal = len(audio)
        half_length = np.ceil((length_signal + 1) / 2.0).astype(np.int)
        signal_frequency = np.fft.fft(audio)
        signal_frequency = abs(signal_frequency[0:half_length]) / length_signal
        len_fts = len(signal_frequency)
        x_axis = np.arange(0, len_fts, 1) * (16000 / length_signal) / 1000.0

        plt.figure()
        plt.plot(x_axis, signal_frequency, color='blue')
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Signal frequency (abs)')
        plt.savefig(os.path.join(folder_figure, 'frequency.png'))
    
    def attack(self, iterations, target_path, lr, bandwidth):
        flag = 0
        model = WaveToLetter.load_model(args.model_path)
        model = model.to(device)
        model.eval()#eval. This is different from stage1
        signal = self.get_signal(self.audio_path)
        orig = self.get_signal(self.orig_path)

        index_max, index_min= self.attention(signal)
        start_attack_time = time.time()
        for i in range(iterations):
            print('Iteration:', str(i))
            mfcc = pytorch_mfcc.MFCC(samplerate=self.sample_rate,winlen=self.window_size,winstep=self.window_stride,numcep=13,nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0,ceplifter=22,appendEnergy=False).cuda()
            mfccs = mfcc(signal)
            mfccs = self.normalize(mfccs)

            if args.printSilence:
                print("mfccs", mfccs)
            inputsMags =  self.mfccs_to_inputs(mfccs)
            out = model(inputsMags)

            path = self.orig_path.split('wav')[0] + 'txt' + self.orig_path.split('wav')[1]
            fp=open(path)
            transcriptReal=fp.readlines()[0]
            print("Ref:",transcriptReal.lower())

            seq_length = out.size(1)
            sizes = Variable(torch.Tensor([1.0]).mul_(int(seq_length)).int(), requires_grad=False) 

            if args.printSilence: 
                print("out",out)
                print("softmax",F.softmax(out, dim=-1).data) 
            
            decoded_output, _, = decoder.decode(F.softmax(out, dim=-1).data, sizes)
            transcript = decoded_output[0][0]      
            print("Hyp:", transcript.lower())

            out = out.transpose(0, 1)
            if args.target:
                transcriptTarget = args.target
            else:
                fp=open(target_path)
                transcriptTarget=fp.readlines()[0]
            print("Tar:", transcriptTarget.lower())

            if transcript.lower()==transcriptTarget.lower() and i>0:
                if args.target:
                    target_path = args.target
                save_path = self.save(signal, target_path, self.orig_path, lr, iterations, i, bandwidth)
                generate_time = time.time() - start_attack_time
                print('Time taken (s): {generate_time:.4f}\t'.format(generate_time=generate_time))
                self.save_figure(signal, save_path)
                break
                
            target = list(filter(None, [self.labels_map.get(x) for x in list(transcriptTarget.upper())]))   
            targets = torch.IntTensor(target) 
            target_sizes = torch.IntTensor([len(target)])
            ctcloss = self.criterion(out, targets, sizes, target_sizes)
            # print("ctcloss:", ctcloss)
            # print("delta_2:", 100*torch.sum((signal - orig)**2))
            # loss = ctcloss + 100*torch.sum((signal - orig)**2)
            loss = ctcloss
            print("loss:", loss)

            loss.backward()
 
            grad = np.array(signal.grad)
            is_nan = np.isnan(grad)
            is_nan_new = is_nan[is_nan == True]
            for j in range(len(grad)):
                if is_nan[j]:
                    grad[j] = 10

            wer = decoder.wer(transcript.lower(), transcriptTarget.lower()) / float(len(transcriptTarget.lower().split()))

            # the iterative proportional clipping method  
            perturbation = lr*torch.from_numpy(grad)
            signal_next_relative = torch.clamp((signal.data - perturbation)/orig, min = 1 - bandwidth, max = 1 + bandwidth)
            signal.data = signal_next_relative.mul(orig)

            # if (i + 1) % 15000 == 0 and flag < 1:
            #     # anneal lr
            #     # lr *= 0.5
            #     lr = lr / args.learning_anneal
            #     flag += 1
            # print("wer", wer)
            # print("lr", lr)
            print("\n")
            signal.grad.data.zero_()
        print("Come to the end")
    
if __name__ == '__main__':

    # Set seeds for determinism
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)

    audio_conf = dict(sample_rate=args.sample_rate,
                        window_size=args.window_size,
                        window_stride=args.window_stride,
                        window=args.window,
                        noise_dir=args.noise_dir,
                        noise_prob=args.noise_prob,
                        noise_levels=(args.noise_min, args.noise_max))
    device = torch.device("cuda" if args.cuda else "cpu")

    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))
    if args.decoder == "beam":
        from decoder import BeamCTCDecoder
        decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                    cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                    beam_width=args.beam_width, num_processes=args.lm_workers)
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    else:
        decoder = None

    load_file_path = './generated_stage1'
    single_file = args.audio_path.split('/')[-1]

    transcript_target = single_file.split('_')[2]
    transcript_origin = single_file.split('_')[0]

    args.audio_path = os.path.join(load_file_path, single_file)
    args.orig_path = os.path.join('./data/original',transcript_origin+'.wav')
    args.target_path = os.path.join('./data/target',transcript_target+'.txt').strip()
    print(args.audio_path)
    print(args.orig_path)
    print(args.target_path)
    
    attacker = Attacker(audio_path=args.audio_path, orig_path=args.orig_path, audio_conf=audio_conf, labels=labels)
    attacker.attack(iterations=args.max_iterations, target_path=args.target_path, lr=args.lr, bandwidth=args.bandwidth)
