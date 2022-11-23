from __future__ import print_function
from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
import pdb
import numpy as np
import torchfile
import shutil
from tqdm import tqdm
from tqdm import TqdmSynchronisationWarning
from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init
from shutil import copyfile
import glob
from torchvision import transforms
from tensorboardX import SummaryWriter
from miscc.datasets import FolderStoryDataset, FolderImageDataset
from fid.vfid_score import fid_score as vfid_score
from fid.fid_score_v import fid_score
from fid.utils import StoryDataset, IgnoreLabelDataset
from miscc.utils import inference_samples

from nltk.tokenize import RegexpTokenizer
class Infer(object):
    def __init__(self, output_dir, ratio, load_ckpt=None, save_img=True):
        self.load_ckpt = load_ckpt
        self.output_dir = output_dir
        self.log_dir = os.path.join(output_dir, 'log')
        self.model_dir = os.path.join(output_dir, 'Model')
        self.save_dir = "./Evaluation/{}".format(cfg.CONFIG_NAME)

        self.video_len = cfg.VIDEO_LEN
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.imbatch_size = cfg.TRAIN.IM_BATCH_SIZE * self.num_gpus
        self.stbatch_size = cfg.TRAIN.ST_BATCH_SIZE * self.num_gpus
        self.ratio = ratio
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True
        
        if not load_ckpt:
            # If not load ckpt, then do evaluation
            self._logger = SummaryWriter(self.log_dir)
        
        if cfg.TRAIN.FLAG and save_img:
            mkdir_p(self.save_dir)
    # ############# For training stageI GAN #############
    def load_network_stageI(self, output_dir, n_words, load_ckpt=None):
        import hashlib
        import importlib
        from model import RNN_ENCODER

        from model import Generator


        ########################
        text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.DIMENSION)
        if load_ckpt != None:
            path = cfg.TRAIN.TEXT_ENCODER #'./text_encoder120.pth'.format(self.model_dir)
            state_dict = torch.load(path,
                       map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from: ', path)

        else:
            import sys
            sys.exit("no path of text encoder!")
        for p in text_encoder.parameters():
            p.requires_grad = False

        netG = Generator(self.video_len)
        netG.apply(weights_init)

        if load_ckpt != None:
            path_G = os.path.join(self.model_dir, "netG_epoch_{}.pth".format(load_ckpt))
            state_dict = torch.load(path_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', path_G)

        if cfg.CUDA:
            netG.cuda()
            text_encoder.cuda()
        return netG, text_encoder

    def calculate_vfid(self, netG, epoch, testloader):
        netG.eval()
        with torch.no_grad():
            eval_modeldataset = StoryDataset(netG, len(testloader), testloader.dataset)
            vfid_value = vfid_score(IgnoreLabelDataset(testloader.dataset),
                eval_modeldataset, cuda=True, normalize=True, r_cache=None
            )
            fid_value = fid_score(IgnoreLabelDataset(testloader.dataset),
                    eval_modeldataset, cuda=True, normalize=True, r_cache=None
                )

        return fid_value, vfid_value

    def calculate_fvd(self, gen_path, epoch, num_of_video):
        from fvd.fvd import calculate_fvd_from_inference_result

        fvd_value = calculate_fvd_from_inference_result(gen_path, num_of_video=num_of_video)
        print('[{}] {}----------'.format(epoch, fvd_value))
        if self._logger:
            self._logger.add_scalar('Off_Evaluation/fvd',  fvd_value,  epoch)
        return fvd_value


    def eval_fid(self, test_loader):
        output_score_filename = os.path.join(self.save_dir, 'fid_score.csv')
        models = os.listdir(self.model_dir)
        with open(output_score_filename, 'a') as f:
            f.write('epoch,fid,vfid\n')
        for epoch in range(121):
            if 'netG_epoch_{}.pth'.format(epoch) in models:
                print('Evaluating epoch {}'.format(epoch))
                netG = self.load_network_stageI(self.output_dir, load_ckpt=epoch)
                fid, vfid = self.calculate_vfid(netG, epoch, test_loader)
                print('[{}] fid:{:.4f}, vfid:{:.4f}'.format(epoch, fid, vfid))
                with open(output_score_filename, 'a') as f:
                    f.write('{},{},{}\n'.format(epoch, fid, vfid))

    def eval_fvd(self, imageloader, storyloader, testloader, stage=1):
        output_score_filename = os.path.join(self.save_dir, 'fvd_score.csv')
        save_dir = os.path.join(self.save_dir, 'epoch') 
        models = os.listdir(self.model_dir)
        with open(output_score_filename, 'a') as f:
            f.write('epoch,fvd\n')
        for epoch in range(121):
            if 'netG_epoch_{}.pth'.format(epoch) in models:
                print('Evaluating epoch {}'.format(epoch))
                netG = self.load_network_stageI(self.output_dir, load_ckpt=epoch)
                inference_samples(netG, testloader, save_dir)
                fvd_value = self.calculate_fvd(save_dir, epoch=epoch, num_of_video=288)
                with open(output_score_filename, 'a') as f:
                    f.write('{},{}\n'.format(epoch, fvd_value))

    def inference(self, imageloader, storyloader, testloader, stage=1):
        netG = self.load_network_stageI(self.output_dir, load_ckpt=self.load_ckpt)
        inference_samples(netG, testloader, save_dir)

    def generate_story(self, netG, dataloader, text_encoder, wordtoix):
        from miscc.utils import images_to_numpy
        import PIL

        origin_img_path = os.path.join(self.save_dir, 'original')
        generated_img_path = os.path.join(self.save_dir, 'generate')
        sequence_img_path = os.path.join(self.save_dir, 'sequence')
        os.makedirs(origin_img_path, exist_ok=True)
        os.makedirs(generated_img_path, exist_ok=True)
        os.makedirs(sequence_img_path, exist_ok=True)

        print('Generating Test Samples...')
        save_images, save_labels = [], []
        story_id = 0
        for batch in tqdm(dataloader):
            real_cpu = batch['images']
            motion_input = batch['description'][:, :, :cfg.TEXT.DIMENSION]
            content_input = batch['description'][:, :, :cfg.TEXT.DIMENSION]
            catelabel = batch['labels']
            real_imgs = Variable(real_cpu)
            motion_input = Variable(motion_input)
            content_input = Variable(content_input)
            if cfg.CUDA:
                real_imgs = real_imgs.cuda()            
                motion_input = motion_input.cuda()
                content_input = content_input.cuda()
                catelabel = catelabel.cuda()

            st_texts = None
            if 'text' in batch:
                st_texts = batch['text']

            new_list = []
            for idx in range(cfg.TRAIN.ST_BATCH_SIZE):
                for j in range(cfg.VIDEO_LEN):
                    cur = st_texts[j]
                    new_list.append(cur[idx])

            captions = []
            cap_lens = []
            for cur_text in new_list:
                        
                if len(cur_text) == 0:
                    continue
                cur_text = cur_text.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(cur_text.lower())
                if len(tokens) == 0:
                    print('cur_text', cur_text)
                    continue

                rev = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0 and t in wordtoix:
                        rev.append(wordtoix[t])
                captions.append(rev)
                cap_lens.append(len(rev))

            max_len = np.max(cap_lens)
            sorted_indices = np.argsort(cap_lens)[::-1]

            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]

            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for pointer in range(len(captions)):
                idx_cap = sorted_indices[pointer]
                cap = captions[idx_cap]
                c_len = len(cap)
                cap_array[pointer, :c_len] = cap


            new_captions = cap_array
            batch_size = new_captions.shape[0]
            new_captions = Variable(torch.from_numpy(new_captions), volatile=True)
            cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

            new_captions = new_captions.cuda()
            cap_lens = cap_lens.cuda()

            batch_size = new_captions.size(0)
            hidden = text_encoder.init_hidden(batch_size)
            words_embs, sent_emb = text_encoder(new_captions, cap_lens, hidden)

            invert_sent_emb = Variable(torch.FloatTensor(sent_emb.size(0), sent_emb.size(1)).fill_(0)).cuda()
            invert_words_embs = Variable(torch.FloatTensor(words_embs.size(0), words_embs.size(1), words_embs.size(2)).fill_(0)).cuda()

            for pointer in range(len(sent_emb)):
                idx_cap = sorted_indices[pointer]
                cur_sent = sent_emb[pointer, :]
                cur_word = words_embs[pointer, :, :]
                invert_sent_emb[idx_cap, :] = cur_sent
                invert_words_embs[idx_cap, :, :] = cur_word
            invert_st_sent_emb = invert_sent_emb.view(-1, cfg.VIDEO_LEN, invert_sent_emb.size(1))

            motion_input = torch.cat((invert_st_sent_emb, catelabel), 2)
            content_input = invert_st_sent_emb
            fake_stories, _,_,_,_ = netG.sample_videos(motion_input, content_input, invert_words_embs)
            real_cpu = real_cpu.transpose(1, 2)
            fake_stories = fake_stories.transpose(1, 2)

            for (fake_story, real_story) in zip(fake_stories, real_cpu):
                origin_story_path = os.path.join(origin_img_path, str(story_id))
                os.makedirs(origin_story_path, exist_ok=True)
                generated_story_path = os.path.join(generated_img_path, str(story_id))
                os.makedirs(generated_story_path, exist_ok=True)

                story = torch.cat([fake_story.cpu(), real_story], dim=2)
                sequence = torch.cat([img for img in story], dim=2)
                sequence_img = images_to_numpy(sequence)
                sequence_img = PIL.Image.fromarray(sequence_img)
                sequence_img.save(os.path.join(sequence_img_path, str(story_id)+'.png'))

                for idx, (fake, real) in enumerate(zip(fake_story, real_story)):
                    fake_img = images_to_numpy(fake)
                    fake_img = PIL.Image.fromarray(fake_img)
                    fake_img.save(os.path.join(generated_story_path, str(idx)+'.png'))

                    real_img = images_to_numpy(real)
                    real_img = PIL.Image.fromarray(real_img)
                    real_img.save(os.path.join(origin_story_path, str(idx)+'.png'))
                
                story_id += 1

    def eval_fid2(self, testloader, video_transforms, image_transforms, storydataset):
        from fid.fid_score import fid_score 
        output_score_filename = os.path.join(self.save_dir, 'fid_score2.csv')
        with open(output_score_filename, 'a') as f:
            f.write('epoch,FID,FSD\n')

        models = os.listdir(self.model_dir)
        captions, ixtoword, wordtoix, n_words = storydataset.return_info()

        for epoch in range(241, 0, -1):
            if 'netG_epoch_{}.pth'.format(epoch) in models:
                if os.path.exists(os.path.join(self.save_dir, 'original')):
                    shutil.rmtree(os.path.join(self.save_dir, 'original'))
                if os.path.exists(os.path.join(self.save_dir, 'generate')):
                    shutil.rmtree(os.path.join(self.save_dir, 'generate'))

                print('Evaluating epoch {}'.format(epoch))
                netG, text_encoder = self.load_network_stageI(self.output_dir, n_words, load_ckpt=epoch)
                with torch.no_grad():
                    self.generate_story(netG, testloader, text_encoder, wordtoix)
                ref_dataloader = FolderStoryDataset(os.path.join(self.save_dir, 'original'), video_transforms)
                gen_dataloader = FolderStoryDataset(os.path.join(self.save_dir, 'generate'), video_transforms)
                vfid = vfid_score(ref_dataloader,
                    gen_dataloader, cuda=True, normalize=True, r_cache=None)
                ref_dataloader = FolderImageDataset(os.path.join(self.save_dir, 'original'), image_transforms)
                gen_dataloader = FolderImageDataset(os.path.join(self.save_dir, 'generate'), image_transforms)

                fid = fid_score(ref_dataloader,
                        gen_dataloader, cuda=True, normalize=True, r_cache=None)

                with open(output_score_filename, 'a') as f:
                    f.write('{},{},{}\n'.format(epoch, fid, vfid))
