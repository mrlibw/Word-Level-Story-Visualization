from __future__ import print_function
from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import time
import pdb
import numpy as np
import torchfile

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init, count_param
from miscc.utils import save_story_results, save_model, save_test_samples, save_image_results
from miscc.utils import KL_loss
from miscc.utils import compute_discriminator_loss, compute_generator_loss
from shutil import copyfile

from fid.vfid_score import fid_score as vfid_score
from fid.fid_score_v import fid_score
from fid.utils import StoryDataset, IgnoreLabelDataset

from torchvision import transforms
from tensorboardX import SummaryWriter
from inception import InceptionV3

from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
class GANTrainer(object):
    def __init__(self, output_dir, args, ratio=1.0):
        if cfg.TRAIN.FLAG:
            output_dir = "{}/".format(output_dir)
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'log')
            self.test_dir = os.path.join(output_dir, 'Test')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            mkdir_p(self.test_dir)
            if not os.path.exists(os.path.join(self.model_dir, 'model.py')):
                copyfile(args.cfg_file, output_dir + 'setting.yml')
                if cfg.CASCADE_MODEL:
                    copyfile('./cascade_model.py', output_dir + 'model.py')
                else:
                    copyfile('./model.py', output_dir + 'model.py')
                copyfile('./trainer.py', output_dir + 'trainer.py')

        self.video_len = cfg.VIDEO_LEN
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.imbatch_size = cfg.TRAIN.IM_BATCH_SIZE * self.num_gpus
        self.stbatch_size = cfg.TRAIN.ST_BATCH_SIZE * self.num_gpus
        self.ratio = ratio
        self.con_ckpt = args.continue_ckpt
        self.fid_eval = False
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True
        self.inception_dim = 2048

        self._logger = SummaryWriter(self.log_dir)

    def load_network_stageI(self, n_words):
        if cfg.CASCADE_MODEL:
            from cascade_model import Generator, STAGE1_D_STY_V2        
        else:
            from model import Generator, STAGE1_D_STY_V2, RNN_ENCODER, CNN_ENCODER
        netG = Generator(self.video_len)
        netG.apply(weights_init)
        netD_st = STAGE1_D_STY_V2()
        netD_st.apply(weights_init)
        

        netG_param_cnt, netD_st_param = count_param(netG), count_param(netD_st)
        total_params = netG_param_cnt + netD_st_param
        

        print('The total parameter is : {}M, netG:{}M, netD_st:{}M'.format(total_params//1e6, netG_param_cnt//1e6,
            netD_st_param//1e6))

        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        if self.con_ckpt:
            print('Continue training from epoch {}'.format(self.con_ckpt))
            path = '{}/netG_epoch_{}.pth'.format(self.model_dir, self.con_ckpt)
            netG.load_state_dict(torch.load(path))            
            path = '{}/netD_st_epoch_last.pth'.format(self.model_dir)
            netD_st.load_state_dict(torch.load(path))
            
        text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.DIMENSION)
        if cfg.TRAIN.TEXT_ENCODER != '':
            path = cfg.TRAIN.TEXT_ENCODER 
            state_dict = torch.load(path,
                       map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from: ', path)

        text_encoder.eval()

        if cfg.CUDA:
            netG.cuda()
            netD_st.cuda()
            text_encoder = text_encoder.cuda()

        return netG, netD_st, text_encoder


    def sample_real_image_batch(self):
        if self.imagedataset is None:
            self.imagedataset = enumerate(self.imageloader)
        batch_idx, batch = next(self.imagedataset)
        b = batch
        if cfg.CUDA:
            for k, v in batch.items():
                if k == 'text' or k == 'full_text':
                    continue
                else:
                    b[k] = v.cuda()

        if batch_idx == len(self.imageloader) - 1:
            self.imagedataset = enumerate(self.imageloader)
        return b


    def train(self, imageloader, storyloader, testloader, storydataset, stage=1):
        c_time = time.time()
        self.imageloader = imageloader
        self.imagedataset = None

        captions, ixtoword, wordtoix, n_words = storydataset.return_info()
        netG, netD_st, text_encoder = self.load_network_stageI(n_words)
        start = time.time()
        im_real_labels = Variable(torch.FloatTensor(self.imbatch_size).fill_(1))
        im_fake_labels = Variable(torch.FloatTensor(self.imbatch_size).fill_(0))
        st_real_labels = Variable(torch.FloatTensor(self.imbatch_size).fill_(1))
        st_fake_labels = Variable(torch.FloatTensor(self.imbatch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(self.imbatch_size)))
        if cfg.CUDA:
            im_real_labels, im_fake_labels = im_real_labels.cuda(), im_fake_labels.cuda()
            st_real_labels, st_fake_labels = st_real_labels.cuda(), st_fake_labels.cuda()
            match_labels = match_labels.cuda()

        image_weight = cfg.IMAGE_RATIO
        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH

        st_optimizerD = optim.Adam(netD_st.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para, lr=cfg.TRAIN.GENERATOR_LR, betas=(0.5, 0.999))

        mse_loss = nn.MSELoss()

        scheduler_stD = ReduceLROnPlateau(st_optimizerD, 'min', verbose=True, factor=0.5, min_lr=1e-7, patience=0)
        scheduler_G = ReduceLROnPlateau(optimizerG, 'min', verbose=True, factor=0.5, min_lr=1e-7, patience=0)
        count = 0

        if not self.con_ckpt:
            start_epoch = 0
        else:
            start_epoch = int(self.con_ckpt)

        non_word_emb = nn.Embedding(1, cfg.TEXT.DIMENSION)
        idx = torch.LongTensor([0])
        non_word_emb = non_word_emb(idx).unsqueeze(2).cuda()


        print('LR DECAY EPOCH: {}'.format(lr_decay_step))

        for epoch in range(start_epoch, self.max_epoch):
            l = self.ratio * (2. / (1. + np.exp(-10. * epoch)) - 1)
            start_t = time.time()
            num_step = len(storyloader)
            stats = {}

            with tqdm(total=len(storyloader), dynamic_ncols=True) as pbar:
                for i, data in enumerate(storyloader):
                    ######################################################
                    # (1) Prepare training data
                    ######################################################
                    im_batch = self.sample_real_image_batch()
                    st_batch = data
                    im_real_cpu = im_batch['images']
                    im_real_imgs = Variable(im_real_cpu)

                    if cfg.DATASET_NAME == "pororo":
                        im_labels = Variable(im_batch['labels'])
                    else:
                        im_labels = None


                    st_real_cpu = st_batch['images']

                    st_texts = None
                    if 'text' in st_batch:
                        st_texts = st_batch['text']

                    st_list = []
                    for idx in range(cfg.TRAIN.ST_BATCH_SIZE):
                        for j in range(cfg.VIDEO_LEN):
                            cur = st_texts[j]
                            st_list.append(cur[idx])

                    captions = []
                    cap_lens = []


                    for cur_text in st_list:
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
                
                    org_st_cap_lens = cap_lens


                    cap_lens = cap_lens[sorted_indices]
                    cap_array = np.zeros((len(captions), max_len), dtype='int64')
                    for pointer in range(len(captions)):
                        idx_cap = sorted_indices[pointer]
                        cap = captions[idx_cap]
                        c_len = len(cap)
                        cap_array[pointer, :c_len] = cap
  

                    st_captions = cap_array
                    batch_size = st_captions.shape[0]
                    st_captions = Variable(torch.from_numpy(st_captions), volatile=True)
                    st_cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)
                    org_st_cap_lens = Variable(torch.from_numpy(org_st_cap_lens), volatile=True)
                    

                    st_captions = st_captions.cuda()
                    st_cap_lens = st_cap_lens.cuda()
                    org_st_cap_lens = org_st_cap_lens.cuda()

                    batch_size = st_captions.size(0)
                    hidden = text_encoder.init_hidden(batch_size)
                    st_words_embs, st_sent_emb = text_encoder(st_captions, st_cap_lens, hidden)
                    st_words_embs = st_words_embs.detach()
                    st_sent_emb =st_sent_emb.detach()
                    invert_st_sent_emb = Variable(torch.FloatTensor(st_sent_emb.size(0), st_sent_emb.size(1)).fill_(0)).cuda()
                    invert_st_words_embs = Variable(torch.FloatTensor(st_words_embs.size(0), st_words_embs.size(1), st_words_embs.size(2)).fill_(0)).cuda()

                    for pointer in range(len(st_sent_emb)):
                        idx_cap = sorted_indices[pointer]
                        cur_sent = st_sent_emb[pointer, :]
                        cur_word = st_words_embs[pointer, :, :]
                        invert_st_sent_emb[idx_cap, :] = cur_sent
                        invert_st_words_embs[idx_cap, :, :] = cur_word
                    new_invert_st_sent_emb = invert_st_sent_emb.view(-1, cfg.VIDEO_LEN, invert_st_sent_emb.size(1))

                    st_real_imgs = Variable(st_real_cpu)

                    if cfg.DATASET_NAME == "pororo":
                        st_labels = Variable(st_batch['labels']) 
                    else:
                        st_labels = None


                    im_texts = None
                    if 'text' in im_batch:
                        im_texts = im_batch['text']

                    im_full_texts = None
                    if 'full_text' in im_batch:
                        im_full_texts = im_batch['full_text']

                    
                    full_invert_sent_emb = Variable(torch.FloatTensor(cfg.TRAIN.IM_BATCH_SIZE, cfg.VIDEO_LEN, cfg.TEXT.DIMENSION).fill_(0)).cuda()
                    req_invert_words_embs = None
                    req_invert_sent_emb = None
                    org_im_cap_lens = None
                    text_pointer = 0
                    for text_idx in im_full_texts:
                        captions = []
                        cap_lens = []
                        for cur_text in text_idx:

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

                        if text_pointer == 0:
                            org_im_cap_lens = cap_lens
                            org_im_cap_lens = Variable(torch.from_numpy(org_im_cap_lens), volatile=True)
                            org_im_cap_lens = org_im_cap_lens.cuda()


                        sorted_cap_lens = cap_lens[sorted_indices]
                        cap_array = np.zeros((len(captions), max_len), dtype='int64')

                        for pointer in range(len(captions)):
                            idx_cap = sorted_indices[pointer]
                            cap = captions[idx_cap]
                            c_len = len(cap)
                            cap_array[pointer, :c_len] = cap
                       
                        new_captions = cap_array
                        batch_size = new_captions.shape[0]
                        new_captions = Variable(torch.from_numpy(new_captions), volatile=True)
                        sorted_cap_lens = Variable(torch.from_numpy(sorted_cap_lens), volatile=True)
                        cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                        new_captions = new_captions.cuda()
                        sorted_cap_lens = sorted_cap_lens.cuda()
                        cap_lens = cap_lens.cuda()

                        batch_size = new_captions.size(0)
                        hidden = text_encoder.init_hidden(batch_size)
                        words_embs, sent_emb = text_encoder(new_captions, sorted_cap_lens, hidden)
                        words_embs = words_embs.detach()
                        sent_emb = sent_emb.detach()

                        invert_sent_emb = Variable(torch.FloatTensor(sent_emb.size(0), sent_emb.size(1)).fill_(0)).cuda()
                        invert_words_embs = Variable(torch.FloatTensor(words_embs.size(0), words_embs.size(1), words_embs.size(2)).fill_(0)).cuda()
                        for pointer in range(len(sent_emb)):
                            idx_cap = sorted_indices[pointer]
                            cur_sent = sent_emb[pointer, :]
                            cur_word = words_embs[pointer, :, :]
                            invert_sent_emb[idx_cap, :] = cur_sent
                            invert_words_embs[idx_cap, :, :] = cur_word

                        full_invert_sent_emb[:,text_pointer,:] = invert_sent_emb
                        if text_pointer == 0:
                            req_invert_words_embs = invert_words_embs
                            req_invert_sent_emb = invert_sent_emb
                        text_pointer += 1


                    if cfg.CUDA:
                        st_real_imgs = st_real_imgs.cuda() 
                        im_real_imgs = im_real_imgs.cuda()                                 
                        

                    if im_labels is not None:  
                        im_labels = im_labels.cuda()
                        st_labels = st_labels.cuda()                    
                        im_motion_input = torch.cat((full_invert_sent_emb[:,0,:], im_labels), 1) 
                        st_motion_input = torch.cat((new_invert_st_sent_emb, st_labels), 2) 
                    else:                    
                        im_motion_input = full_invert_sent_emb[:,0,:]
                        st_motion_input = new_invert_st_sent_emb

                    im_content_input = full_invert_sent_emb
                    st_content_input = new_invert_st_sent_emb


                    #######################################################
                    # (2) Generate fake stories and images
                    ######################################################
                    with torch.no_grad():
                        st_fake, m_mu, m_logvar, c_mu, c_logvar = \
                            netG.sample_videos(st_motion_input, st_content_input, invert_st_words_embs) 

                        im_fake, im_mu, im_logvar, cim_mu, cim_logvar = \
                            netG.sample_images(im_motion_input, im_content_input, req_invert_words_embs) 

                    if im_labels is not None:          
                        characters_mu = (st_labels.mean(1)>0).type(torch.FloatTensor).cuda() 
                        st_mu = torch.cat((c_mu, st_motion_input[:,:, :cfg.TEXT.DIMENSION].mean(1).squeeze(), characters_mu), 1)
                    else:
                        st_mu = torch.cat((c_mu, st_motion_input[:,:, :cfg.TEXT.DIMENSION].mean(1).squeeze()), 1)
                    im_mu = torch.cat((im_motion_input, cim_mu), 1)

                    ############################
                    # (3) Update D network
                    ###########################
                    netD_st.zero_grad()
                    se_accD = 0


                    new_word_emb = torch.ones(invert_st_words_embs.size(0), invert_st_words_embs.size(1), 40).cuda()
                    new_word_emb = new_word_emb * non_word_emb
                    if invert_st_words_embs.size(2) <= 40:
                        new_word_emb[:,:,:invert_st_words_embs.size(2)] = invert_st_words_embs
                    else:
                        ix = list(np.arange(invert_st_words_embs.size(2)))  
                        np.random.shuffle(ix)
                        ix = ix[:40]
                        ix = np.sort(ix)
                        new_word_emb = invert_st_words_embs[:,:,ix]
                   

                    new_word_emb_img = torch.ones(req_invert_words_embs.size(0), req_invert_words_embs.size(1), 40).cuda()
                    new_word_emb_img = new_word_emb_img * non_word_emb

                    if req_invert_words_embs.size(2) <= 40:
                        new_word_emb_img[:,:,:req_invert_words_embs.size(2)] = req_invert_words_embs
                    else:
                        ix = list(np.arange(req_invert_words_embs.size(2)))  
                        np.random.shuffle(ix)
                        ix = ix[:40]
                        ix = np.sort(ix)
                        new_word_emb_img = req_invert_words_embs[:,:,ix]

                    
                    im_errD, im_errD_real, im_errD_fake = \
                        compute_discriminator_loss(netD_st, im_real_imgs, im_fake,
                                               im_real_labels, im_fake_labels,
                                               im_mu, self.gpus, new_word_emb_img, 0)




                    st_errD, st_errD_real, st_errD_fake = \
                        compute_discriminator_loss(netD_st, st_real_imgs, st_fake,
                                               st_real_labels, st_fake_labels,
                                               st_mu, self.gpus, new_word_emb, 1)


                    errD = im_errD + st_errD
                    errD.backward(retain_graph=True)
                    st_optimizerD.step()
                    step = i+num_step*epoch

                    ############################
                    # (2) Update G network
                    ###########################
                    netG.zero_grad()

                    st_fake, m_mu, m_logvar, c_mu, c_logvar = netG.sample_videos(st_motion_input, st_content_input, invert_st_words_embs)
                    im_fake, im_mu, im_logvar, cim_mu, cim_logvar = netG.sample_images(im_motion_input, im_content_input,
                        req_invert_words_embs)
                    encoder_decoder_loss = 0
                    
                    if im_labels is not None:
                        characters_mu = (st_labels.mean(1)>0).type(torch.FloatTensor).cuda()
                        st_mu = torch.cat((c_mu, st_motion_input[:,:, :cfg.TEXT.DIMENSION].mean(1).squeeze(), characters_mu), 1)
                    else:
                        st_mu = torch.cat((c_mu, st_motion_input[:,:, :cfg.TEXT.DIMENSION].mean(1).squeeze()), 1)
                    im_mu = torch.cat((im_motion_input, cim_mu), 1)
                    
                    im_errG = compute_generator_loss(netD_st, im_fake, im_real_imgs,
                                                im_real_labels, im_mu, self.gpus, 
                                                match_labels, new_word_emb_img, 0)

                    st_errG = compute_generator_loss(netD_st, st_fake, st_real_imgs,
                                                st_real_labels, st_mu, self.gpus, 
                                                match_labels, new_word_emb, 1)

                    im_kl_loss = KL_loss(cim_mu, cim_logvar)
                    st_kl_loss = KL_loss(c_mu, c_logvar)

                    errG_total = im_errG + im_kl_loss * cfg.TRAIN.COEFF.KL \
                                    + self.ratio * (st_errG * image_weight + st_kl_loss * cfg.TRAIN.COEFF.KL)

                    errG_total.backward(retain_graph=True)
                    optimizerG.step()



                    count = count + 1
                    pbar.update(1)

                    if i % 20 == 0:
                        step = i+num_step*epoch
                        for key, value in stats.items():
                            self._logger.add_scalar(key, value, step)
            print('''[%d/%d]
                    im_errG: %.2f st_errD: %.2f im_errG: %.2f st_errG: %.2f'''
                    % (epoch, self.max_epoch, im_errG, st_errD,
                    im_errG, st_errG))
            with torch.no_grad():
                fake, _,_,_,_ = netG.sample_videos(st_motion_input, st_content_input, invert_st_words_embs)
                st_result = save_story_results(st_real_cpu, fake, st_texts, epoch, self.image_dir, i)

            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in st_optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr
                lr_decay_step *= 2

            g_lr, im_lr, st_lr = 0, 0, 0
            for param_group in optimizerG.param_groups:
                g_lr = param_group['lr']
            for param_group in st_optimizerD.param_groups:
                st_lr = param_group['lr']

            if cfg.EVALUATE_FID_SCORE:
                self.calculate_vfid(netG, epoch, testloader)

            time_mins = int((time.time() - c_time)/60)
            time_hours = int(time_mins / 60)
            epoch_mins = int((time.time()-start_t)/60)
            epoch_hours = int(epoch_mins / 60)

            print("----[{}/{}]Epoch time:{} hours {} mins, Total time:{} hours----".format(epoch, self.max_epoch, epoch_hours, epoch_mins, time_hours))

            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD_st, epoch, self.model_dir)
        save_model(netG, netD_st, self.max_epoch, self.model_dir)
