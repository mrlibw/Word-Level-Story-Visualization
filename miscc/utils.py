import os
import errno
import numpy as np
import PIL
from copy import deepcopy
from miscc.config import cfg
import pdb
from torch.nn import init
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
import random
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def contrastive_loss(features1, features2, labels,
              batch_size, eps=1e-8):
    if features1.dim() == 2:
        features1 = features1.unsqueeze(0)
        features2 = features2.unsqueeze(0)

    features1_norm = torch.norm(features1, 2, dim=2, keepdim=True)
    features2_norm = torch.norm(features2, 2, dim=2, keepdim=True)
    scores0 = torch.bmm(features1, features2.transpose(1, 2))
    norm0 = torch.bmm(features1_norm, features2_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.GAMMA

    scores0 = scores0.squeeze()
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1

def check_is_order(sequence):
    return (np.diff(sequence)>=0).all()

def create_random_shuffle(stories,  random_rate=0.5):
    o3n_data, labels = [], []
    device = stories.device
    stories = stories.cpu()
    story_size = len(stories)
    for idx, result in enumerate(stories):
        video_len = result.shape[1]
        label = 1 if random_rate > np.random.random() else 0
        if label == 0:
            o3n_data.append(result.clone())
        else:
            random_sequence = random.sample(range(video_len), video_len)
            while (check_is_order(random_sequence)): # make sure not sorted
                np.random.shuffle(random_sequence)
            shuffled_story = result[:, list(random_sequence), :, :].clone()
            story_size_idx = random.randint(0, story_size-1)
            if story_size_idx != idx:
                story_mix = random.sample(range(video_len), 1)
                shuffled_story[:, story_mix, :, : ] = stories[story_size_idx, :, story_mix, :, :].clone()
            o3n_data.append(shuffled_story)
        labels.append(label)

    order_labels = Variable(torch.from_numpy(np.array(labels)).float(), requires_grad=True).detach()
    shuffle_imgs = Variable(torch.stack(o3n_data, 0), requires_grad=True)
    return shuffle_imgs.to(device), order_labels.to(device)

def compute_discriminator_loss(netD, real_imgs, fake_imgs,
                               real_labels, fake_labels,
                               conditions, gpus, new_word_emb, st_idx):
    criterion = nn.BCELoss()
    cate_criterion =nn.MultiLabelSoftMarginLoss()
    batch_size = real_imgs.size(0)
    fake = fake_imgs.detach()

    cond = conditions.detach() 
    real_features = netD(real_imgs, new_word_emb, st_idx) 
    fake_features = netD(fake, new_word_emb, st_idx)

    real_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (real_features), gpus)
    fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)
    uncond_errD_real = criterion(real_logits, real_labels)
    uncond_errD_fake = criterion(fake_logits, fake_labels)
    errD = uncond_errD_real + uncond_errD_fake 
    errD_real = uncond_errD_real 
    errD_fake = uncond_errD_fake 

    return errD, errD_real.data, errD_fake.data

def compute_generator_loss(netD, fake_imgs, real_imgs, real_labels, 
                        conditions, gpus, match_labels, new_word_emb, st_idx):
    criterion = nn.BCELoss()
    cate_criterion =nn.MultiLabelSoftMarginLoss()
    batch_size = cfg.TRAIN.IM_BATCH_SIZE

    cond = conditions.detach()
    fake_features = nn.parallel.data_parallel(netD, (fake_imgs, new_word_emb, st_idx), gpus)
    real_features = nn.parallel.data_parallel(netD, (real_imgs, new_word_emb, st_idx), gpus)
    fake_logits = \
            nn.parallel.data_parallel(netD.get_uncond_logits,
                                      (fake_features), gpus)

    uncond_errD_fake = criterion(fake_logits, real_labels)
    errD_fake = uncond_errD_fake

    # additional losses between real and fake images.
    fake_features = fake_features.view(fake_features.size(0),-1)
    real_features = real_features.view(real_features.size(0),-1)
    contra_loss0, contra_loss1 = contrastive_loss(fake_features, real_features,
                                match_labels, batch_size)
    contra_loss = (contra_loss0 + contra_loss1) * \
                cfg.TRAIN.LAMBDA

    errD_fake += contra_loss


    return errD_fake

def KL_loss(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

#############################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


#############################
def save_img_results(data_img, fake, texts, epoch, image_dir):
    num = cfg.VIS_COUNT
    fake = fake[0:num]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, '%s/real_samples_epoch_%03d.png' % 
            (image_dir, epoch), normalize=True)
        vutils.save_image(
            fake.data, '%s/fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)
    else:
        vutils.save_image(
            fake.data, '%s/lr_fake_samples_epoch_%03d.png' %
            (image_dir, epoch), normalize=True)

    if texts is not None:
        fid = open('%s/lr_fake_samples_epoch_%03d.txt' % (image_dir, epoch), 'wb')
        for i in range(num):
            fid.write(str(i) + ':' + texts[i] + '\n')
        fid.close()

##########################\
def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(1,2,0)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')

def save_story_results(ground_truth, images, texts, name, image_dir, step=0, lr = False):
    video_len = cfg.VIDEO_LEN
    all_images = []
    for i in range(images.shape[0]):
        all_images.append(vutils.make_grid(torch.transpose(images[i], 0,1), video_len))
    all_images= vutils.make_grid(all_images, 1)
    all_images = images_to_numpy(all_images)
    
    if ground_truth is not None:
        gts = []
        for i in range(ground_truth.shape[0]):
            gts.append(vutils.make_grid(torch.transpose(ground_truth[i], 0,1), video_len))
        gts = vutils.make_grid(gts, 1)
        gts = images_to_numpy(gts)
        all_images = np.concatenate([all_images, gts], axis = 1)


    output = PIL.Image.fromarray(all_images)
    if lr:
        output.save('{}/lr_samples_{}_{}.png'.format(image_dir, name, step))
    else:
        output.save('{}/fake_samples_{}_{}.png'.format(image_dir, name, step))

    if texts is not None:
        fid = open('{}/fake_samples_{}.txt'.format(image_dir, name), 'w')
        for idx in range(images.shape[0]):
            fid.write(str(idx) + '--------------------------------------------------------\n')
            for i in range(len(texts)):
                fid.write(texts[i][idx] +'\n' )
            fid.write('\n\n')
        fid.close()
    return all_images

def save_image_results(ground_truth, images, size=cfg.IMSIZE):
    video_len = cfg.VIDEO_LEN
    st_bs = cfg.TRAIN.ST_BATCH_SIZE
    images = images.reshape(st_bs, video_len, -1, size, size)
    all_images = []
    for i in range(images.shape[0]):
        all_images.append(vutils.make_grid(images[i], video_len))
    all_images= vutils.make_grid(all_images, 1)
    all_images = images_to_numpy(all_images)
    
    if ground_truth is not None:
        ground_truth = ground_truth.reshape(st_bs, video_len, -1, size, size)
        gts = []
        for i in range(ground_truth.shape[0]):
            gts.append(vutils.make_grid(ground_truth[i], video_len))
        gts = vutils.make_grid(gts, 1)
        gts = images_to_numpy(gts)
        all_images = np.concatenate([all_images, gts], axis = 1)
    return all_images

def save_all_img(images, count, image_dir):
    bs, size_c, v_len, size_w, size_h = images.shape
    for b in range(bs):
        imgs = images[b].transpose(0,1)
        for i in range(v_len):
            count += 1
            png_name = os.path.join(image_dir, "{}.png".format(count))
            vutils.save_image(imgs[i], png_name)
    return count

def get_multi_acc(predict, real):
    predict = 1/(1+np.exp(-predict))
    correct = 0
    for i in range(predict.shape[0]):
        for j in range(predict.shape[1]):
            if real[i][j] == 1 and predict[i][j]>=0.5 :
                correct += 1
    acc = correct / float(np.sum(real))
    return acc

def save_model(netG, netD_st, epoch, model_dir, whole=False):
    if whole == True:
        torch.save(netG, '%s/netG.pkl' % (model_dir))
        torch.save(netD_st, '%s/netD_st.pkl' % (model_dir))
        print('Save G/D model')
        return
    torch.save(netG.state_dict(),'%s/netG_epoch_%d.pth' % (model_dir, epoch))
    torch.save(netD_st.state_dict(),'%s/netD_st_epoch_last.pth' % (model_dir))
    print('Save G/D models ')

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)

def save_test_samples(netG, dataloader, save_path):
    print('Generating Test Samples...')
    save_images = []
    save_labels = []
    for i, batch in enumerate(dataloader, 0):
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
        motion_input = torch.cat((motion_input, catelabel), 2)
        fake, _,_,_,_ = netG.sample_videos(motion_input, content_input)
        save_story_results(real_cpu, fake, batch['text'], '{:03d}'.format(i), save_path)
        save_images.append(fake.cpu().data.numpy())
        save_labels.append(catelabel.cpu().data.numpy())
    save_images = np.concatenate(save_images, 0)
    save_labels = np.concatenate(save_labels, 0)
    np.save(save_path + '/images.npy', save_images)
    np.save(save_path + '/labels.npy', save_labels)

def save_train_samples(netG, dataloader, save_path):
    print('Generating Train Samples...')
    save_images = []
    save_labels = []
    for i, batch in enumerate(dataloader, 0):
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
        motion_input = torch.cat((motion_input, catelabel), 2)
        fake, _,_,_,_ = netG.sample_videos(motion_input, content_input)
        save_story_results(real_cpu, fake, batch['text'], '{:05d}'.format(i), save_path)
        save_images.append(fake.cpu().data.numpy())
        save_labels.append(catelabel.cpu().data.numpy())
    save_images = np.concatenate(save_images, 0)
    save_labels = np.concatenate(save_labels, 0)
    np.save(save_path + '/images.npy', save_images)
    np.save(save_path + '/labels.npy', save_labels)


def inference_samples(netG, dataloader, save_path, text_encoder, wordtoix):
    print('Generate and save images...')

    mkdir_p(save_path)
    mkdir_p('./Evaluation/ref')
    cnt_gen = 0
    cnt_ref = 0
    for i, batch in enumerate(tqdm(dataloader, desc='Saving')):
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

        fake, _,_,_,_ = netG.sample_videos(motion_input, content_input, invert_words_embs)
        cnt_gen = save_all_img(fake, cnt_gen, save_path)
        cnt_ref = save_all_img(real_imgs, cnt_ref, './Evaluation/ref')

  
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count 


if __name__ == "__main__":
    test = torch.randn((14, 3, 5, 64,64))
    output, labels = create_random_shuffle(test)
    print(output.shape)
