
"""Provides data for training and testing."""
import os
import numpy as np
import PIL
import torch
import json
import torch.utils.data
import glob
import random


class Shoes(torch.utils.data.Dataset):
    def __init__(self, path, split='train', existed_npy=False, transform=None):
        super(Shoes, self).__init__()
        self.transform = transform
        self.path = path   # '/home/share/wenhaokun/shoes_data/
        self.readpath = 'relative_captions_shoes.json'
        self.existed_npy = existed_npy
        if split == 'train':
            textfile = 'train_im_names.txt'

        elif split == 'test':
            textfile = 'eval_im_names.txt'

        with open(os.path.join(self.path, self.readpath)) as handle:
            self.dictdump = json.loads(handle.read())
        
        text_file = open(os.path.join(self.path, textfile),'r')
        imgnames = text_file.readlines()
        imgnames = [imgname.strip('\n') for imgname in imgnames] # img list

        self.imgfolder = os.listdir(self.path)
        self.imgfolder = [self.imgfolder[i] for i in range(len(self.imgfolder)) if 'womens' in self.imgfolder[i]]

        ###########################
        if not self.existed_npy:
            self.imgimages_all = []
            for i in range(len(self.imgfolder)):
                path = os.path.join(self.path,self.imgfolder[i])
                imgfiles = [f for f in glob.glob(path + "/*/*.jpg", recursive=True)]
                self.imgimages_all += imgfiles
        else:
            self.imgimages_all = np.load(os.path.join(self.path, 'imgimages_all.npy'), allow_pickle=True).tolist()
            
        self.imgs = self.imgimages_all
        self.imgimages_raw = [os.path.basename(imgname) for imgname in self.imgimages_all]

        #############################
        if not self.existed_npy:
            self.relative_pairs = self.get_relative_pairs(self.dictdump, imgnames, self.imgimages_all, self.imgimages_raw)
        else:
            if split == 'train':
                self.relative_pairs = np.load(os.path.join(self.path, 'relative_pairs_train.npy'), allow_pickle=True).tolist()
            elif split == 'test':
                self.relative_pairs = np.load(os.path.join(self.path, 'relative_pairs_test.npy'), allow_pickle=True).tolist()

    def get_relative_pairs(self, dictdump, imgnames, imgimages_all, imgimages_raw):
        relative_pairs = []
        for i in range(len(imgnames)):
            ind = [k for k in range(len(dictdump))
                    if dictdump[k]['ImageName'] == imgnames[i]
                    or dictdump[k]['ReferenceImageName'] == imgnames[i]]
            for k in ind:
                # either belong to the target image ('ImageName')
                # or reference image ('ReferenceImageName')
                if imgnames[i] == dictdump[k]['ImageName']:
                    target_imagename = imgimages_all[imgimages_raw.index(
                        imgnames[i])]
                    source_imagename = imgimages_all[imgimages_raw.index(
                        dictdump[k]['ReferenceImageName'])]
                else:
                    source_imagename = imgimages_all[imgimages_raw.index(
                        imgnames[i])]
                    target_imagename = imgimages_all[imgimages_raw.index(
                        dictdump[k]['ImageName'])]
                text = dictdump[k]['RelativeCaption'].strip()
                relative_pairs.append({
                    'source': source_imagename,
                    'target': target_imagename,
                    'mod': text
                })
        return relative_pairs

    def __len__(self):
        return len(self.relative_pairs)
  
    
    def __getitem__(self, idx):

        caption = self.relative_pairs[idx]
        out = {}
        out['source_img_data'] = self.get_img(caption['source'])
        out['target_img_data'] = self.get_img(caption['target'])
        out['mod'] = {'str': caption['mod']}

        return out

    def get_img(self, img_path):
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def get_all_texts(self):
        #############################
        if not self.existed_npy:
            text_file = open(os.path.join(self.path, 'train_im_names.txt'),'r')
            imgnames = text_file.readlines()
            imgnames = [imgname.strip('\n') for imgname in imgnames] # img list
            train_relative_pairs = self.get_relative_pairs(self.dictdump, imgnames, self.imgimages_all, self.imgimages_raw)
            texts = []
            for caption in train_relative_pairs:
                mod_texts = caption['mod']
                texts.append(mod_texts)
        else:
            texts = np.load(os.path.join(self.path, 'all_texts.npy'), allow_pickle=True).tolist()
        return texts

    def get_test_queries(self):       # query
        test_queries = []
        for idx in range(len(self.relative_pairs)):
            caption = self.relative_pairs[idx]
            mod_str = caption['mod']
            candidate = caption['source']
            target = caption['target']

            out = {}
            out['source_img_id'] = self.imgimages_all.index(candidate)
            out['source_img_data'] = self.get_img(candidate)
            out['target_img_id'] = self.imgimages_all.index(target)
            out['target_img_data'] = self.get_img(target)
            out['mod'] = {'str':mod_str}
            test_queries.append(out)
        return test_queries

    def get_test_targets(self):     
        text_file = open(os.path.join(self.path, 'eval_im_names.txt'),'r')
        imgnames = text_file.readlines()
        imgnames = [imgname.strip('\n') for imgname in imgnames] # img list
        test_target = []
        for i in imgnames:
            out = {}
            out['target_img_id'] = self.imgimages_raw.index(i)
            out['target_img_data'] = self.get_img(self.imgimages_all[self.imgimages_raw.index(i)])
            test_target.append(out)
        return test_target


class FashionIQ(torch.utils.data.Dataset):
    def __init__(self, path, gallery_all=False, name = 'dress',split = 'train',transform=None):
        super(FashionIQ, self).__init__()

        self.path = path
        self.image_dir = self.path + 'resized_image'
        self.split_dir = self.path + 'image_splits'
        self.caption_dir = self.path + 'captions'
        self.name = name
        self.split = split
        self.transform = transform
        self.gallery_all = gallery_all

        self.test_targets = []
        self.test_queries = []

        with open(os.path.join(self.caption_dir, "cap.{}.{}.json".format(self.name, self.split)), 'r') as f:
            self.ref_captions = json.load(f)
        with open(os.path.join(self.split_dir, "split.{}.{}.json".format(self.name, self.split)), 'r') as f:
            self.images = json.load(f)
    
    def concat_text(self, captions):
        text = "<BOS> {} <AND> {} <EOS>".format(captions[0], captions[1])
        return text
    
    def __len__(self):
        return len(self.ref_captions)
        
    
    def __getitem__(self, idx):
        caption = self.ref_captions[idx]
        mod_str = self.concat_text(caption['captions'])
        candidate = caption['candidate']
        target = caption['target']

        out = {}
        out['source_img_data'] = self.get_img(candidate)
        out['target_img_data'] = self.get_img(target)
        out['mod'] = {'str': mod_str}

        return out

    def get_img(self,image_name):
        img_path = os.path.join(self.image_dir,self.name,image_name + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img

    def get_all_texts(self):
        texts = []
        with open(os.path.join(self.caption_dir, "cap.{}.{}.json".format(self.name, 'train')), 'r') as f:
            train_captions = json.load(f)
        for caption in train_captions:
            mod_texts = caption['captions']
            texts.append(mod_texts[0])
            texts.append(mod_texts[1])
        return texts

    def get_test_queries(self):       # query
        self.test_queries = []
        for idx in range(len(self.ref_captions)):
            caption = self.ref_captions[idx]
            mod_str = self.concat_text(caption['captions'])
            candidate = caption['candidate']
            target = caption['target']
            out = {}
            out['source_img_id'] = self.images.index(candidate)
            out['source_img_data'] = self.get_img(candidate)
            out['target_img_id'] = self.images.index(target)
            out['target_img_data'] = self.get_img(target)
            out['mod'] = {'str': mod_str}

            self.test_queries.append(out)
        
        return self.test_queries


    def get_test_targets(self):       # 所有的image
        if self.gallery_all:
            self.test_targets = []
            for idx in range(len(self.images)):
                target = self.images[idx]
                out = {}
                out['target_img_id'] = idx
                out['target_img_data'] = self.get_img(target)
                self.test_targets.append(out)
        else:
            test_targets_id = []
            queries = self.get_test_queries()
            for i in queries:
                if i['source_img_id'] not in test_targets_id:
                    test_targets_id.append(i['source_img_id'])
                if i['target_img_id'] not in test_targets_id:
                    test_targets_id.append(i['target_img_id'])
        
            self.test_targets = []
            for i in test_targets_id:
                out = {}
                out['target_img_id'] = i
                out['target_img_data'] = self.get_img(self.images[i])
                self.test_targets.append(out)   
        return self.test_targets


class Fashion200k(torch.utils.data.Dataset):
    """Fashion200k dataset."""

    def __init__(self, path, split='train', transform=None):
        super(Fashion200k, self).__init__()

        self.split = split
        self.transform = transform
        self.img_path = path + '/'

        # get label files for the split
        label_path = path + '/labels/'
        from os import listdir
        from os.path import isfile
        from os.path import join
        label_files = [
            f for f in listdir(label_path) if isfile(join(label_path, f))
        ]
        label_files = [f for f in label_files if split in f]

        # read image info from label files
        self.imgs = []
        self.test_queries = []

        def caption_post_process(s):
            return s.strip().replace('.',
                                     'dotmark').replace('?', 'questionmark').replace(
                                         '&', 'andmark').replace('*', 'starmark')

        for filename in label_files:
            print('read ', filename)
            with open(label_path + '/' + filename) as f:
                lines = f.readlines()
            for line in lines:
                line = line.split('	')
                img = {
                    'file_path': line[0],
                    'detection_score': line[1],
                    'captions': [caption_post_process(line[2])],
                    'split': split,
                    'modifiable': False
                }
                self.imgs += [img]
        print('Fashion200k:', len(self.imgs), 'images')

        # generate query for training or testing
        if split == 'train':
            self.caption_index_init_()
        else:
            self.generate_test_queries_()

    def get_loader(self, batch_size, shuffle=False, drop_last=False, num_workers=0):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda i: i)

    def get_test_queries(self):
        return self.test_queries

    def get_different_word(self, source_caption, target_caption):
        source_words = source_caption.split()
        target_words = target_caption.split()
        for source_word in source_words:
            if source_word not in target_words:
                break
        for target_word in target_words:
            if target_word not in source_words:
                break
        mod_str = 'replace ' + source_word + ' with ' + target_word
        return source_word, target_word, mod_str

    def generate_test_queries_(self):
        file2imgid = {}
        for i, img in enumerate(self.imgs):
            file2imgid[img['file_path']] = i
        with open(self.img_path + '/test_queries.txt') as f:
            lines = f.readlines()
        self.test_queries = []
        for line in lines:
            source_file, target_file = line.split()
            idx = file2imgid[source_file]
            target_idx = file2imgid[target_file]
            source_caption = self.imgs[idx]['captions'][0]
            target_caption = self.imgs[target_idx]['captions'][0]
            source_word, target_word, mod_str = self.get_different_word(
                source_caption, target_caption)
            self.test_queries += [{
                'source_img_id': idx,
                'source_caption': source_caption,
                'target_caption': target_caption,
                'mod': {
                    'str': mod_str
                }
            }]

    def caption_index_init_(self):
        """ index caption to generate training query-target example on the fly later"""

        # index caption 2 caption_id and caption 2 image_ids
        caption2id = {}
        id2caption = {}
        caption2imgids = {}
        for i, img in enumerate(self.imgs):
            for c in img['captions']:
                #if not caption2id.has_key(c):
                if c not in caption2id:
                    id2caption[len(caption2id)] = c
                    caption2id[c] = len(caption2id)
                    caption2imgids[c] = []
                caption2imgids[c].append(i)
        self.caption2imgids = caption2imgids
        print(len(caption2imgids), 'unique cations')

        # parent captions are 1-word shorter than their children
        parent2children_captions = {}
        for c in caption2id.keys():
            for w in c.split():
                p = c.replace(w, '')
                p = p.replace('  ', ' ').strip()
                #if not parent2children_captions.has_key(p):
                if p not in parent2children_captions:
                    parent2children_captions[p] = []
                if c not in parent2children_captions[p]:
                    parent2children_captions[p].append(c)
        self.parent2children_captions = parent2children_captions

        # identify parent captions for each image
        for img in self.imgs:
            img['modifiable'] = False
            img['parent_captions'] = []
        for p in parent2children_captions:
            if len(parent2children_captions[p]) >= 2:
                for c in parent2children_captions[p]:
                    for imgid in caption2imgids[c]:
                        self.imgs[imgid]['modifiable'] = True
                        self.imgs[imgid]['parent_captions'] += [p]
        num_modifiable_imgs = 0
        for img in self.imgs:
            if img['modifiable']:
                num_modifiable_imgs += 1
        print('Modifiable images', num_modifiable_imgs)

    def caption_index_sample_(self, idx):
        while not self.imgs[idx]['modifiable']:
            idx = np.random.randint(0, len(self.imgs))

        # find random target image (same parent)
        img = self.imgs[idx]
        while True:
            p = random.choice(img['parent_captions'])
            c = random.choice(self.parent2children_captions[p])
            if c not in img['captions']:
                break
        target_idx = random.choice(self.caption2imgids[c])

        # find the word difference between query and target (not in parent caption)
        source_caption = self.imgs[idx]['captions'][0]
        target_caption = self.imgs[target_idx]['captions'][0]
        source_word, target_word, mod_str = self.get_different_word(
            source_caption, target_caption)
        return idx, target_idx, source_word, target_word, mod_str

    def get_all_texts(self):
        texts = []
        for img in self.imgs:
            for c in img['captions']:
                texts.append(c)
        return texts

    def __len__(self):
        return len(self.imgs)
   
    def __getitem__(self, idx):
        idx, target_idx, source_word, target_word, mod_str = self.caption_index_sample_(
            idx)
        out = {}
        out['source_img_id'] = idx
        out['source_img_data'] = self.get_img(idx)
        out['source_caption'] = self.imgs[idx]['captions'][0]
        out['target_img_id'] = target_idx
        out['target_img_data'] = self.get_img(target_idx)
        out['target_caption'] = self.imgs[target_idx]['captions'][0]
        out['mod'] = {'str': mod_str}
        return out

    def get_img(self, idx, raw_img=False):
        img_path = self.img_path + self.imgs[idx]['file_path']
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        if raw_img:
            return img
        if self.transform:
            img = self.transform(img)
        return img




