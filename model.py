import torch 
import numpy as np 
import torch.nn as nn 
import torchvision
import text_model as text_model_
import torch.nn.functional as F
import sys
from torch.nn.parameter import Parameter
import resnet
import math
from blocks import *
from collections import defaultdict
import timm

def get_correction(self, src_experts, trg_experts):
    # src : {B x D}
    # trg : {N x D}
    # return : {B x N x D}
    B = src_experts[self.modalities[0]].size(0)
    N = trg_experts[self.modalities[0]].size(0)
    diff_feature = defaultdict(list)
    for bi in range(B):
        # {mod: N x D}
        new_src_expert = {mod: src_experts[mod][bi].unsqueeze(0).expand(N,-1) 
                            for mod in self.modalities}
        if self.correction_type in ['duda', 'diffpool','tirgpool','concat','fdpool','fuspool']:
            diff_feature[self.modalities[0]].append(
                self.norm_layer['text'](
                    self.correction_layer(new_src_expert, trg_experts)
                )
            )
        else:
            # {mod: N x D}
            for mod, layer in zip(self.modalities, self.correction_layer):
                diff_feature[mod].append(
                    self.norm_layer['text'](layer(new_src_expert[mod], trg_experts[mod]))
                )
    for mod in self.modalities:
        diff_feature[mod] = torch.stack(diff_feature[mod], 0) # B x N x D
        if self.correction_type in ['duda', 'diffpool','tirgpool','concat','fdpool','fuspool']:
            break
    return diff_feature
    

class FusDiff(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.fc1 = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim*3, embed_dim),
            nn.ReLU()
        )
        
    def forward(self, x_before, x_after):
        #x_before = self.cg1(x_before)
        #x_after = self.cg2(x_after)
        
        x_before_ = self.fc1(torch.cat((x_before * x_after, x_before), -1)) # B x D
        x_after_ = self.fc2(torch.cat((x_before * x_after, x_after), -1)) # B x D
        x_diff = x_after_ - x_before_
        
        x = torch.cat((x_before, x_diff, x_after), -1) # B x 3*D
        x = self.fc(x)
        
        return x
        

class local_conv(nn.Module):
    def __init__(self, img_channel=2048, text_dim=1024, T=7.0):
        super().__init__()
        self.img1x1conv = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)  
        self.T = T
        #self.T = Parameter(torch.FloatTensor([T]))
        self.inception_gamma = Inception2d(in_dim=2048)
        self.inception_beta = Inception2d(in_dim=2048)
        
    def forward(self, img_tensor, text_embed):         #  text_embed   N,L,1024
        img_embed = self.img1x1conv(img_tensor)
        n,c,h,w = img_embed.size()
        img_embed = img_embed.view(n,c,h*w)
        # print(text_embed.size())
        # print(img_embed.size())   # N,L,49
        dot_product = torch.bmm(text_embed, img_embed)
        atten = self.softmax(dot_product / self.T)
        sentence_cat = []
        for i in range(img_tensor.size(2)*img_tensor.size(3)):
            sentence = torch.sum(text_embed * atten[:,:,i].unsqueeze(-1), dim=1)  # N,1024
            sentence_cat.append(sentence)
        sentence_cat = torch.stack(sentence_cat).permute(1, 2, 0).contiguous()  # N,1024,49
 
        x = torch.cat([img_embed, sentence_cat], dim=1).view(n,-1,h,w)  # N,2048,7,7
        gamma = self.inception_gamma(x)
        beta = self.inception_beta(x)
        return gamma * img_tensor + beta, atten


class global_conv(nn.Module):
    def __init__(self, img_channel=2048, text_dim=1024, T=4.0):
        super().__init__()
        self.img1x1conv = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.T = T
        #self.T = Parameter(torch.FloatTensor([T]))
        self.inception_gamma = Inception1d(in_dim=2048)
        self.inception_beta = Inception1d(in_dim=2048)

    def forward(self, img_tensor, text_embed):    # text_embed N,L,1024

        img_embed = self.img1x1conv(img_tensor)   # N,1024,7,7
        n,c,h,w = img_embed.size()
        img_embed = img_embed.view(n,c,h*w)
        dot_product = torch.bmm(text_embed, img_embed)
        attn = self.softmax(dot_product / self.T)
        img_cat = []
        for i in range(text_embed.shape[1]):
            img = torch.sum(img_embed * attn[:,i,:].unsqueeze(1), dim=-1) # N,1024
            img_cat.append(img)
        img_cat = torch.stack(img_cat).permute(1, 2, 0).contiguous()   # N,1024,L
        
        x = torch.cat([text_embed.permute(0, 2, 1).contiguous(), img_cat], dim=1) # N,2048,L

        gamma = self.inception_gamma(x)
        beta = self.inception_beta(x)
        return gamma * img_cat + beta, attn


class img_backbone(nn.Module):
    def __init__(self, dim=1024, dropout_p=0.2):
        super().__init__()
        # self.img_model = resnet.resnet50(pretrained=True)
        self.img_model = timm.create_model('swin_large_patch4_window7_224_in22k', pretrained=True, num_classes=0)
        self.img_pool = GeM(p=3)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.img1x1conv1 = nn.Conv2d(in_channels=768, out_channels=1024, kernel_size=1, bias=False)
        # self.img1x1conv2 = nn.Conv2d(in_channels=1024 , out_channels=1536, kernel_size=1, bias=False)
        self.img1x1conv3 = nn.Conv2d(in_channels=1536 , out_channels=2048, kernel_size=1, bias=False)
        self.img_fc = nn.Linear(2048, dim)
        self.final_fc = nn.Linear(3328, 2048)
        self.img_model.fc = nn.Sequential()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, imgs):
        imgs, early_stage = self.img_model(imgs)
        imgs = imgs.unsqueeze(dim=-1).unsqueeze(dim=-1)
        stage2 = early_stage[1]
        stage3 = early_stage[2]
        imgs = self.img1x1conv3(imgs)
        img_feature = self.img_fc(self.img_pool(imgs))

        stage2= self.avgpool(stage2.transpose(1, 2)).squeeze(dim=-1)
        stage3= self.avgpool(stage3.transpose(1, 2)).squeeze(dim=-1)
        final = self.final_fc(torch.cat((img_feature,stage2,stage3),dim=1)).unsqueeze(dim=-1).unsqueeze(dim=-1)
        imgs_final =final+imgs
        low_level_feature = imgs
        return imgs, img_feature, low_level_feature


class LabelSmooth1(torch.nn.Module):
    def __init__(self, epsilon=0.1, reduction='none'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = torch.nn.functional.log_softmax(preds, dim=-1)
        loss = -log_preds.sum(dim=-1)
        nll = torch.nn.functional.nll_loss(log_preds, target, reduction=self.reduction)
        return (self.epsilon* loss / n + (1-self.epsilon) * nll)


class LabelSmooth2(torch.nn.Module):
    def __init__(self, epsilon=0.1, reduction='none'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    def forward(self, preds, target):
        n_class = preds.size(-1)
        one_hot = torch.zeros_like(preds).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.epsilon) + (1 - one_hot) * self.epsilon / (n_class - 1)
        log_prb = torch.nn.functional.log_softmax(preds, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        return loss


class compose_local(nn.Module):
    def __init__(self, texts, T, word_dim=768, lstm_dim=1024, dim=1024, dropout_p=0.2):
        super().__init__()
        self.loss_weight = torch.nn.Parameter(torch.FloatTensor((10.,)))
        self.text_model = text_model_.TextLSTMModel(
            texts_to_build_vocab=texts,
            word_embed_dim=word_dim,
            lstm_hidden_dim=lstm_dim)
        self.img_model = img_backbone()
        self.local_fuse = local_conv(T = T)
        self.correct_module = FusDiff(embed_dim=dim)
        self.relu = nn.ReLU()
        self.local_pool = GeM(p=3)
        self.local_fc = nn.Linear(2048, dim)
        self.dropout = nn.Dropout(dropout_p)

    def extract_img_feature(self, imgs):
        imgs, img_feature, low_level_feature = self.img_model(imgs)
        return imgs, img_feature

    def compose_img_text(self, imgs, texts):
        lstm_tensor = self.text_model(texts) # N,L,C
        imgs, img_feature = self.extract_img_feature(imgs)

        local_out, attn = self.local_fuse(imgs, lstm_tensor) 
        local_feature = self.local_fc(self.local_pool(local_out))  
        return local_out, local_feature, lstm_tensor

    def correct_img_text(self, reference, target):
        # lstm_tensor = self.text_model(texts) # N,L,C
        # reference_imgs, _ = self.extract_img_feature(reference)
        # target_imgs, _ = self.extract_img_feature(target)

        diff = self.correct_module(reference, target) 
        # diff_feature = self.local_fc(self.local_pool(diff))  
        return diff


    def compute_loss(self, img1, mods, img2, target_global_tensor, target_global_feature, query_global_feature):
        target_local_tensor, target_local_feature = self.extract_img_feature(img2)
        query_local_tensor, query_local_feature, mod_feature = self.compose_img_text(img1, mods)
        B, C, H, W = img1.shape
        mask = torch.zeros(B,C,H,W).cuda()
        _, simi_local, _ = self.compose_img_text(mask, mods)
        diff = self.correct_img_text(query_local_feature, target_local_feature)
        mod = torch.nn.functional.adaptive_max_pool2d(mod_feature,(1,1024)).squeeze(1)
        loss = {}
        loss['img'] = self.compute_l2(target_local_feature, target_global_feature) + self.compute_l2(query_local_feature, query_global_feature)
        loss['class'] = self.compute_batch_based_classification_loss_(query_local_feature, target_local_feature)
        loss['perceptual'] = self.compute_l2(query_local_tensor, target_local_tensor)
        loss['mul_kl'] = self.mutual_learning(query_local_feature, target_local_feature, query_global_feature, target_global_feature)
        return loss

    def mutual_learning(self, query1, target1, query2, target2):
        query1 = F.normalize(query1, p=2, dim=-1)
        query2 = F.normalize(query2, p=2, dim=-1)
        target1 = F.normalize(target1, p=2, dim=-1)
        target2 = F.normalize(target2, p=2, dim=-1)
        x1 = 10.0 * torch.mm(query1, target1.transpose(0, 1)) 
        x2 = 10.0 * torch.mm(query2, target2.transpose(0, 1))

        log_soft_x1 = F.log_softmax(x1, dim=1)
        soft_x2 = F.softmax(torch.autograd.Variable(x2), dim=1)
        kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')
        return kl


    def compute_batch_based_classification_loss_(self, mod_img1, img2, negtive=None):
        mod_img1 = F.normalize(mod_img1, p=2, dim=-1)
        img2 = F.normalize(img2, p=2, dim=-1)
        x = torch.mm(mod_img1, img2.transpose(0, 1)) 
        if negtive is not None:
            negtive = F.normalize(negtive)
            y = torch.mm(mod_img1, negtive.transpose(0, 1)).diag().unsqueeze(-1)
            x = torch.cat([x, y], dim=-1)
        
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        loss = F.cross_entropy(self.loss_weight * x, labels)   # loss_weight temperature
        return loss
    
    def compute_l2(self, x1, x2):
        l2_loss = torch.nn.MSELoss()
        return l2_loss(x1, x2)


class compose_global(nn.Module):
    def __init__(self, texts, T, word_dim=768, lstm_dim=1024, dim=1024, dropout_p=0.2):
        super().__init__()
        self.loss_weight = torch.nn.Parameter(torch.FloatTensor((10.,)))
        self.text_model = text_model_.TextLSTMModel(
            texts_to_build_vocab=texts,
            word_embed_dim=word_dim,
            lstm_hidden_dim=lstm_dim)
        self.img_model = img_backbone()
        self.correct_module = FusDiff(embed_dim=dim)
        self.global_fuse = global_conv(T=T)
        self.relu = nn.ReLU()
        self.global_pool = GeM(p=3)
        self.global_fc = nn.Linear(1024, dim)
        self.dropout = nn.Dropout(dropout_p)

    def extract_img_feature(self, imgs):
        imgs, img_feature, low_level_feature = self.img_model(imgs)
        return imgs, img_feature

    def correct_img_text(self, reference, target):
        # lstm_tensor = self.text_model(texts) # N,L,C
        # reference_imgs, _ = self.extract_img_feature(reference)
        # target_imgs, _ = self.extract_img_feature(target)

        diff = self.correct_module(reference, target) 
        # diff_feature = self.local_fc(self.local_pool(diff))  
        return diff

    def compose_img_text(self, imgs, texts):
        lstm_tensor = self.text_model(texts) # N,L,C
        imgs, img_feature = self.extract_img_feature(imgs)

        global_out, attn = self.global_fuse(imgs, lstm_tensor)
        global_feature = self.global_fc(self.global_pool(global_out))
        return global_feature, lstm_tensor

    def compute_loss(self, img1, mods, img2, target_local_tensor, target_local_feature, query_local_feature):
        target_global_tensor, target_global_feature = self.extract_img_feature(img2)
        query_global_feature, mod_feature = self.compose_img_text(img1, mods)
        B, C, H, W = img1.shape
        mask = torch.zeros(B,C,H,W).cuda()
        simi_global,_  = self.compose_img_text(mask, mods)
        diff = self.correct_img_text(query_global_feature, target_global_feature)
        mod = torch.nn.functional.adaptive_max_pool2d(mod_feature,(1,1024)).squeeze(1)

        loss = {}
        loss['img'] =  self.compute_l2(target_local_feature, target_global_feature) + self.compute_l2(query_local_feature, query_global_feature)
        loss['class'] = self.compute_batch_based_classification_loss_(query_global_feature, target_global_feature)
        loss['mul_kl'] = self.mutual_learning(query_global_feature, target_global_feature, query_local_feature, target_local_feature)
        # loss['correct'] = self.compute_batch_based_classification_loss_(diff, mod)
        # loss['simi'] = self.compute_batch_based_classification_loss_(simi_global, target_global_feature)
        return loss


    def mutual_learning(self, query1, target1, query2, target2):
        query1 = F.normalize(query1, p=2, dim=-1)
        query2 = F.normalize(query2, p=2, dim=-1)
        target1 = F.normalize(target1, p=2, dim=-1)
        target2 = F.normalize(target2, p=2, dim=-1)
        x1 = 10.0 * torch.mm(query1, target1.transpose(0, 1)) 
        x2 = 10.0 * torch.mm(query2, target2.transpose(0, 1))

        log_soft_x1 = F.log_softmax(x1, dim=1)
        soft_x2 = F.softmax(torch.autograd.Variable(x2), dim=1)
        kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')
        return kl


    def compute_batch_based_classification_loss_(self, mod_img1, img2, negtive=None):
        mod_img1 = F.normalize(mod_img1, p=2, dim=-1)
        img2 = F.normalize(img2, p=2, dim=-1)
        x = torch.mm(mod_img1, img2.transpose(0, 1)) 
        if negtive is not None:
            negtive = F.normalize(negtive)
            y = torch.mm(mod_img1, negtive.transpose(0, 1)).diag().unsqueeze(-1)
            x = torch.cat([x, y], dim=-1)

        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()

        loss = F.cross_entropy(self.loss_weight * x, labels)   # loss_weight temperature
        return loss
    
    def compute_l2(self, x1, x2):
        l2_loss = torch.nn.MSELoss()
        return l2_loss(x1, x2)


class InnerProduct(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, trg_embds, src_embds, subspaces,
                l2renorm=False, tol=1e-5, dist=False, val=False):
        sims = sharded_cross_view_inner_product(
            trg_embds, src_embds, subspaces,
            l2renorm, tol, dist, val)
        return sims


def sharded_cross_view_inner_product(trg_embds, src_embds, subspaces,
                                     l2renorm=False, tol=1e-5, dist=False, val=False):
    '''
    args
      trg_embds: {attr0: B x N x D or N x D}
      src_embds: {attr0: B x D}
    '''
    B = src_embds[subspaces[0]].size(0)
    
    device = trg_embds[subspaces[0]].device
    trg_dim_idx = len(trg_embds[subspaces[0]].size()) - 1 # 2 or 1
    N = trg_embds[subspaces[0]].size(trg_dim_idx - 1)
    # B x N 
    sims = torch.zeros((B, N), device=device)

    if l2renorm:
        l2_mass_trg, l2_mass_src = 0, 0
        for idx, modality in enumerate(subspaces):
            trg_embd_ = trg_embds[modality] # B x N x D or N x D
            l2_mass_trg += trg_embd_.pow(2).sum(trg_dim_idx)
            src_embd_ = src_embds[modality] # B x D
            l2_mass_src += src_embd_.pow(2).sum(1)
        l2_mass_trg = torch.sqrt(l2_mass_trg.clamp(min=1e-6)).unsqueeze(trg_dim_idx)
        l2_mass_src = torch.sqrt(l2_mass_src.clamp(min=1e-6)).unsqueeze(1)
    else:
        l2_mass_trg, l2_mass_src = 1, 1

    for idx, modality in enumerate(subspaces):
        trg_embd_ = trg_embds[modality] / l2_mass_trg # B x N x D or N x D
        src_embd_ = src_embds[modality] / l2_mass_src # B x D
        if dist:
            sims += (trg_embd_ - src_embd_.unsqueeze(1)).pow(2).sum(2) # B x N
        else:
            if trg_dim_idx == 2:
                tmp = torch.matmul(trg_embd_, src_embd_.unsqueeze(-1)) # B x N x 1
                sims += tmp.squeeze(-1) # B x N
            else:
                sims += torch.matmul(src_embd_, trg_embd_.t())  # B x N
    # print(torch.max(sims))
    if torch.isnan(sims).sum().item():
        import ipdb; ipdb.set_trace()
        raise ValueError("Found nans in similarity matrix!")

    return sims

