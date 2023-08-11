import numpy as np
import torch
from tqdm import tqdm as tqdm
import torch.nn.functional as F

def test(params, local_model, global_model, testset, dataname):
    """Tests a model over the given testset."""
    if dataname == 'fashion200k':
        local_model.eval()
        global_model.eval()
        test_queries = testset.get_test_queries()
        with torch.no_grad():
            all_imgs = []
            all_captions = []
            all_queries = []
            all_target_captions = []
            if test_queries:
                # compute test query features
                imgs = []
                mods = []
                for t in tqdm(test_queries):
                    imgs += [testset.get_img(t['source_img_id'])]
                    mods += [t['mod']['str']]
                    if len(imgs) >= params.batch_size or t is test_queries[-1]:
                        if 'torch' not in str(type(imgs[0])):
                            imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs = torch.stack(imgs).float()
                        imgs = torch.autograd.Variable(imgs).cuda()
                        mods = [t.encode('utf-8').decode('utf-8') for t in mods]
                        f = torch.cat([F.normalize(local_model.compose_img_text(imgs, mods)[1]), F.normalize(global_model.compose_img_text(imgs, mods)[0])], dim=1)
                        f = f.data.cpu().numpy()
                        all_queries += [f]
                        imgs = []
                        mods = []
                all_queries = np.concatenate(all_queries)
                all_target_captions = [t['target_caption'] for t in test_queries]

                # compute all image features
                imgs = []
                for i in tqdm(range(len(testset.imgs))):
                #for i in tqdm(range(100)):
                    imgs += [testset.get_img(i)]
                    if len(imgs) >= params.batch_size or i == len(testset.imgs) - 1:
                        if 'torch' not in str(type(imgs[0])):
                            imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs = torch.stack(imgs).float()
                        imgs = torch.autograd.Variable(imgs).cuda()
                        imgs = torch.cat([F.normalize(local_model.extract_img_feature(imgs)[1]), F.normalize(global_model.extract_img_feature(imgs)[1])], dim=1).data.cpu().numpy()
                        all_imgs += [imgs]
                        imgs = []
                all_imgs = np.concatenate(all_imgs)
                all_captions = [img['captions'][0] for img in testset.imgs]

            # feature normalization
            for i in range(all_queries.shape[0]):
                all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
            for i in range(all_imgs.shape[0]):
                all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

            # match test queries to target images, get nearest neighbors
            sims = all_queries.dot(all_imgs.T)
            if test_queries:
                for i, t in enumerate(test_queries):
                    sims[i, t['source_img_id']] = -10e10  # remove query image
            nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]

            # compute recalls
            out = []
            nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
            for k in [1, 10, 50]:
                r = 0.0
                for i, nns in enumerate(nn_result):
                    if all_target_captions[i] in nns[:k]:
                        r += 1
                r /= len(nn_result)
                out += [('recall_top' + str(k) + '_correct_composition', r)]

            return out

    else:
        local_model.eval()
        global_model.eval()
        with torch.no_grad():
            test_queries = testset.get_test_queries()
            test_targets = testset.get_test_targets()

            all_queries = []
            all_imgs = []
            if test_queries:
                # compute test query features
                imgs = []
                mods = []
                for t in tqdm(test_queries):
                    imgs += [t['source_img_data']]
                    mods += [t['mod']['str']]
                    if len(imgs) >= params.batch_size or t is test_queries[-1]:
                        if 'torch' not in str(type(imgs[0])):
                            imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs = torch.stack(imgs).float().cuda()
                        f = torch.cat([F.normalize(local_model.compose_img_text(imgs, mods)[1]), F.normalize(global_model.compose_img_text(imgs, mods)[0])], dim=1)
                        f = f.data.cpu().numpy()
                        all_queries += [f]
                        imgs = []
                        mods = []
                all_queries = np.concatenate(all_queries)

                # compute all image features
                imgs = []
                logits = []
                for t in tqdm(test_targets):
                    imgs += [t['target_img_data']]
                    if len(imgs) >= params.batch_size or t is test_targets[-1]:
                        if 'torch' not in str(type(imgs[0])):
                            imgs = [torch.from_numpy(d).float() for d in imgs]
                        imgs = torch.stack(imgs).float().cuda()
                        imgs = torch.cat([F.normalize(local_model.extract_img_feature(imgs)[1]), F.normalize(global_model.extract_img_feature(imgs)[1])], dim=1).data.cpu().numpy()
                        all_imgs += [imgs]
                        imgs = []
                all_imgs = np.concatenate(all_imgs)

        # feature normalization
        for i in range(all_queries.shape[0]):
            all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
        for i in range(all_imgs.shape[0]):
            all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])
        
        
        # match test queries to target images, get nearest neighbors
        sims = all_queries.dot(all_imgs.T)
        
        test_targets_id = []
        for i in test_targets:
            test_targets_id.append(i['target_img_id'])
        for i, t in enumerate(test_queries):
            if not t['source_img_id'] in test_targets_id:
                continue
            sims[i, test_targets_id.index(t['source_img_id'])] = -10e10


        nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]

        # compute recalls
        out = []
        for k in [1, 10, 50]:
            r = 0.0
            for i, nns in enumerate(nn_result):
            
                if not test_queries[i]['target_img_id'] in test_targets_id:
                    continue
                if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                    r += 1
            r = 100 * r / len(nn_result)
            out += [('{}_r{}'.format(dataname, k), r)]
        return out




