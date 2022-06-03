import torch
import time
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_self_train(batch):
    batch_mod = {'sketch_img': [], 'sketch_boxes': [],
                 'positive_img': [], 'positive_boxes': [],
                 'negative_img': [], 'negative_boxes': [],
                 }
    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['positive_img'].append(i_batch['positive_img'])
        batch_mod['negative_img'].append(i_batch['negative_img'])
        batch_mod['sketch_boxes'].append(torch.tensor(i_batch['sketch_boxes']).float())
        batch_mod['positive_boxes'].append(torch.tensor(i_batch['positive_boxes']).float())
        batch_mod['negative_boxes'].append(torch.tensor(i_batch['negative_boxes']).float())

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)
    batch_mod['negative_img'] = torch.stack(batch_mod['negative_img'], dim=0)

    return batch_mod


def collate_self_test(batch):
    batch_mod = {'sketch_img': [], 'sketch_boxes': [], 'sketch_path': [],
                 'positive_img': [], 'positive_boxes': [], 'positive_path': [],
                 }

    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['sketch_path'].append(i_batch['sketch_path'])
        batch_mod['positive_img'].append(i_batch['positive_img'])
        batch_mod['positive_path'].append(i_batch['positive_path'])
        batch_mod['sketch_boxes'].append(torch.tensor(i_batch['sketch_boxes']).float())
        batch_mod['positive_boxes'].append(torch.tensor(i_batch['positive_boxes']).float())

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)

    return batch_mod


def evaluate(model, datloader_Test):
    Image_Feature_ALL = []
    Image_Name = []
    Sketch_Feature_ALL = []
    Sketch_Name = []
    start_time = time.time()
    model.eval()
    for i_batch, sanpled_batch in enumerate(datloader_Test):
        sketch_feature, positive_feature= model.test_forward(sanpled_batch)
        Sketch_Feature_ALL.extend(sketch_feature)
        Sketch_Name.extend(sanpled_batch['sketch_path'])

        for i_num, positive_name in enumerate(sanpled_batch['positive_path']):
            if positive_name not in Image_Name:
                Image_Name.append(sanpled_batch['positive_path'][i_num])
                Image_Feature_ALL.append(positive_feature[i_num])

    rank = torch.zeros(len(Sketch_Name))
    Image_Feature_ALL = torch.stack(Image_Feature_ALL)

    for num, sketch_feature in enumerate(Sketch_Feature_ALL):
        s_name = Sketch_Name[num]
        sketch_query_name = '_'.join(s_name.split('/')[-1].split('_')[:-1])
        position_query = Image_Name.index(sketch_query_name)

        distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
        target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                              Image_Feature_ALL[position_query].unsqueeze(0))

        rank[num] = distance.le(target_distance).sum()

    top1 = rank.le(1).sum().numpy() / rank.shape[0]
    top10 = rank.le(10).sum().numpy() / rank.shape[0]

    print('Time to EValuate:{}'.format(time.time() - start_time))
    return top1, top10
