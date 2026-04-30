import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

def ca_weight(proj_query, proj_key):
    [b, c, h, w] = proj_query.shape
    proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h).permute(0, 2, 1)
    proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w).permute(0, 2, 1)
    proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
    proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
    energy_H = torch.bmm(proj_query_H, proj_key_H).view(b, w, h, h).permute(0, 2, 1, 3)
    energy_W = torch.bmm(proj_query_W, proj_key_W).view(b, h, w, w)
    concate = torch.cat([energy_H, energy_W], 3)

    return concate

def ca_map(attention, proj_value):
    [b, c, h, w] = proj_value.shape
    proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(b * w, -1, h)
    proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(b * h, -1, w)
    att_H = attention[:, :, :, 0:h].permute(0, 2, 1, 3).contiguous().view(b * w, h, h)
    att_W = attention[:, :, :, h:h + w].contiguous().view(b * h, w, w)
    out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(b, w, -1, h).permute(0, 2, 3, 1)
    out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(b, h, -1, w).permute(0, 2, 1, 3)
    out = out_H + out_W

    return out

class CCAttention(nn.Module):
    def __init__(self, in_dim, n_classes):
        super(CCAttention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(in_dim, n_classes)
        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))



    def forward(self, x, H, W):
        device = x.device
        _, C = x.shape
        feat_token = x[:, 0:]
        cnn_feat = feat_token.transpose(0, 1).view(1, C, H, W)

        proj_query = self.query_conv(cnn_feat)
        proj_key = self.key_conv(cnn_feat)
        proj_value = self.value_conv(cnn_feat)

        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        # print('attention.shape:', attention.shape)
        out = ca_map(attention, proj_value)
        out = self.gamma * out + cnn_feat

        proj_query1 = self.query_conv1(out)
        proj_key1 = self.key_conv1(out)
        proj_value1 = self.value_conv1(out)

        energy1 = ca_weight(proj_query1, proj_key1)
        attention1 = F.softmax(energy1, 1)
        out1 = ca_map(attention1, proj_value1)
        out1 = self.gamma1 * out1 + out

        out1 = out1.flatten(2).transpose(1, 2)
        B, N, C = out1.shape
        out1 = torch.reshape(out1, (N, C))
        out1 = self.fc(out1)
        return out1, x

class PPEG(nn.Module):
    def __init__(self, dim=1024):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        _, C = x.shape
        feat_token = x[:, 0:]
        cnn_feat = feat_token.transpose(0, 1).view(1, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        B, N, C = x.shape
        x = torch.reshape(x, (N, C))
        return x

class MSRAMIL_C(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(MSRAMIL_C, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        self.poslayer = PPEG(1024)
        self.poslayer1 = PPEG(512)
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc1 = [nn.Linear(size[1], size[1]), nn.ReLU()]
        if dropout:
            fc1.append(nn.Dropout(0.25))
        self.fc1 = nn.Sequential(*fc)
        self.fc2 = nn.Sequential(*fc1)
        self.attention_net = CCAttention(in_dim=size[1], n_classes=1)
        self.attention_net1 = CCAttention(in_dim=size[1], n_classes=1)
        # self.attention_net1 = attention_net
        self.classifiers = nn.Linear(size[0], n_classes)
        self.classifiers1 = nn.Linear(size[1], n_classes)
        self.classifiers2 = nn.Linear(size[1], n_classes)
        self.classifiers3 = nn.Linear(size[1], n_classes)
        self.classifiers4 = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.linear = nn.Linear(5, 512)  # 特征个数修改
        self.relu = nn.ReLU(True)
        self.linear1 = nn.Linear(5, 512)  # 特征个数修改
        self.relu1 = nn.ReLU(True)

        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.linear = self.linear.to(device)
        self.relu = self.relu.to(device)
        self.linear1 = self.linear1.to(device)
        self.relu1 = self.relu1.to(device)
        self.poslayer = self.poslayer.to(device)
        self.poslayer1 = self.poslayer1.to(device)
        self.attention_net = self.attention_net.to(device)
        self.attention_net1 = self.attention_net1.to(device)
        self.classifiers = self.classifiers.to(device)
        self.classifiers1 = self.classifiers1.to(device)
        self.classifiers2 = self.classifiers2.to(device)
        self.classifiers3 = self.classifiers3.to(device)
        self.classifiers4 = self.classifiers4.to(device)
        self.fc1 = self.fc1.to(device)
        self.fc2 = self.fc2.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()


    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        self.k_sample = 200

        k_sample = int(min(self.k_sample, A.shape[1] // 2))

        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k_sample, device)
        n_targets = self.create_negative_targets(k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        k_sample = int(min(self.k_sample, A.shape[1] // 2))

        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h0, clinical_features, label=None, instance_eval=False, return_features=False,
                attention_only=False):
        ps = h0.size(1)
        h0 = h0.squeeze()
        H = h0.shape[0]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h1 = torch.cat([h0, h0[:add_length, :]], dim=0)
        h1 = self.poslayer(h1, _H, _W)

        H1 = h1.shape[0]
        _H1, _W1 = int(np.ceil(np.sqrt(H1))), int(np.ceil(np.sqrt(H1)))
        add_length1 = _H1 * _W1 - H1
        h2 = torch.cat([h1, h1[:add_length1, :]], dim=0)

        h2 = self.fc1(h2)

        A, h = self.attention_net(h2, _H, _W)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)

        split_size = A.shape[1] // 3
        A_parts = torch.split(A, split_size, dim=1)
        final_instances = []
        final_score = []

        for part in A_parts:
            num_top_indices = int(0.2 * part.size(1))
            num_top_indices = max(num_top_indices, 1)
            top_indices = torch.topk(part, num_top_indices, dim=1)[1]
            unique_top_indices = torch.unique(top_indices)
            selected_remaining_h = torch.index_select(h, dim=0, index=unique_top_indices.view(-1))
            selected_remaining_scores = torch.index_select(part, dim=1, index=unique_top_indices)
            final_instances.append(selected_remaining_h)
            final_score.append(selected_remaining_scores)

        M1, M2, M3 = final_instances[0], final_instances[1], final_instances[2]

        S1, S2, S3 = final_score[0], final_score[1], final_score[2]

        M11 = torch.mm(S1, M1)
        M22 = torch.mm(S2, M2)
        M33 = torch.mm(S3, M3)

        logits1 = self.classifiers1(M11)
        logits2 = self.classifiers2(M22)
        logits3 = self.classifiers3(M33)

        h_mult = torch.cat((M1, M2, M3), dim=0)
        H_mult = h_mult.shape[0]
        _H_mult, _W_mult = int(np.ceil(np.sqrt(H_mult))), int(np.ceil(np.sqrt(H_mult)))
        add_length_mult = _H_mult * _W_mult - H_mult
        h1_mult = torch.cat([h_mult, h_mult[:add_length_mult, :]], dim=0)
        h1_mult = self.poslayer1(h1_mult, _H_mult, _W_mult)

        H1_mult = h1_mult.shape[0]
        _H1_mult, _W1_mult = int(np.ceil(np.sqrt(H1_mult))), int(np.ceil(np.sqrt(H1_mult)))
        add_length1_mult = _H1_mult * _W1_mult - H1_mult
        h2_mult = torch.cat([h1_mult, h1_mult[:add_length1_mult, :]], dim=0)

        h2_mult = self.fc2(h2_mult)

        A_mult, h_mult = self.attention_net1(h2_mult, _H_mult, _W_mult)  # NxK
        A_mult = torch.transpose(A_mult, 1, 0)  # KxN
        if attention_only:
            return A_mult
        A_raw_mult = A_mult
        A_mult = F.softmax(A_mult, dim=1)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A_mult, h_mult, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A_mult, h_mult, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A_mult, h_mult)
        logits_WSI = self.classifiers4(M)
        clinical = self.linear(clinical_features)
        clinical = self.relu(clinical)
        M = torch.cat([M, clinical], dim=1)
        logits_CLI = self.classifiers(M)
        logits = logits_CLI * 0.5 + logits_WSI * 0.5
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})

        return logits1, logits2, logits3, logits, Y_prob, Y_hat, A_raw, results_dict

class MSRAMIL(nn.Module):
    def __init__(self, num_feats= 1024 ,gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(MSRAMIL, self).__init__()
        # self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.size_dict = {"small": [num_feats, 512, 256], "big": [num_feats, 512, 384]}
        size = self.size_dict[size_arg]

        self.poslayer = PPEG(num_feats)
        self.poslayer1 = PPEG(512)
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc1 = [nn.Linear(size[1], size[1]),
               nn.ReLU()]
        if dropout:
            fc1.append(nn.Dropout(0.25))
        self.fc1 = nn.Sequential(*fc)
        self.fc2 = nn.Sequential(*fc1)
        self.attention_net = CCAttention(in_dim=size[1], n_classes=1)
        self.attention_net1 = CCAttention(in_dim=size[1], n_classes=1)
        # self.attention_net1 = attention_net
        self.classifiers = nn.Linear(size[1], n_classes)
        self.classifiers1 = nn.Linear(size[1], n_classes)
        self.classifiers2 = nn.Linear(size[1], n_classes)
        self.classifiers3 = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping


        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.poslayer = self.poslayer.to(device)
        self.poslayer1 = self.poslayer1.to(device)
        self.attention_net = self.attention_net.to(device)
        self.attention_net1 = self.attention_net1.to(device)
        self.classifiers = self.classifiers.to(device)
        self.classifiers1 = self.classifiers1.to(device)
        self.classifiers2 = self.classifiers2.to(device)
        self.classifiers3 = self.classifiers3.to(device)
        self.fc1 = self.fc1.to(device)
        self.fc2 = self.fc2.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)


    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        self.k_sample = 200
        k_sample = int(min(self.k_sample, A.shape[1] // 2))

        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k_sample, device)
        n_targets = self.create_negative_targets(k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        k_sample = int(min(self.k_sample, A.shape[1] // 2))

        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h0, label=None, instance_eval=False, return_features=False, attention_only=False):

        H = h0.shape[0]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h1 = torch.cat([h0, h0[:add_length, :]], dim=0)
        h1 = self.poslayer(h1, _H, _W)

        H1 = h1.shape[0]
        _H1, _W1 = int(np.ceil(np.sqrt(H1))), int(np.ceil(np.sqrt(H1)))
        add_length1 = _H1 * _W1 - H1
        h2 = torch.cat([h1, h1[:add_length1, :]], dim=0)

        h2 = self.fc1(h2)
        A, h = self.attention_net(h2, _H, _W)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)
        split_size = A.shape[1] // 3
        A_parts = torch.split(A, split_size, dim=1)

        final_instances = []
        final_score = []


        for part in A_parts:
            num_top_indices = int(0.2 * part.size(1))


            num_top_indices = max(num_top_indices, 1)

            top_indices = torch.topk(part, num_top_indices, dim=1)[1]

            unique_top_indices = torch.unique(top_indices)

            selected_remaining_h = torch.index_select(h, dim=0, index=unique_top_indices.view(-1))
            selected_remaining_scores = torch.index_select(part, dim=1, index=unique_top_indices)

            final_instances.append(selected_remaining_h)
            final_score.append(selected_remaining_scores)

        M1, M2, M3 = final_instances[0], final_instances[1], final_instances[2]
        S1, S2, S3 = final_score[0], final_score[1], final_score[2]

        M11 = torch.mm(S1, M1)
        M22 = torch.mm(S2, M2)
        M33 = torch.mm(S3, M3)

        logits1 = self.classifiers1(M11)
        logits2 = self.classifiers2(M22)
        logits3 = self.classifiers3(M33)

        h_mult = torch.cat((M1, M2, M3), dim=0)
        H_mult = h_mult.shape[0]
        _H_mult, _W_mult = int(np.ceil(np.sqrt(H_mult))), int(np.ceil(np.sqrt(H_mult)))
        add_length_mult = _H_mult * _W_mult - H_mult
        h1_mult = torch.cat([h_mult, h_mult[:add_length_mult, :]], dim=0)
        h1_mult = self.poslayer1(h1_mult, _H_mult, _W_mult)
        H1_mult = h1_mult.shape[0]
        _H1_mult, _W1_mult = int(np.ceil(np.sqrt(H1_mult))), int(np.ceil(np.sqrt(H1_mult)))
        add_length1_mult = _H1_mult * _W1_mult - H1_mult
        h2_mult = torch.cat([h1_mult, h1_mult[:add_length1_mult, :]], dim=0)
        h2_mult = self.fc2(h2_mult)
        A_mult, h_mult = self.attention_net1(h2_mult, _H_mult, _W_mult)
        A_mult = torch.transpose(A_mult, 1, 0)
        if attention_only:
            return A_mult
        A_raw_mult = A_mult
        A_mult = F.softmax(A_mult, dim=1)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A_mult, h_mult, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A_mult, h_mult, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A_mult, h_mult)

        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})

        return logits1, logits2, logits3, logits, Y_prob, Y_hat, A_raw, results_dict



