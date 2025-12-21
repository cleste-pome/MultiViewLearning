import csv
import random
import torch
import os
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from tqdm import tqdm
from utils.dataloader import dataset_with_info
from models import MvAEModel
from utils import Logger
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from utils.metric import compute_metric, cluster_accuracy
from utils.metric2csv import find_max_weighted_sum_index, save_lists_to_file
from utils.move import move
from torch.utils.data import DataLoader, Subset
from utils.plot import plot_acc


def seed_setting(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def orthogonal_loss(shared, specific):
    _shared = shared.detach()
    _shared = _shared - _shared.mean(dim=0)
    correlation_matrix = _shared.t().matmul(specific)
    norm = torch.norm(correlation_matrix, p=1)
    return norm


def setup_log_directory(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        print("Logs directory created at:", log_path)
    else:
        print("Logs directory already exists at:", log_path)


def create_dataset_info_file(file_datasetInfo):
    headers = ['Dataname', 'number of data', 'views', 'clusters', 'each view']
    with open(file_datasetInfo, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    print(f"{file_datasetInfo} has been created")


def prepare_neighbors(dataset, train_index, view_num):
    train_ins_num = len(train_index)  # 训练集的样本数量
    neighbors_num = int(train_ins_num / 4)
    pos_num = 21
    neg_num = int((neighbors_num - pos_num - 1) / 2)
    nbr_idx, neg_idx = [], []

    for v in range(view_num):
        # 将数据转换为numpy数组，确保兼容sklearn的输入要求
        X_np = np.array([dataset[i][0][v].numpy() if isinstance(dataset[i][0][v], torch.Tensor)
                         else dataset[i][0][v] for i in train_index], dtype=object)

        # 检查是否能转换为标准的np.ndarray
        if all(x.shape == X_np[0].shape for x in X_np):  # 如果所有样本的特征维度一致
            X_np = np.vstack(X_np)  # 将X_np转换为标准的二维np.ndarray

        nbrs_v, neg_v = np.zeros((train_ins_num, pos_num - 1)), np.zeros((train_ins_num, neg_num))
        nbrs = NearestNeighbors(n_neighbors=neighbors_num, algorithm='auto').fit(X_np)
        dis, idx = nbrs.kneighbors(X_np)

        for i in range(train_ins_num):
            nbrs_v[i][:] = idx[i][1:pos_num]
            neg_v[i][:] = idx[i][-neg_num:]

        nbr_idx.append(torch.LongTensor(nbrs_v))
        neg_idx.append(torch.LongTensor(neg_v))

    nbr_idx = torch.cat(nbr_idx, dim=-1)
    neg_idx = torch.cat(neg_idx, dim=-1)
    return nbr_idx, neg_idx


def hungarian_align_pseudo_to_true(pseudo_y: np.ndarray,
                                   true_y: np.ndarray,
                                   num_classes) -> np.ndarray:
    """
    用匈牙利算法将 KMeans 的簇标签 pseudo_y 对齐到真实标签 true_y 的编号空间。

    参数
    - pseudo_y: ndarray, shape (N,), KMeans簇id（0..K-1，顺序任意）
    - true_y:   ndarray, shape (N,), 真实类别id（0..C-1）
    - num_classes: 可选，类别数C（也可用 max+1 自动推断；建议传入更稳）

    返回
    - pseudo_align_y: ndarray, shape (N,), 将 pseudo_y 映射到真实类编号后的结果
    """
    pseudo_y = np.asarray(pseudo_y).reshape(-1)
    true_y = np.asarray(true_y).reshape(-1)
    if pseudo_y.shape[0] != true_y.shape[0]:
        raise ValueError(f"Length mismatch: pseudo_y has {pseudo_y.shape[0]} samples, true_y has {true_y.shape[0]} samples")

    if num_classes is None:
        num_classes = int(max(pseudo_y.max(initial=0), true_y.max(initial=0))) + 1

    # confusion matrix: rows=cluster, cols=true class
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for c, t in zip(pseudo_y, true_y):
        cm[int(c), int(t)] += 1

    # 匈牙利：最大化匹配数 <=> 最小化 -cm
    row_ind, col_ind = linear_sum_assignment(-cm)

    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    # 若某些簇id超出 num_classes 或未被匹配（理论上不会，除非标签有洞/不连续），给个兜底映射到自身
    pseudo_align_y = np.array([mapping.get(int(ci), int(ci)) for ci in pseudo_y], dtype=np.int64)
    return pseudo_align_y


def train_one_epoch(args, train_loader, model, mse_loss_fn, optimizer, dataset, ins_num, view_num, epoch, device, nbr_idx, neg_idx):

    # TODO 1.train_loader
    criterion = nn.CrossEntropyLoss()
    for x, y, train_idx, pu in train_loader:
        optimizer.zero_grad()
        model.train()

        for v in range(view_num):
            x[v] = x[v].to(device)

        # TODO 2.1 classes
        hidden_share, hidden_specific, hidden, recs, classes = model(x)
        loss_rec, loss_mi, loss_ad, loss_class = 0, 0, 0, 0

        # 判断最小值并调整 y
        if y.min() == 1:
            y = (y - 1).long().to(device)  # 如果最小值为 1
        elif y.min() == 0:
            y = y.long().to(device)  # 如果最小值为 0，保持原值并转为长整型

        for v in range(view_num):
            loss_rec += mse_loss_fn(recs[v], x[v])
            loss_mi += orthogonal_loss(hidden_share, hidden_specific[v])
            loss_ad += model.discriminators_loss(hidden_specific, v)

            # Y_pre = KMeans(n_clusters=nc, n_init=50).fit_predict(hidden.detach().cpu().numpy()).astype(np.int64)
            # Y_ndarray = y.detach().cpu().numpy().astype(np.int64)
            # loss_class += 1 - cluster_accuracy(Y_ndarray, Y_pre) # TODO 用kmeans的输出做约束

            loss_class += criterion(classes, y) # TODO 默认不用额外softmax，criterion()内部有

        loss_con = contrastive_loss(args, hidden, nbr_idx, neg_idx, train_idx)
        total_loss = (
            loss_rec +
            args.lambda_ma * (loss_mi + loss_ad) +
            args.lambda_con * loss_con +
            loss_class
        )
        total_loss.backward()
        optimizer.step()


def contrastive_loss(args, hidden, nbr_idx, neg_idx, train_idx):
    if not args.do_contrast:
        return 0
    loss_con = 0
    for i in range(len(train_idx)):
        index = train_idx[i]
        if int(index) < len(train_idx) - 1:
            hidden_positive = hidden[nbr_idx[index]]
            positive = torch.exp(torch.cosine_similarity(hidden[i].unsqueeze(0), hidden_positive.detach()))
            hidden_negative = hidden[neg_idx[index]]
            negative = torch.exp(torch.cosine_similarity(hidden[i].unsqueeze(0), hidden_negative.detach())).sum()
            loss_con -= torch.log((positive / negative)).sum()
            torch.cuda.empty_cache()
    return loss_con / len(train_idx)


def evaluate_model(args, model, test_loader, nc, ins_num, view_num, epoch, device):
    # TODO 2.test_loader
    with torch.no_grad():
        for x, y, idx, pu in test_loader:
            for v in range(view_num):
                x[v] = x[v].to(device)
            model.eval()
            # TODO 2.2 classes
            hidden_share, hidden_specific, hidden, recs, classes = model(x)
            label = np.array(y)
            y_pred_2 = KMeans(n_clusters=nc, n_init=50).fit_predict(hidden.cpu().numpy()) # TODO y_pred_2: K-means
            ACC2, NMI2, Purity2, ARI2, F_score2, Precision2, Recall2 = compute_metric(label, y_pred_2)

            # TODO 3. class metric
            y_pred = torch.argmax(classes, dim=1).detach().cpu().numpy() # TODO y_pred: self.classifier(hidden)
            ACC, NMI, Purity, ARI, F_score, Precision, Recall = compute_metric(label, y_pred)
            return ACC, NMI, Purity, ARI, F_score, Precision, Recall, ACC2, NMI2, Purity2, ARI2, F_score2, Precision2, Recall2


def log_metrics(logger, epoch, acc, nmi, pur, ari):
    info = {
        "epoch": epoch,
        "acc": acc,
        "nmi": nmi,
        "ari": ari,
        "pur": pur
    }
    logger.info(str(info))


def plot_and_save_results(acc_list, nmi_list, pur_list, ari_list, dataset_name, args):
    plot_name = f'{dataset_name}_{args.ratio_noise}_{args.ratio_conflict}_{args.missing_ratio}'
    plot_acc(acc_list, dataset_name, 'acc')
    plot_acc(nmi_list, dataset_name, 'nmi')
    plot_acc(pur_list, dataset_name, 'pur')
    plot_acc(ari_list, dataset_name, 'ari')
    save_lists_to_file(acc_list, nmi_list, pur_list, ari_list, dataset_name, plot_name)


def log_best_results(logger, acc_list, nmi_list, pur_list, ari_list):
    max_index = find_max_weighted_sum_index(
        acc_list, nmi_list, pur_list, ari_list,
        acc_weight=0.25, nmi_weight=0.25, pur_weight=0.25, ari_weight=0.25
    )
    print(f'Max metric: epoch{max_index + 1}\n'
          f'1.acc:{acc_list[max_index] * 100:.2f}%\n'
          f'2.nmi:{nmi_list[max_index] * 100:.2f}%\n'
          f'3.pur:{pur_list[max_index] * 100:.2f}%\n'
          f'4.ari:{ari_list[max_index] * 100:.2f}%')
    info = {
        "[MAX]epoch": max_index + 1,
        "acc": acc_list[max_index],
        "nmi": nmi_list[max_index],
        "purity": pur_list[max_index],
        "ari": ari_list[max_index]
    }
    logger.info(str(info))


def print_metrics_table(epoch,
                        train_cls, train_km,
                        test_cls, test_km):
    """
    每个输入都是 (ACC, NMI, PUR, ARI)
    """
    header = (
        f"\nEpoch {epoch}\n"
        "+---------+-----------+--------+--------+--------+\n"
        "| Split   | Method    |  ACC   |  NMI   |  ARI   |\n"
        "+---------+-----------+--------+--------+--------+"
    )

    row_fmt = "| {:<7} | {:<9} | {:>6.4f} | {:>6.4f} | {:>6.4f} |"

    footer = "+---------+-----------+--------+--------+--------+"

    print(header)
    print(row_fmt.format("Train", "CLS",    *train_cls))
    print(row_fmt.format("Train", "KMeans", *train_km))
    print(footer)
    print(row_fmt.format("Test",  "CLS",    *test_cls))
    print(row_fmt.format("Test",  "KMeans", *test_km))
    print(footer)



def train_and_evaluate_model(args, dataset_name, dataset, ins_num, view_num, nc, input_dims, logger, Train_loader, Test_loader, nbr_idx, neg_idx):
    device = "cuda:0"

    # 创建一个新的CSV文件
    csv_file_path = os.path.join(args.log_path, f'{dataset_name}_results.csv')
    with (open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile):
        fieldnames = ['feature_dim', 'epoch', 'acc', 'nmi', 'purity', 'ari']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 遍历不同的 feature_dim
        for feature_dim in tqdm(args.feature_dims):
            print(f'Feature dim: {feature_dim}')
            h_dims = [500, 200]
            model = MvAEModel(input_dims, view_num, feature_dim, h_dims, nc).to(device)
            mse_loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            acc_list, nmi_list, pur_list, ari_list = [], [], [], []

            for epoch in tqdm(range(args.train_epoch)):
                train_one_epoch(args, Train_loader, model, mse_loss_fn, optimizer, dataset, ins_num, view_num, epoch, device,
                                nbr_idx, neg_idx)

                # 每隔 `eval_interval` 轮测试一次
                if (epoch + 1) % args.eval_interval == 0:
                    # TODO Train_loader: y_pred: 分类头; y_pred_2: K-means
                    acc_tr, nmi_tr, pur_tr, ari_tr, _, _, _, acc_tr2, nmi_tr2, pur_tr2, ari_tr2, _, _, _ = \
                        evaluate_model(args, model, Train_loader, nc, ins_num, view_num, epoch, device)
                    # TODO Test_loader: y_pred: 分类头; y_pred_2: K-means
                    acc_te, nmi_te, pur_te, ari_te, _, _, _, acc_te2, nmi_te2, pur_te2, ari_te2, _, _, _ = \
                        evaluate_model(args, model, Test_loader, nc, ins_num, view_num, epoch, device)
                    print_metrics_table(
                        epoch + 1,
                        train_cls=(acc_tr, nmi_tr, pur_tr, ari_tr),
                        train_km=(acc_tr2, nmi_tr2, pur_tr2, ari_tr2),
                        test_cls=(acc_te, nmi_te, pur_te, ari_te),
                        test_km=(acc_te2, nmi_te2, pur_te2, ari_te2)
                    )

                    acc_list.append(acc_te)
                    nmi_list.append(nmi_te)
                    pur_list.append(pur_te)
                    ari_list.append(ari_te)

                    log_metrics(logger, epoch + 1, acc_te, nmi_te, pur_te, ari_te)

                    # 将测试结果写入CSV文件
                    writer.writerow({
                        'feature_dim': feature_dim,
                        'epoch': epoch + 1,
                        'acc': acc_te,
                        'nmi': nmi_te,
                        'purity': pur_te,
                        'ari': ari_te
                    })

            plot_and_save_results(acc_list, nmi_list, pur_list, ari_list, f'{dataset_name}_fdim{feature_dim}', args)
            log_best_results(logger, acc_list, nmi_list, pur_list, ari_list)

    print(f'Results saved to {csv_file_path}')


if __name__ == '__main__':
    # for missratio in [0.0]:
    # 超参数设置
    class Args:
        seed = 42
        log_path = '1.logs'
        folder_path = "dataset"
        train_epoch = 200 # 500
        eval_interval = 20  # 100 每隔多少轮进行一次测试
        lr = 0.001
        feature_dims = [128]  # feature_dim 列表
        ratio_noise = 0.0
        ratio_conflict = 0.0
        missing_ratio = 0.0
        lambda_ma = 0.01
        lambda_con = 0.01
        do_contrast = True  # Do you have enough cuda memory, bro?


    args = Args()
    seed_setting(args.seed)

    setup_log_directory(args.log_path)
    file_datasetInfo = f'{args.log_path}/datasetInfo.csv'
    create_dataset_info_file(file_datasetInfo)

    file_names = os.listdir(args.folder_path)
    for dataset_name in file_names:
        if dataset_name.endswith(".mat"):
            dataset_name = dataset_name[:-4]
        else:
            continue

        logger = Logger.get_logger(__file__, dataset_name)
        dataset, ins_num, view_num, nc, input_dims, _ = dataset_with_info(
            dataset_name, file_datasetInfo, args.folder_path)

        index = np.arange(len(dataset))
        missing_ratio = args.missing_ratio
        dataset.addMissing(index, missing_ratio)  # TODO 选取missing ratio比例样本的随机(1到view-1)个视图做缺失处理
        ratio_conflict = args.ratio_conflict
        dataset.addConflict(index, ratio_conflict)  # TODO 选取conflict ratio比例的样本，随机选择一个视图的数据用另一个类别的样本的同视图数据替换
        ratio_noise = args.ratio_noise
        sigma = 0.5
        dataset.addNoise(index, ratio_noise, sigma)  # TODO 选取noise ratio比例的样本，随机(1到view-1)个视图做添加高斯噪声处理

        # ======= 每次运行都随机的 train/test 划分（不受全局seed影响）=======
        split_seed = int.from_bytes(os.urandom(8), "little")  # 每次启动程序都不同
        rng_split = np.random.default_rng(split_seed)

        index_dataset = np.arange(ins_num)
        rng_split.shuffle(index_dataset)

        split = int(0.8 * ins_num)
        train_index, test_index = index_dataset[:split], index_dataset[split:]

        # ======= DataLoader：每个 epoch 都会 shuffle（同一次运行内）=======
        # 可选：给 DataLoader 一个 generator，使 shuffle 可控且每个 epoch 都会变（状态会前进）
        g = torch.Generator()
        g.manual_seed(args.seed)  # 训练过程可复现（在同一次 split 条件下）

        train_loader = DataLoader(
            Subset(dataset, train_index),
            batch_size=split,
            shuffle=True,
            generator=g
        )

        test_loader = DataLoader(
            Subset(dataset, test_index),
            batch_size=ins_num - split,
            shuffle=False
        )

        nbr_idx, neg_idx = prepare_neighbors(dataset, train_index, view_num)

        train_and_evaluate_model(args, dataset_name, dataset, ins_num, view_num, nc, input_dims, logger,
                                 train_loader, test_loader, nbr_idx, neg_idx)

    # move(missratio)
