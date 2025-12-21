import csv
import os


def find_max_weighted_sum_index(acc_list, nmi_list, pur_list, ari_list, acc_weight, nmi_weight, pur_weight, ari_weight):
    max_sum = float('-inf')
    max_index = -1

    for i, (acc, nmi, pur, ari) in enumerate(zip(acc_list, nmi_list, pur_list, ari_list)):
        current_sum = acc * acc_weight + nmi * nmi_weight + pur * pur_weight + ari * ari_weight
        if current_sum > max_sum:
            max_sum = current_sum
            max_index = i

    return max_index


def save_lists_to_file(acc_list, nmi_list, pur_list, ari_list, data_name, missing_ratio):
    # 创建logs文件夹
    csv_path = f'3.csv'
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    # 创建以data_name命名的csv文件路径
    file_path = os.path.join(csv_path, f'{data_name}_{missing_ratio}.csv')

    # 写入数据到CSV文件
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # 写入表头
        csvwriter.writerow(['epoch', 'acc', 'nmi', 'pur', 'ari'])
        # 写入数据
        epoch = 1
        for acc, nmi, pur, ari in zip(acc_list, nmi_list, pur_list, ari_list):
            csvwriter.writerow([epoch, acc, nmi, pur, ari])
            epoch += 1

    print(f'Metrics have been saved at {file_path}')
