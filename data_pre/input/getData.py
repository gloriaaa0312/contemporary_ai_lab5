# 这个文件主要用于读取数据
import json


# 可能会出现UnicodeDecodeError报错 需要尝试使用ANSI这一编码方式打开
def getEncoding(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            f.readline()
            return 'utf-8'
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='ANSI') as f:
                f.readline()
                return 'ANSI'
        except UnicodeDecodeError:
            exit(-1)


def run():
    train_guid = []
    test_guid = []
    train_data = []
    test_data = []
    labels = dict()

    # 将train.txt中的内容存入字典（labels
    with open('../train.txt', 'r', encoding='utf-8') as f:
        # 跳过表头
        next(f)
        for line in f:
            line_ = line.split(',')
            subline_ = line_[1].split('\n')
            labels[int(line_[0])] = subline_[0]
            train_guid.append(int(line_[0]))
    # print(labels)

    with open('../test_without_label.txt', 'r', encoding='utf-8') as f:
        # 跳过表头
        next(f)
        for line in f:
            line_ = line.split(',')
            labels[int(line_[0])] = ''
            test_guid.append(int(line_[0]))

    # print(labels)

    # 遍历训练数据
    for i in train_guid:
        path = '../data/' + str(i) + '.txt'
        encoding = getEncoding(path)
        with open(path, 'r', encoding=encoding) as f:
            text = f.read().split('\n')[0]
            # print(text)
            tag = labels.get(i)
            data = {
                'guid': i,
                'text': text,
                'img': str(i) + '.jpg',
                'tag': tag
            }
            train_data.append(data)

    # 遍历测试数据
    for i in test_guid:
        path = '../data/' + str(i) + '.txt'
        # print(path)
        encoding = getEncoding(path)
        with open(path, 'r', encoding=encoding) as f:
            text = f.read().split('\n')[0]
            # print(text)
            tag = labels.get(i)
            data = {
                'guid': i,
                'text': text,
                'tag': tag,
                'img': str(i) + '.jpg'
            }
            test_data.append(data)


    # print(len(train_data))
    # print(len(test_data))


    with open('trainData.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open('testData.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)