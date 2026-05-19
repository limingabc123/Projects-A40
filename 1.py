"""
BERT预训练数据集生成模块
============================
本模块实现了为BERT预训练生成数据集的完整流程，包括：
1. 读取WikiText-2语料库
2. 生成下一句预测(NSP)任务的训练数据
3. 生成遮蔽语言模型(MLM)任务的训练数据
4. 将文本转换为预训练数据集

参考：Dive into Deep Learning 14.9节
"""

import os
import sys
import random
import torch

# 修复 torchvision 兼容性问题
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fix_torchvision

from d2l import torch as d2l


# ============================================================
# 第一部分：数据读取
# ============================================================

# 注册WikiText-2数据集到d2l的数据中心
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')


def _read_wiki(data_dir):
    """读取WikiText-2数据集并返回段落列表。

    Parameters
    ----------
    data_dir : str
        数据集所在目录的路径（形参）。
        实参示例：'../data/wikitext-2'

    Returns
    -------
    paragraphs : list[list[str]]
        二维列表，每个元素是一个段落（句子列表）。
        每个段落被句号 ' . ' 分割成多个句子字符串。
    """
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 大写字母转换为小写字母，按 ' . ' 拆分句子
    # 仅保留至少有两句话的段落
    paragraphs = []
    for line in lines:
        sentences = [s for s in line.strip().lower().split(' . ') if s]
        if len(sentences) >= 2:
            paragraphs.append(sentences)
    random.shuffle(paragraphs)
    return paragraphs


# ============================================================
# 第二部分：下一句预测 (NSP) 任务的辅助函数
# ============================================================

def _get_next_sentence(sentence, next_sentence, paragraphs):
    """生成下一句预测任务的单个训练样本。

    以50%的概率返回真实的下一句（正样本），
    以50%的概率从语料库中随机选择一个句子作为下一句（负样本）。

    Parameters
    ----------
    sentence : str
        当前句子（形参）。
        实参示例：'the cat sat on the mat'
    next_sentence : str
        真实的下一句（形参）。
        实参示例：'it was a sunny day'
    paragraphs : list[list[str]]
        整个语料库的段落列表（形参），用于随机抽取负样本。
        实参示例：[['sentence a', 'sentence b'], ['sentence c', 'sentence d']]

    Returns
    -------
    sentence : str
        当前句子（不变）。
    next_sentence : str
        下一句（可能被替换为随机句子）。
    is_next : bool
        True表示next_sentence是真实的下一句，False表示是随机选择的。
    """
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs是三重列表的嵌套
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    """从一个段落中生成下一句预测任务的所有训练样本。

    遍历段落中的连续句子对，为每一对调用 _get_next_sentence 生成样本，
    然后拼接成BERT输入格式：[<cls>, tokens_a, <sep>, tokens_b, <sep>]。

    Parameters
    ----------
    paragraph : list[str]
        单个段落，包含多个句子字符串（形参）。
        实参示例：['the cat sat', 'on the mat', 'it was sunny']
    paragraphs : list[list[str]]
        整个语料库的段落列表（形参），传递给 _get_next_sentence。
    vocab : d2l.Vocab
        词表对象（形参），用于验证词元（此处传递但未直接使用）。
    max_len : int
        BERT输入序列的最大长度（形参）。
        实参示例：64

    Returns
    -------
    nsp_data_from_paragraph : list[tuple]
        每个元素是 (tokens, segments, is_next) 三元组：
        - tokens: 拼接后的词元列表 [<cls>, ..., <sep>, ..., <sep>]
        - segments: 段标记列表，0表示句子A，1表示句子B
        - is_next: 布尔值，是否为真实的下一句
    """
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 考虑1个'<cls>'词元和2个'<sep>'词元，所以总长度不能超过max_len
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


# ============================================================
# 第三部分：遮蔽语言模型 (MLM) 任务的辅助函数
# ============================================================

def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    """对BERT输入序列执行遮蔽操作，生成MLM任务的训练数据。

    遮蔽策略（遵循原始BERT论文）：
    - 80% 的概率：替换为 '<mask>' 词元
    - 10% 的概率：保持原词不变
    - 10% 的概率：替换为词表中的随机词

    Parameters
    ----------
    tokens : list[str]
        BERT输入序列的词元列表（形参）。
        实参示例：['<cls>', 'the', 'cat', '<sep>', 'sat', '<sep>']
    candidate_pred_positions : list[int]
        候选预测位置的索引列表（形参），不包含特殊词元的位置。
        实参示例：[1, 2, 4]
    num_mlm_preds : int
        要预测的词元数量（形参），通常为 round(len(tokens) * 0.15)。
        实参示例：10
    vocab : d2l.Vocab
        词表对象（形参），用于获取随机替换词元。

    Returns
    -------
    mlm_input_tokens : list[str]
        经过遮蔽处理后的词元列表（可能包含 '<mask>' 或随机词元）。
    pred_positions_and_labels : list[tuple[int, str]]
        每个元素是 (位置索引, 原始词元) 的二元组，
        记录了被预测位置及其真实标签。
    """
    # 为遮蔽语言模型的输入创建新的词元副本
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # 打乱候选位置，以便随机选择要遮蔽的词元
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%的时间：将词替换为"<mask>"词元
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的时间：保持词不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间：用随机词替换该词
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    """从BERT输入序列中生成遮蔽语言模型任务的完整训练数据。

    该函数先筛选出候选预测位置（排除特殊词元），
    再调用 _replace_mlm_tokens 执行遮蔽操作，
    最后将词元和标签转换为词表索引。

    Parameters
    ----------
    tokens : list[str]
        BERT输入序列的词元列表（形参）。
        实参示例：['<cls>', 'the', 'cat', '<sep>', 'sat', '<sep>']
    vocab : d2l.Vocab
        词表对象（形参），用于将词元转换为索引。

    Returns
    -------
    vocab[mlm_input_tokens] : list[int]
        遮蔽后的输入词元索引列表。
    pred_positions : list[int]
        被预测的词元位置索引列表（已排序）。
    vocab[mlm_pred_labels] : list[int]
        被预测位置的真实词元索引列表（已排序）。
    """
    candidate_pred_positions = []
    # tokens是一个字符串列表
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 遮蔽语言模型任务中预测15%的随机词元
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


# ============================================================
# 第四部分：填充与数据集构建
# ============================================================

def _pad_bert_inputs(examples, max_len, vocab):
    """将BERT预训练样本填充到统一长度，转换为张量。

    Parameters
    ----------
    examples : list[tuple]
        预训练样本列表（形参），每个样本是一个五元组：
        (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next)
        来自 _get_mlm_data_from_tokens 和 _get_nsp_data_from_paragraph 的输出。
    max_len : int
        序列最大长度（形参），用于填充。实参示例：64
    vocab : d2l.Vocab
        词表对象（形参），用于获取 '<pad>' 的索引。

    Returns
    -------
    all_token_ids : list[Tensor]
        填充后的词元ID张量列表，形状 [max_len]。
    all_segments : list[Tensor]
        填充后的段标记张量列表，形状 [max_len]。
    valid_lens : list[Tensor]
        有效长度（不含填充）的标量张量列表。
    all_pred_positions : list[Tensor]
        填充后的预测位置张量列表，形状 [max_num_mlm_preds]。
    all_mlm_weights : list[Tensor]
        MLM损失权重张量列表，真实预测位置权重为1.0，填充位置为0.0。
    all_mlm_labels : list[Tensor]
        MLM预测标签张量列表，形状 [max_num_mlm_preds]。
    nsp_labels : list[Tensor]
        NSP标签的标量张量列表（0或1）。
    """
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # 填充词元的预测将通过乘以0权重在损失中过滤掉
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)


class _WikiTextDataset(torch.utils.data.Dataset):
    """用于BERT预训练的WikiText-2数据集类。

    该类继承自 torch.utils.data.Dataset，整合了NSP和MLM两个
    预训练任务的数据生成流程。

    Parameters (构造函数形参)
    ----------
    paragraphs : list[list[str]]
        段落列表（形参），每个段落是句子字符串列表。
        实参来自 _read_wiki() 的返回值。
    max_len : int
        BERT输入序列的最大长度（形参）。
        实参示例：64

    Attributes
    ----------
    vocab : d2l.Vocab
        从语料库构建的词表，min_freq=5，包含特殊词元。
    all_token_ids : list[Tensor]
        所有样本的词元ID张量。
    all_segments : list[Tensor]
        所有样本的段标记张量。
    valid_lens : list[Tensor]
        所有样本的有效长度。
    all_pred_positions : list[Tensor]
        所有样本的MLM预测位置。
    all_mlm_weights : list[Tensor]
        所有样本的MLM损失权重。
    all_mlm_labels : list[Tensor]
        所有样本的MLM预测标签。
    nsp_labels : list[Tensor]
        所有样本的NSP标签。
    """

    def __init__(self, paragraphs, max_len):
        # 输入paragraphs[i]是代表段落的句子字符串列表；
        # 而输出paragraphs[i]是代表段落的句子列表，其中每个句子都是词元列表
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # 获取下一句子预测任务的数据
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # 获取遮蔽语言模型任务的数据
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # 填充输入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        """根据索引获取单个预训练样本。

        Parameters
        ----------
        idx : int
            样本索引（形参）。

        Returns
        -------
        tuple
            包含7个张量的元组：
            (token_ids, segments, valid_len, pred_positions,
             mlm_weights, mlm_labels, nsp_label)
        """
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.all_token_ids)


# ============================================================
# 第五部分：数据加载入口函数
# ============================================================

def load_data_wiki(batch_size, max_len):
    """下载WikiText-2数据集并生成BERT预训练数据迭代器。

    这是整个模块的入口函数，串联了所有数据处理步骤。

    Parameters
    ----------
    batch_size : int
        小批量大小（形参）。实参示例：512
    max_len : int
        BERT输入序列的最大长度（形参）。实参示例：64

    Returns
    -------
    train_iter : DataLoader
        预训练数据的迭代器，每个批次包含7个张量。
    vocab : d2l.Vocab
        从语料库构建的词表对象。
    """
    num_workers = d2l.get_dataloader_workers()
    # 优先使用本地数据目录，如果不存在则尝试下载
    local_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'data', 'wikitext-2')
    if os.path.exists(os.path.join(local_data_dir, 'wiki.train.tokens')):
        data_dir = local_data_dir
    else:
        data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab


# ============================================================
# 第六部分：主程序入口
# ============================================================

if __name__ == '__main__':
    # 设置批量大小和最大序列长度
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)

    # 打印一个小批量的形状
    for (tokens_X, segments_X, valid_lens_x, pred_positions_X,
         mlm_weights_X, mlm_Y, nsp_y) in train_iter:
        print('tokens_X.shape:', tokens_X.shape)
        print('segments_X.shape:', segments_X.shape)
        print('valid_lens_x.shape:', valid_lens_x.shape)
        print('pred_positions_X.shape:', pred_positions_X.shape)
        print('mlm_weights_X.shape:', mlm_weights_X.shape)
        print('mlm_Y.shape:', mlm_Y.shape)
        print('nsp_y.shape:', nsp_y.shape)
        break

    # 打印词表大小
    print(f'\n词表大小: {len(vocab)}')
