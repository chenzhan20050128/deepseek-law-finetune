# 导入 json 模块，用于处理 JSON 格式的数据
import json

# 这个脚本的目的是将来自不同来源、具有不同字段结构的法律问答数据集，
# 统一处理成标准的 `{ "question": "...", "answer": "..." }` 格式。
# 这种标准化的数据格式便于后续的微调流程统一处理。

# --- 数据解析函数定义 ---
# 每个函数对应一种特定格式的数据集文件。

def parse_92k_data(data_path):
    """
    解析 "answer_with_law_92k.json" 数据集。
    这个数据集的字段已经是 'question' 和 'answer'。
    """
    with open(data_path, encoding='utf8') as f:  # 以 utf8 编码打开文件
        data = json.load(f)  # 加载 JSON 数据

    parsed_data = []  # 初始化一个列表来存储解析后的数据
    for d in data:
        ret = {
            'question': d['question'],  # 直接提取 'question' 字段
            'answer': d['answer'],      # 直接提取 'answer' 字段
        }
        parsed_data.append(ret)

    return parsed_data


def parse_52k_data(data_path):
    """
    解析 "CrimeKgAssitant_after_clean_52k.json" 数据集。
    这个数据集使用 'input' 和 'output' 作为字段名。
    """
    with open(data_path, encoding='utf8') as f:
        data = json.load(f)

    parsed_data = []
    for d in data:
        ret = {
            'question': d['input'],   # 将 'input' 字段映射到 'question'
            'answer': d['output'],  # 将 'output' 字段映射到 'answer'
        }
        parsed_data.append(ret)

    return parsed_data


def parse_fakao_data(data_path):
    """
    解析 "fakao_gpt4.json" 数据集。
    这个数据集的 'input' 字段包含了 "Question:" 前缀，需要移除。
    """
    with open(data_path, encoding='utf8') as f:
        data = json.load(f)

    parsed_data = []
    for d in data:
        ret = {
            # 移除 'input' 字段值中的 "Question:" 前缀和两端可能存在的空格
            'question': d['input'].strip('Question:').strip(),
            'answer': d['output'],
        }
        parsed_data.append(ret)

    return parsed_data


def parse_zixun_data(data_path):
    """
    解析 "zixun_gpt4.json" 数据集。
    这个数据集使用 'query' 和 'response' 作为字段名。
    """
    with open(data_path, encoding='utf8') as f:
        data = json.load(f)

    parsed_data = []
    for d in data:
        ret = {
            'question': d['query'],     # 将 'query' 字段映射到 'question'
            'answer': d['response'],  # 将 'response' 字段映射到 'answer'
        }
        parsed_data.append(ret)

    return parsed_data


# --- 主处理流程 ---

# `parser_config` 是一个配置字典，它将数据文件的路径映射到对应的解析函数。
# 通过修改这个字典，可以灵活地选择要处理哪些文件。
#
# 示例1：处理训练数据集
# parser_config = {
#     '/path/to/your/answer_with_law_92k.json': parse_92k_data,
#     '/path/to/your/CrimeKgAssitant_after_clean_52k.json': parse_52k_data,
#     '/path/to/your/fakao_gpt4.json': parse_fakao_data,
# }
# processed_data_save_path = '/path/to/your/finetune_processed_train.json'

# 示例2：处理评估数据集 (当前脚本激活的配置)
# 注意：这里的路径是示例路径，你需要根据你的实际文件位置进行修改。
parser_config = {
    '/root/autodl-tmp/ds_finetune-main/zixun_gpt4.json': parse_zixun_data,
}
# 定义处理完成后保存的文件路径
processed_data_save_path = '/root/autodl-tmp/ds_finetune-main/finetune_processed_eval.json'

# 初始化一个列表，用于汇总所有处理过的数据
processed_data = []

# 遍历配置字典，对每个文件执行相应的解析函数
for data_path, func in parser_config.items():
    print(f'processing data from {data_path}')
    try:
        parsed_data = func(data_path)  # 调用解析函数
        print(f'get {len(parsed_data)} records')
        processed_data.extend(parsed_data)  # 将解析后的数据追加到总列表中
    except FileNotFoundError:
        print(f"Warning: File not found at {data_path}. Skipping.")
    except Exception as e:
        print(f"An error occurred while processing {data_path}: {e}")


# 打印处理后的总记录数
print(f'total {len(processed_data)} processed records')

# 将汇总后的数据写入一个新的 JSON 文件
# 使用 'w' 模式（写入），如果文件已存在则会覆盖
# `encoding='utf8'` 确保中文字符能正确写入
# `ensure_ascii=False` 允许直接写入非 ASCII 字符（如中文），而不是转义成 \uXXXX
# `indent=4` 使输出的 JSON 文件格式化，有 4 个空格的缩进，更易于阅读
with open(processed_data_save_path, 'w', encoding='utf8') as f:
    json.dump(processed_data, f, ensure_ascii=False, indent=4)
    print(f'processed data saved to {processed_data_save_path}')
