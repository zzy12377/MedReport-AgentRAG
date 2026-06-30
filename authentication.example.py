# -*- coding: utf-8 -*-
"""
authentication.example.py

MedRAG 本地运行配置文件（示例）。
将此文件重命名为 authentication.py 并填入你自己的 API Key。

注意：不要把真实 API Key 发给别人，也不要上传到公开仓库。
"""

# 数据路径
ob_path = "./dataset/df/train"
test_folder_path = "./dataset/df/test"
ground_truth_file_path = "./dataset/AI Data Set with Categories.csv"
augmented_features_path = "./dataset/knowledge graph of DDXPlus.xlsx"

# SiliconFlow 配置
# 在这里填你自己的硅基流动 API Key，例如：api_key = "sk-xxxxxxxx"
api_key = "your_siliconflow_api_key_here"
base_url = "https://api.siliconflow.cn/v1"

# 模型配置
chat_model = "Qwen/Qwen3-8B"
embedding_model = "BAAI/bge-m3"

# SiliconFlow embedding 出错时，本地 sentence-transformers 回退模型。
# 小模型下载更快；如果你更想本地也用 bge-m3，可以把第一项改成 "BAAI/bge-m3"。
local_embedding_model = "BAAI/bge-small-en-v1.5"
local_embedding_fallback_models = [
    local_embedding_model,
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-m3",
]

# Hugging Face Token（当前不用，先留空）
hf_token = ""

# 可选配置（以下都有默认值，一般不需要改）：
# local_embedding_cache_dir = "./models/embedding"
# local_embedding_device = "cpu"
# local_embedding_batch_size = 16
# local_embedding_max_memory_gb = 16.0
# auto_install_sentence_transformers = True
