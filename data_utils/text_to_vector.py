import os
import sys
import torch
import torch.nn.functional as F
from transformers import (
    logging,
    CLIPTokenizer, CLIPModel,
    BertTokenizer, BertModel,
    AutoTokenizer, AutoModel
)
from sentence_transformers import SentenceTransformer

from data_utils.constants.label_text_mapper import getLabelMapper

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _extract_tensor(output, name="unknown"):
    """
    把不同模型返回的输出对象，统一抽成张量。
    """
    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds

    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output

    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state[:, 0, :]

    raise TypeError(
        f"{name} 输出类型无法转换为张量，实际类型为：{type(output)}"
    )


def label_to_vector(dataset="seed", LM="bert", onehot=False,
                    LabelTextMapper=None, device="cuda"):
    """
    将类别文本映射成向量表示

    支持的文本编码器：
        - clip
        - bert
        - sbert
        - roberta_go

    返回：
        vector_dim: 向量维度
        all_text_embs: {class_id: numpy_vector}
    """
    logging.set_verbosity_error()
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    if LabelTextMapper is None:
        label_map = getLabelMapper(dataset, onehot)
    else:
        label_map = LabelTextMapper

    cids = list(label_map.keys())
    texts = list(label_map.values())

    print(f"[{LM.upper()}] 提取文本向量中 (device: {device})")

    model = None
    tokenizer = None

    if LM.lower() == "clip":
        model_path = r"D:/大学/脑机接口/local_clip_model"

        tokenizer = CLIPTokenizer.from_pretrained(model_path)
        model = CLIPModel.from_pretrained(model_path).to(device)
        model.eval()

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = _move_batch_to_device(inputs, device)

        with torch.no_grad():
            output = model.get_text_features(**inputs)
            embeds = _extract_tensor(output, name="clip")

    elif LM.lower() == "bert":
        model_path = r"D:/大学/脑机接口/bert-base-uncased"

        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertModel.from_pretrained(model_path).to(device)
        model.eval()

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = _move_batch_to_device(inputs, device)

        with torch.no_grad():
            output = model(**inputs)
            embeds = _extract_tensor(output, name="bert")

    elif LM.lower() == "sbert":
        model_name = "all-MiniLM-L6-v2"

        model = SentenceTransformer(model_name, device=str(device))
        embeds = model.encode(
            texts,
            convert_to_tensor=True,
            device=str(device)
        )

        if not isinstance(embeds, torch.Tensor):
            embeds = torch.tensor(embeds, dtype=torch.float32, device=device)

    elif LM.lower() == "roberta_go":
        model_name = "SamLowe/roberta-base-go_emotions"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = _move_batch_to_device(inputs, device)

        with torch.no_grad():
            output = model(**inputs)
            embeds = _extract_tensor(output, name="roberta_go")

    else:
        raise ValueError(
            f"不支持的 LM 类型: {LM}（可选: clip / bert / sbert / roberta_go）"
        )

    embeds = embeds.float()
    embeds = F.normalize(embeds, p=2, dim=-1)

    vector_dim = embeds.shape[-1]
    all_text_embs = {}

    for i, cid in enumerate(cids):
        all_text_embs[cid] = embeds[i].detach().cpu().numpy()

    del model
    if tokenizer is not None:
        del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vector_dim, all_text_embs