# -*- coding: utf-8 -*-
"""Build a Chinese DDXPlus mirror without overwriting the original dataset.

The script is intentionally offline and deterministic. It keeps original labels
for evaluation while adding Chinese display fields for retrieval and reports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


DISEASE_ZH = {
    "URTI": "上呼吸道感染",
    "Bronchitis": "支气管炎",
    "Pneumonia": "肺炎",
    "Bronchiectasis": "支气管扩张",
    "Tuberculosis": "结核病",
    "Influenza": "流行性感冒",
    "HIV (initial infection)": "HIV 初次感染",
    "Chagas": "恰加斯病",
    "Stable angina": "稳定型心绞痛",
    "Unstable angina": "不稳定型心绞痛",
    "Possible NSTEMI / STEMI": "疑似非 ST 段抬高型或 ST 段抬高型心肌梗死",
    "Myocarditis": "心肌炎",
    "Atrial fibrillation": "心房颤动",
    "Panic attack": "惊恐发作",
    "Anemia": "贫血",
    "GERD": "胃食管反流病",
    "Localized edema": "局部水肿",
    "Pulmonary embolism": "肺栓塞",
    "Anaphylaxis": "过敏性休克",
    "SLE": "系统性红斑狼疮",
    "Acute rhinosinusitis": "急性鼻窦炎",
    "Acute otitis media": "急性中耳炎",
    "Viral pharyngitis": "病毒性咽炎",
    "Epiglottitis": "会厌炎",
    "Acute laryngitis": "急性喉炎",
    "COVID-19": "新型冠状病毒感染",
    "Boerhaave": "Boerhaave 综合征",
    "Spontaneous pneumothorax": "自发性气胸",
    "Sarcoidosis": "结节病",
    "Cluster headache": "丛集性头痛",
    "Migraine": "偏头痛",
    "Meningitis": "脑膜炎",
    "Stroke": "卒中",
    "TIA": "短暂性脑缺血发作",
    "Hypoglycemia": "低血糖",
    "Hyperthyroidism": "甲状腺功能亢进",
    "Hypothyroidism": "甲状腺功能减退",
    "Acute dystonic reactions": "急性肌张力障碍反应",
}

DISEASE_ZH.update(
    {
        "Acute COPD exacerbation / infection": "急性慢阻肺加重或感染",
        "Acute pulmonary edema": "急性肺水肿",
        "Allergic sinusitis": "过敏性鼻窦炎",
        "Bronchospasm / acute asthma exacerbation": "支气管痉挛或急性哮喘加重",
        "Chronic rhinosinusitis": "慢性鼻窦炎",
        "Croup": "哮吼",
        "Ebola": "埃博拉病毒病",
        "Guillain-Barré syndrome": "吉兰-巴雷综合征",
        "Inguinal hernia": "腹股沟疝",
        "Larygospasm": "喉痉挛",
        "Myasthenia gravis": "重症肌无力",
        "Pancreatic neoplasm": "胰腺肿瘤",
        "Pericarditis": "心包炎",
        "PSVT": "阵发性室上性心动过速",
        "Pulmonary neoplasm": "肺部肿瘤",
        "Scombroid food poisoning": "鲭鱼中毒",
        "Spontaneous rib fracture": "自发性肋骨骨折",
        "Whooping cough": "百日咳",
    }
)


PHRASE_ZH = {
    "Avez-vous objectivé ou ressenti de la fièvre?": "是否出现或感觉发热",
    "Habitez-vous avec 4 personnes ou plus?": "是否与 4 人或更多人共同居住",
    "Avez-vous eu des sueurs importantes?": "是否出现明显出汗",
    "Avez-vous de la douleur à quelque part en lien avec votre raison de consultation?": "是否有与本次就诊原因相关的疼痛",
    "Caractérisez votre douleur:": "疼痛性质",
    "Avez-vous de la douleur quelque part?": "疼痛部位",
    "Quelle est l’intensité de la douleur?": "疼痛强度",
    "Quelle est l'intensité de la douleur?": "疼痛强度",
    "Est-ce que la douleur se propage vers un autre endroit?": "疼痛是否向其他部位放射",
    "À quel point la douleur est-elle précisément localisée?": "疼痛定位是否明确",
    "A quelle vitesse la douleur est-elle apparue ?": "疼痛出现速度",
    "À quelle vitesse la douleur est-elle apparue ?": "疼痛出现速度",
    "Avez-vous une toux produisant des crachats colorés ou plus abondants qu’habituellement?": "是否有咳嗽并伴有有色或较平时更多的痰",
    "Fumez-vous la cigarette quotidiennement?": "是否每天吸烟",
    "Avez-vous mal à la gorge?": "是否咽痛",
    "Avez-vous de la toux?": "是否咳嗽",
    "Avez-vous voyagé dans les 4 dernières semaines?": "过去 4 周内是否旅行",
    "Êtes-vous en contact régulièrement avec de la fumée de cigarette sans être vous-même fumeur?": "是否经常接触二手烟",
    "Vous sentez-vous essoufflé ou avez vous de la difficulté à respirer de façon importante?": "是否明显气短或呼吸困难",
    "Avez-vous visualisé du sang dans vos crachats?": "痰中是否见血",
    "Souffrez-vous de diabète?": "是否患有糖尿病",
    "Souffrez-vous d’hypertension ou prenez-vous des médicaments pour traiter la haute pression?": "是否患有高血压或服用降压药",
    "Souffrez-vous d’hypercholestérolémie ou prenez-vous des médicaments pour traiter un taux de cholestérol élevé?": "是否患有高胆固醇血症或服用降脂药",
    "Prenez-vous de l’alcool de façon excessive ou avez-vous une dépendance à l’alcool?": "是否过量饮酒或存在酒精依赖",
    "Avez-vous déjà fait une crise de coeur ou faites-vous de l’angine?": "是否曾发生心脏病发作或心绞痛",
    "Faites-vous de l’activité physique régulièrement, soit 4 fois par semaine ou plus?": "是否规律运动，每周 4 次或以上",
    "Avez-vous des symptômes qui sont pires lors de l’effort physique et soulagés par le repos?": "症状是否在体力活动时加重并在休息后缓解",
    "Avez-vous des membres de votre famille proche qui ont eu un problème de maladie cardiovasculaire avant l’âge de 50 ans?": "近亲中是否有人 50 岁前发生心血管疾病",
    "Avez-vous de l’enflure à un ou plusieurs endroits sur votre corps?": "身体一个或多个部位是否肿胀",
    "Êtes-vous connu pour un problème au niveau d’une valve cardiaque?": "是否已知存在心脏瓣膜问题",
    "Avez-vous pris du poids récemment?": "近期是否体重增加",
    "Êtes-vous atteint d’insuffisance cardiaque?": "是否患有心力衰竭",
    "Avez-vous déjà fait une thrombophlébite profonde?": "是否曾发生深静脉血栓性静脉炎",
    "Prenez-vous un médicaments de la classe des bloqueurs des canaux calciques?": "是否服用钙通道阻滞剂类药物",
    "À quel endroit est situé l’enflure?": "肿胀部位",
    "Avez-vous une maladie endocrinienne ou un dysfonctionnement hormonal?": "是否存在内分泌疾病或激素功能异常",
    "A-t-on déjà diagnostiqué chez vous un syndrome d’apnées hypopnées obstructives du sommeil (SAHOS)?": "是否曾被诊断阻塞性睡眠呼吸暂停低通气综合征",
}

PHRASE_ZH["Caractérisez votre douleur"] = "疼痛性质"


WORD_ZH = [
    ("sensitive", "敏感样"),
    ("heavy", "沉重样"),
    ("sharp", "锐痛"),
    ("tedious", "钝痛"),
    ("burning", "烧灼样"),
    ("temple", "太阳穴"),
    ("shoulder", "肩部"),
    ("epigastric", "上腹部"),
    ("right", "右侧"),
    ("left", "左侧"),
    ("daily", "每日"),
]


def _natural_sort_key(path: Path) -> List[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def disease_zh(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = text.replace("_", " ").strip()
    exact = DISEASE_ZH.get(text) or DISEASE_ZH.get(normalized)
    if exact:
        return exact
    lowered = normalized.lower()
    for key, zh in DISEASE_ZH.items():
        if key.lower() == lowered:
            return zh
    compact = re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()
    for key, zh in DISEASE_ZH.items():
        if re.sub(r"[^a-z0-9]+", " ", key.lower()).strip() == compact:
            return zh
    return "未映射疾病"


def translate_fragment(text: Any) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    compact = re.sub(r"[:：]+", ":", value)
    key, sep, tail = compact.partition(":")
    base = PHRASE_ZH.get(key.strip()) or PHRASE_ZH.get(value.strip())
    if base is None:
        for phrase, zh in PHRASE_ZH.items():
            if key.strip().startswith(phrase.rstrip(":?？")) or value.startswith(phrase.rstrip(":?？")):
                base = zh
                break
    translated = base if base else value
    if sep and tail:
        translated = f"{translated}：{translate_value(tail.strip())}"
    translated = _replace_words(translated)
    if _looks_untranslated_question(translated):
        return "症状问诊项"
    if _has_latin_or_french(translated) and not _has_chinese(translated):
        return "症状问诊项"
    return translated


def translate_value(value: str) -> str:
    if re.fullmatch(r"V_\d+", value):
        return f"编码 {value}"
    return _replace_words(value)


def _replace_words(text: str) -> str:
    result = text
    for src, dst in WORD_ZH:
        result = re.sub(re.escape(src), dst, result, flags=re.IGNORECASE)
    result = result.replace("(L)", "（左）").replace("(R)", "（右）")
    return result


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _has_latin_or_french(text: str) -> bool:
    return bool(re.search(r"[A-Za-zÀ-ÿ]", text or ""))


def _looks_untranslated_question(text: str) -> bool:
    value = str(text or "")
    return bool(re.search(r"\b(Avez|Êtes|Est-ce|Quelle|Quel|Souffrez|Prenez|Fumez|Habitez|vous|votre)\b", value, re.I))


def translate_symptoms(value: Any) -> str:
    fragments = [part.strip() for part in str(value or "").split(";") if part.strip()]
    return "；".join(translate_fragment(part) for part in fragments)


def translate_differential(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    rows = []
    for part in [item.strip() for item in text.split(";") if item.strip()]:
        match = re.match(r"(?P<name>.+?)\s*\((?P<score>[0-9.]+)\)\s*$", part)
        if match:
            rows.append(f"{disease_zh(match.group('name').strip())}（{match.group('score')}）")
        else:
            rows.append(disease_zh(part))
    return "；".join(rows)


def sex_zh(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text == "M":
        return "男"
    if text == "F":
        return "女"
    return str(value or "")


def build_text_zh(data: Dict[str, Any]) -> str:
    parts = []
    if data.get("Age"):
        parts.append(f"年龄：{data.get('Age')} 岁")
    if data.get("Sex"):
        parts.append(f"性别：{sex_zh(data.get('Sex'))}")
    if data.get("Symptoms_zh"):
        parts.append(f"症状与病史：{data.get('Symptoms_zh')}")
    if data.get("Differential Diagnosis_zh"):
        parts.append(f"鉴别诊断候选：{data.get('Differential Diagnosis_zh')}")
    if data.get("Diagnosis_zh"):
        parts.append(f"最终诊断：{data.get('Diagnosis_zh')}")
    return "\n".join(parts)


def convert_case_file(src: Path, dst: Path) -> bool:
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] 跳过 {src}: {exc}")
        return False

    out = dict(data)
    out.setdefault("Text_original", data.get("Text", ""))
    out.setdefault("Symptoms_original", data.get("Symptoms", ""))
    out.setdefault("Diagnosis_original", data.get("Diagnosis") or data.get("Processed Diagnosis") or "")
    out.setdefault("Differential Diagnosis_original", data.get("Differential Diagnosis", ""))
    out["Sex_zh"] = sex_zh(data.get("Sex"))
    out["Symptoms_zh"] = translate_symptoms(data.get("Symptoms", ""))
    out["Diagnosis_zh"] = disease_zh(out.get("Diagnosis_original", ""))
    out["Differential Diagnosis_zh"] = translate_differential(data.get("Differential Diagnosis", ""))
    out["Text_zh"] = build_text_zh(out)
    out["Text"] = out["Text_zh"]

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def convert_split(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.is_dir():
        print(f"[WARN] 数据目录不存在：{src_dir}")
        return 0
    count = 0
    for src in sorted(src_dir.glob("*.json"), key=_natural_sort_key):
        if convert_case_file(src, dst_dir / src.name):
            count += 1
    return count


def convert_kg(src_path: Path, dst_path: Path) -> int:
    if not src_path.exists():
        print(f"[WARN] KG 文件不存在：{src_path}")
        return 0
    try:
        import pandas as pd
    except Exception as exc:
        print(f"[WARN] 当前环境无法读取 Excel KG，已跳过 KG 中文 JSONL：{exc}")
        return 0
    try:
        df = pd.read_excel(src_path, usecols=["subject", "relation", "object"])
    except Exception as exc:
        print(f"[WARN] KG Excel 读取失败，已跳过：{exc}")
        return 0

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with dst_path.open("w", encoding="utf-8") as f:
        for _, row in df.dropna(subset=["subject", "relation", "object"]).iterrows():
            head = str(row["subject"]).strip()
            relation = str(row["relation"]).strip()
            tail = str(row["object"]).strip()
            head_zh = disease_zh(head)
            relation_zh = translate_relation(relation)
            tail_zh = translate_kg_text(tail)
            item = {
                "head": head_zh,
                "relation": relation_zh,
                "tail": tail_zh,
                "head_original": head,
                "relation_original": relation,
                "tail_original": tail,
                "text": f"{head_zh} {relation_zh} {tail_zh}",
                "source": "DDXPlus_KG_ZH",
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def translate_relation(value: Any) -> str:
    text = str(value or "").replace("_", " ").strip()
    mapping = {
        "has symptomatology": "症状表现为",
        "has anamnesis": "病史提示",
        "has risk factor": "危险因素为",
        "has differential diagnosis": "鉴别诊断包括",
        "is associated with": "相关于",
        "has lifestyle": "生活方式相关",
        "has therapy": "治疗方式包括",
        "has treatment": "治疗方式包括",
        "has complication": "并发症包括",
        "has biological": "生物学相关",
    }
    translated = mapping.get(text.lower(), _replace_words(text))
    if _has_latin_or_french(translated) and not _has_chinese(translated):
        return "相关信息"
    return translated


def translate_kg_text(value: Any) -> str:
    text = _replace_words(str(value or "").strip())
    replacements = [
        ("Worsening shortness of breath", "气短加重"),
        ("chronic cough with sputum production", "慢性咳嗽伴咳痰"),
        ("Increased sputum purulence and volume", "痰液脓性和痰量增加"),
        ("Common in smokers or individuals exposed to pollutants", "常见于吸烟者或污染物暴露人群"),
        ("History of chronic obstructive pulmonary disease", "慢性阻塞性肺疾病病史"),
        ("Bronchodilators and antibiotics if bacterial infection is present", "如存在细菌感染，可使用支气管扩张剂和抗生素"),
        ("Persistent cough lasting more than 3 weeks", "持续超过 3 周的咳嗽"),
        ("night sweats", "夜间盗汗"),
        ("weight loss", "体重下降"),
        ("Sore throat", "咽痛"),
        ("nasal congestion", "鼻塞"),
        ("cough", "咳嗽"),
        ("fever", "发热"),
        ("pain", "疼痛"),
        ("fatigue", "乏力"),
        ("shortness of breath", "气短"),
        ("smoking", "吸烟"),
        ("pollutants", "污染物"),
    ]
    for src, dst in replacements:
        text = re.sub(re.escape(src), dst, text, flags=re.IGNORECASE)
    if _has_latin_or_french(text) and not _has_chinese(text):
        return "知识图谱描述项"
    if _looks_untranslated_question(text):
        return "知识图谱描述项"
    return text or "知识图谱描述项"


def copy_readme(output_dir: Path, train_count: int, test_count: int, kg_count: int) -> None:
    readme = f"""# DDXPlus 中文镜像数据

本目录由 `python scripts/build_ddxplus_zh_dataset.py` 生成。

## 内容

- `df/train`: 中文化训练病例，共 {train_count} 条
- `df/test`: 中文化测试病例，共 {test_count} 条
- `kg/ddxplus_kg_zh.jsonl`: 中文化知识图谱三元组，共 {kg_count} 条

## 设计说明

原始 `dataset` 目录不会被覆盖。每条病例保留 `*_original` 字段用于复现实验和 Recall 评估，同时新增：

- `Text_zh`: 中文检索文本
- `Symptoms_zh`: 中文症状/病史文本
- `Diagnosis_zh`: 中文诊断展示名
- `Differential Diagnosis_zh`: 中文鉴别诊断候选

当前脚本使用离线医学词典和规则翻译，适合课程演示和中文检索镜像构建。若需要逐句高质量翻译，可后续接入 Qwen/GLM/Ollama 或云端 LLM 翻译，并继续保留原始字段。
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Chinese mirror files for DDXPlus train/test/KG.")
    parser.add_argument("--dataset-dir", default="./dataset")
    parser.add_argument("--output-dir", default="./data_zh")
    parser.add_argument("--clear", action="store_true", help="Clear output dir before rebuilding.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    if args.clear and output_dir.exists():
        shutil.rmtree(output_dir)

    train_count = convert_split(dataset_dir / "df" / "train", output_dir / "df" / "train")
    test_count = convert_split(dataset_dir / "df" / "test", output_dir / "df" / "test")
    kg_count = convert_kg(dataset_dir / "knowledge graph of DDXPlus.xlsx", output_dir / "kg" / "ddxplus_kg_zh.jsonl")
    copy_readme(output_dir, train_count, test_count, kg_count)

    print(json.dumps({
        "status": "done",
        "output_dir": str(output_dir),
        "train_cases": train_count,
        "test_cases": test_count,
        "kg_triples": kg_count,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
