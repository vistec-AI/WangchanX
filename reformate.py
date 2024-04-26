from datasets import load_dataset
import argparse
import json
from pathlib import Path
import logging
import os
import random
PATTERNS = {
    "iapp_wiki_qa_squad": [
       {
            "instruction": "Instruction: จงอ่านบริบท และตอบคำถาม โดยจะคำตอบ ต้องมาจากบริบท ตอบสั้นๆ",
            "input": "บริบท: {context}\nคำถาม: {question}",
            "output": "{answers}"
        },
        {
            "instruction": "From the context, Respond the question in short span.",
            "input": "Context: {context}\nQuestion: {question}",
            "output": "{answers}"
        },
        {
            "instruction": "กำหนดบทความพื้นหลังให้ แล้วตอบสั้นๆ",
            "input": "พื้นหลัง: {context}\nจงตอบคำถาม: {question}",
            "output": "{answers}"
        },
        {
            "instruction": "Read the context and answer the question in one or few words.",
            "input": "Context: {context}\nQuestion: '{question}'",
            "output": "{answers}"
        },
        {
            "instruction": "From Background, Please answer this question in short span: {question}",
            "input": "Background: {context}",
            "output": "{answers}"
        },
        {
            "instruction": "This is extractive question answering task. So, answer in short span.",
            "input": "Background: {context}\n\nQuestion: {question}",
            "output": "{answers}"
        },
        {
            "instruction": "อ่านบริบท แล้วตอบคำถามนี้สั้นๆ: {question}",
            "input": "บริบท: {context}",
            "output": "{answers}"
        },
        {
            "instruction": "จากเนื้อหา จงตอบคำถามนี้สั้นๆ: {question}",
            "input": "เนื้อหา: {context}",
            "output": "{answers}"
        },
        {
            "instruction": "อ่านและทำความเข้าใจ บทความก่อนที่จะตอบคำถาม จากบทความนั้น โดยตอบเพียงแค่ไม่กี่คำ",
            "input": "บทความ: {context}\n\nQ: {question}",
            "output": "{answers}"
        },
        {
            "instruction": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\nInstruction:\nAnswer the question according to the context in few words.",
            "input": "Context:\n{context}\nQuestion:\n{question}",
            "output": "{answers}"
        },
        {
            "instruction": "นายคือผู้ช่วยฉัน ในการอ่านข้อความ แล้วตอบคำถามออกมาให้ถูกต้อง กระชับ สั้นและตรงประเด็น โดยคำตอบจะอยู่ในเนื้อหา บทความ นายต้องอ่านให้รอบคอบ และตอบให้ถูกต้องครบถ้วน เพราะนายเก่งในการตอบคำถาม",
            "input": "เนื้อหาบทความ: {context}\n\nQuestion: จากเนื้อหาบทความ คำถามคือ '{question}'",
            "output": "{answers}"
        },
    ],
    "math_14k": [
        {
            "instruction": "Question: {instruction}",
            "input": "{input}",
            "output": "{answer}"
        },
        {
            "instruction": "help me to solve maths.",
            "input": "{instruction}",
            "output": "{answer}"
        },
        {
            "instruction": "แก้สมการคณิตศาสตร์ให้หน่อย",
            "input": "Instruction: {instruction}",
            "output": "{answer}"
        },
        {
            "instruction": "{instruction} ",
            "input": "จงแสดงวิธีการแก้ปัญหานี้",
            "output": "{answer}"
        },
        {
            "instruction": "I want you to act as a math teacher. I will provide some mathematical equations or concepts, and it will be your job to explain them in easy-to-understand terms. This could include providing step-by-step instructions for solving a problem.",
            "input": "Question: {instruction}",
            "output": "{answer}"
        },
        {
            "instruction": "แสดงวิธีทำ วิธีคิด ในการแก้ไขปัญหานี้",
            "input": "Problem: {instruction}",
            "output": "{answer}"
        },
        {
            "instruction": "{instruction}",
            "input": "คำสั่ง: แก้ปัญหานี้ให้ที แสดงวิธีทำทีละขั้นตอน",
            "output": "{answer}"
        },
        {
            "instruction": "Give me the answer for this problem.",
            "input": "โจทย์ปัญหา: {instruction}",
            "output": "{answer}"
        },
        {
            "instruction": "let's solve this math step by step.",
            "input": "{instruction}",
            "output": "{answer}"
        },
    ]
}
def generate_instruction_dataset(sample,TEMPLATES):
        template = random.sample(TEMPLATES, k=1)[0]
        return {
            "instruction": template["instruction"].format(**sample),
            "input": template["input"].format(**sample) if template["input"] is not None else None,
            "output": template["output"].format(**sample)
        }

def setup_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='reformatted')
    parser.add_argument('--data', type=str, default='Rasu23/iapp_wiki_qa_squad_cleaned')
    return parser.parse_args()

def reformat_rawdataset_iapp(examples):
    TEMPLATES = PATTERNS["iapp_wiki_qa_squad"]
    examples["context"] =  examples["context"]
    examples["question"] =  examples["question"]
    examples["answers"] =  examples["answers_text"]
    a = generate_instruction_dataset(examples,TEMPLATES)

    comb = a["instruction"]
    if a["input"] != "" and a["input"] != None:
        comb = a["input"] +"\n\n" + a["instruction"]

    a["messages"] = [
        {"content":  comb.strip() , "role": "user"},
        {"content": a["output"], "role": "assistant"},
    ]
    return a

def reformat_rawdataset_math(examples):
    TEMPLATES = PATTERNS["math_14k"]
    examples["input"] =  examples["context"]
    examples["instruction"] =  examples["instruction"]
    examples["answer"] =  examples["answer"]
    a = generate_instruction_dataset(examples,TEMPLATES)

    comb = a["instruction"]
    if a["input"] != "" and a["input"] != None:
        comb = a["input"] +"\n\n" + a["instruction"]

    a["messages"] = [
        {"content":  comb.strip() , "role": "user"},
        {"content": a["output"], "role": "assistant"},
    ]
    return a
def save_to_json(data, path):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4,ensure_ascii=False)
    except IOError as e:
        logging.error(f"Error saving data to {path}: {e}")

def main():
    args = setup_arg_parser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data)
    reformat_data = dataset["train"].filter(lambda example: example["question"] != "" and example["question"] !=None and example["answers_text"] != "" and example["answers_text"] !=None ).map(reformat_rawdataset_iapp)
    reformat_data.to_json(output_dir / "iapp_train.json") 

if __name__ == "__main__":
    main()