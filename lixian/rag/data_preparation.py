import jieba, json, pdfplumber

# 对长文本进行切分
def split_text_fixed_size(text, chunk_size, overlap_size):
    new_text = []
    for i in range(0, len(text), chunk_size):
        if i == 0:
            new_text.append(text[0:chunk_size])
        else:
            new_text.append(text[i - overlap_size:i + chunk_size])
            # new_text.append(text[i:i + chunk_size])
    return new_text

def read_data(query_data_path, knowledge_data_path):
    with open(query_data_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    pdf = pdfplumber.open(knowledge_data_path)
    # 标记当前页与其文本知识
    pdf_content = []
    for page_idx in range(len(pdf.pages)):
        text = pdf.pages[page_idx].extract_text()
        new_text = split_text_fixed_size(text, chunk_size=100, overlap_size=5)
        for chunk_text in new_text:
            pdf_content.append({
                'page'   : 'page_' + str(page_idx + 1),
                'content': chunk_text
            })
    return questions, pdf_content
