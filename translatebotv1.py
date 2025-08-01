import gradio as gr
from transformers import pipeline

# Initialize pipelines for both directions
modelname = "facebook/nllb-200-distilled-600M"
recorder = ""
translator_zh_en = pipeline(
    task="translation",
    model=modelname,
    tokenizer=modelname,
    src_lang="zho_Hans", tgt_lang="eng_Latn", device=0
)
translator_en_zh = pipeline(
    task="translation",
    model=modelname,
    tokenizer=modelname,
    src_lang="eng_Latn", tgt_lang="zho_Hans", device=0
)

def do_translate(text, direction):
    if direction == "Chinese → English":
        out = translator_zh_en(text, max_length=512)
    else:
        out = translator_en_zh(text, max_length=512)
    return out[0]["translation_text"]

def on_select(evt: gr.SelectData):
    return evt.value

def recording(text,direction):
    global recorder
    if direction == "Chinese → English":
        out = translator_zh_en(text, max_length=512)
    else:
        out = translator_en_zh(text, max_length=512)
    if not recorder:
        recorder+=f"{text}->{out[0]["translation_text"]}"
    else:
        recorder+=f"\n{text}->{out[0]["translation_text"]}"
    return recorder


with gr.Blocks() as demo:
    gr.Markdown("## Chinese ↔ English Translator")
    with gr.Row():
        direction = gr.Radio(
            choices=["Chinese → English", "English → Chinese"],
            value="Chinese → English",
            label="Translation Direction"
        )
    input_text = gr.Textbox(label="Source Text", lines=5, placeholder="输入中文或英文段落…")
    output_text = gr.Textbox(label="Translated Text", lines=5)
    sel_storage = gr.Textbox(visible=False, label="selection")
    output_text2 = gr.Textbox(label="recorded tex", lines=20, show_copy_button=True)
    translate_btn = gr.Button("Translate")
    record_btn = gr.Button("record")
    
    
    input_text.select(on_select, None, sel_storage)

    translate_btn.click(
        fn=do_translate,
        inputs=[input_text, direction],
        outputs=output_text
    )
    
    
    record_btn.click(
        fn=recording,
        inputs=[sel_storage, direction],
        outputs=output_text2
    )
demo.launch(inbrowser=True)