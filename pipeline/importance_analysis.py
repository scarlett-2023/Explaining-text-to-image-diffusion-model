import os, importlib.util, spacy, torch, clip
from config import TXT_PATH, BASE_MODELS, OUTPUT_ROOT, K_IMPORTANCE

def get_high_importance_func_indices(text, main_batch, model_name):
    nlp = spacy.load("en_core_web_sm")
    FUNCTION_WORD_POS = {'ADP','CCONJ','DET','PART','PRON','SCONJ','AUX','INTJ','NUM','SYM','X'}
    segmenter = main_batch.OrderedPhraseSegmenter(max_phrase_length=main_batch.MAX_PHRASE_LENGTH)
    units = segmenter.segment(text)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = main_batch.StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16 if device.type=="cuda" else torch.float32, use_safetensors=True).to(device)
    pipe.enable_attention_slicing()
    model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
    model_clip.eval()
    generator = torch.Generator(device=device).manual_seed(42)
    base_latents = torch.randn((1, 4, 64, 64), generator=generator, device=device, dtype=pipe.unet.dtype)
    with torch.no_grad():
        original_image = pipe(prompt=text, latents=base_latents.clone(), num_inference_steps=25, guidance_scale=7.5, height=512, width=512).images[0]
        original_feature = main_batch.get_clip_feature(original_image, preprocess_clip, model_clip, device)
        original_image.close()
    perturbator = main_batch.MExGenPerturbator(K=main_batch.K, N_MULTIPLIER=main_batch.N_MULTIPLIER)
    samples = perturbator.generate(units)
    explainer = main_batch.TextImageSimilarityExplainer(original_feature, units)
    explanation_result = explainer.explain(samples, base_latents, pipe, preprocess_clip, model_clip, device)
    unit_weights = explanation_result["unit_weights"]
    doc = nlp(text)
    unit_types = []
    for unit in units:
        found_type = "other"
        for token in doc:
            if token.text == unit:
                found_type = ("function" if token.pos_ in FUNCTION_WORD_POS else "content")
                break
        unit_types.append(found_type)
    
    # Filter to only include function words
    function_words = [(i, units[i], unit_weights.get(units[i], 0)) 
                     for i in range(len(units)) if unit_types[i] == "function"]
    
    # Sort function words by importance (absolute value)
    importance_sorted = sorted(function_words, key=lambda x: abs(x[2]), reverse=True)
    
    # Get only the highest importance function word (if any exist)
    top_func_ids = [importance_sorted[0][0]] if importance_sorted else []
    
    return units, top_func_ids

def main():
    spec = importlib.util.spec_from_file_location("main_batch", "main_batch.py")
    main_batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_batch)
    with open(TXT_PATH, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    all_func_indices = []
    for text in texts:
        units, func_ids = get_high_importance_func_indices(text, main_batch, BASE_MODELS["original"])
        all_func_indices.append(func_ids)
    # Save analysis results for finetune and evaluation
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(os.path.join(OUTPUT_ROOT, "hi_func_indices.json"), "w") as f:
        import json
        json.dump(all_func_indices, f)
    print("Analysis complete, results saved")

if __name__ == "__main__":
    main()