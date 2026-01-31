from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model

def load_lora_whisper(model_name, lora_cfg, forced_decoder_ids):
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    model.config.forced_decoder_ids = forced_decoder_ids

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model
