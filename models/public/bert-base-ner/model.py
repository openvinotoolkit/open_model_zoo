from transformers.models.bert import BertForTokenClassification

def create_model(model_dir):
    return BertForTokenClassification.from_pretrained(model_dir)