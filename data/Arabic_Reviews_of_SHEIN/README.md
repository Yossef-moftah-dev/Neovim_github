---
language:
- ar
license: apache-2.0
size_categories:
- 1K<n<10K
task_categories:
- text-classification
- text2text-generation
- text-generation
- sentence-similarity
- zero-shot-classification
dataset_info:
  features:
  - name: text
    dtype: string
  - name: raw_text
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 514231
    num_examples: 2415
  download_size: 240066
  dataset_size: 514231
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---
# Arabic Reviews of SHEIN Online Store

## Description:
This dataset contains Arabic-language reviews of products from the SHEIN online store. The reviews cover various aspects of the products and overall customer satisfaction. 
**The goal of collecting the dataset** is to include a wide range of common phrases and terms used in daily conversation, reflecting the diversity of the dialects of the Arabic language, especially in Saudi Arabia.
## Languages
Arabic 
### Data Columns:
**raw_text**: (Comment )The text of the review written by the customer.     
**text**: (Cleaned_Comment) Removing emojis and repeated characters from the text of the review written by the customer.      
**label**: (Rating) The numerical rating given by the customer, indicating their satisfaction level with the product.     


```
@misc {ruqiya_bin_safi_2024,
	author       = { {Ruqiya Bin Safi} },
	title        = { Arabic_Reviews_of_SHEIN (Revision 9eb4068) },
	year         = 2024,
	url          = { https://huggingface.co/datasets/Ruqiya/Arabic_Reviews_of_SHEIN },
	doi          = { 10.57967/hf/2232 },
	publisher    = { Hugging Face }
}
```