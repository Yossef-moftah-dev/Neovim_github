# Arabic Sentiment Analysis with AraBERT

Classifies Arabic e-commerce product reviews as **Positive**, **Negative**, or **Neutral** using a fine-tuned [AraBERT](https://huggingface.co/aubmindlab/bert-base-arabertv02) model.

## Business problem

An Arab e-commerce platform (e.g., Noon, Amazon.eg) receives about **50,000 Arabic reviews per day**. Classifying each one supports:

- product rankings
- seller ratings
- customer-service prioritization

Arabic adds dialect variation, slang, code-switching, and seasonal vocabulary drift (e.g., Ramadan, sales events). This project tackles them with a pretrained Arabic encoder, careful evaluation, and an optimized serving path.

## Data and model

- **Model:** [aubmindlab/bert-base-arabertv02](https://huggingface.co/aubmindlab/bert-base-arabertv02)
- **Dataset:** [Ruqiya/Arabic_Reviews_of_SHEIN](https://huggingface.co/datasets/Ruqiya/Arabic_Reviews_of_SHEIN)

No data or model weights are stored in this repository. The dataset is downloaded from the Hugging Face Hub at runtime.

## License

Released under the [MIT License](LICENSE). Check the dataset and pretrained model licenses before redistributing derived artifacts.