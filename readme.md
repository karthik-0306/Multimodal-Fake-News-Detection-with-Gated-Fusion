# 🛡️ Multimodal Fake News Detection with Gated Fusion

A Deep Learning system that dynamically weights textual (**DeBERTa-v3**) and visual (**EfficientNet-B3**) features to identify misinformation in news articles.

## 🚀 Key Performance Metrics
By implementing a dataset-aware training policy and a gated fusion mechanism, this model achieved:
- **Peak Fake News Recall:** 95.54% (GossipCop)
- **Robustness:** Successfully filters noise from celebrity imagery to focus on textual facts.

## 📂 Pre-trained Weights
Because the model weights (~800MB each) exceed GitHub's limits, they are hosted on Google Drive. 
Download the ZIP file and place the `.pt` files in your root directory.

🔗 [**Download Model Weights**](https://drive.google.com/file/d/1xh74NZ5DQTX5zL7Pb-2rRfTLYdTu8fTY/view?usp=sharing)

## 🧠 Architecture Overview
Unlike simple concatenation, our **Gated Fusion** mechanism learns a value $g$ for every sample:
$$fused = g \cdot text\_features + (1 - g) \cdot image\_features$$
This allows the model to ignore misleading images and prioritize factual text, or vice versa.


## 🛠️ Installation & Usage
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Download the weights from the link above.
4. Open the `multimodal-fake-news-detection-with-gated-fusion.ipynb` to run inference on your own samples.