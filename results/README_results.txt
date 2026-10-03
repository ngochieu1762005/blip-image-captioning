Image Captioning with BLIP on Flickr8k

Run: flickr8k_blip_decoder_v1
Quick run: False

Dataset repository: intro/flickr8k
Repository revision recorded at setup: c7a02cb487da0eca729e133194b0f842cb578c08
Data loaded from converted Parquet files; source URLs are recorded in parquet_source.json.
The repository revision is metadata, not a pinned Parquet revision.
Model: Salesforce/blip-image-captioning-base, commit 82a37760796d32b1411fe092ab5d4e227313294b

Training: frozen visual encoder; train text decoder; one caption per image per epoch,
cyclic over 5 captions. Best checkpoint selected by validation loss on caption_0.
Test: 5 references/image, identical generation settings for both checkpoints.
BLEU: NLTK corpus_bleu, unsmoothed, lowercase regex word tokenizer.
CIDEr: pycocoevalcap.cider.Cider, same regex tokenizer; not CIDEr-D or official PTB scores.

Pretraining-data overlap has not been audited. Fine-tuning may improve or reduce caption quality.
seconds_per_image measures batch throughput, including processing/generation/decoding after warm-up.
Human-review fields are intentionally blank until someone inspects the images.
Best weights and epoch resume state are stored separately in the run folder.
