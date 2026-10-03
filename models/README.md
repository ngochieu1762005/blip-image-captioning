# Selected checkpoint

Download the complete [best_model folder](https://drive.google.com/drive/folders/178szo47NNO2jFe5UiGKgDSyiUOwgaijO) from Google Drive and place its contents in `models/best_model/`.

The folder includes `model.safetensors`, `config.json`, tokenizer files, and `preprocessor_config.json`. All are needed to load the checkpoint locally. The weight file is approximately 990 MB and is not included in this repository.

The selected epoch and validation loss are recorded in `../best_checkpoint_selection.json`.

```bash
python scripts/infer.py --model models/best_model --image path/to/image.jpg
```

`latest_training_state.pt` stores the state needed to resume training. It is not required for inference.
