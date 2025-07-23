# ScaleDP + EXPO

# Install
```bash
# Download the robomimic dataset
python download_robomimic_dataset.py \
    --tasks square  \
    --dataset_types ph \
    --hdf5_types image \
    --download_dir /path/to/robomimic_dataset
```

# Citation
```
# ScaleDP
@article{zhu2024scaling,
  title={Scaling diffusion policy in transformer to 1 billion parameters for robotic manipulation},
  author={Zhu, Minjie and Zhu, Yichen and Li, Jinming and Wen, Junjie and Xu, Zhiyuan and Liu, Ning and Cheng, Ran and Shen, Chaomin and Peng, Yaxin and Feng, Feifei and others},
  journal={arXiv preprint arXiv:2409.14411},
  year={2024}
}

# EXPO
@article{dong2025expo,
  title={EXPO: Stable Reinforcement Learning with Expressive Policies},
  author={Dong, Perry and Li, Qiyang and Sadigh, Dorsa and Finn, Chelsea},
  journal={arXiv preprint arXiv:2507.07986},
  year={2025}
}

# robomimic
@inproceedings{robomimic2021,
  title={What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author={Mandlekar, Ajay and Xu, Danfei and Wong, Josiah and Nasiriany, Soroush and Wang, Chen and Kulkarni, Rohun and Fei-Fei, Li and Savarese, Silvio and Zhu, Yuke and Mart{\'\i}n-Mart{\'\i}n, Roberto},
  booktitle={5th Annual Conference on Robot Learning},
  year={2021}
}
```