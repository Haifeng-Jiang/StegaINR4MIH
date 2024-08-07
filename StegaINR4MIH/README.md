# StegaINR4MIH

This is the official code for "StegaINR4MIH: steganography by implicit neural
representation for multi-image hiding"

# Running the code

This code requires Python 3. 

You can find the pretrained models at `2024-06-21_23-40-13_celebahq128_experiment_INR_prune\inr_model_50000.pt`.

---


To train a cover function:
```
python train_INR.py 
```

---
To train multi-image stego function:
```
python train_multi_task_INR.py 
```

---


To recover secret or stego image:
```
python sample_INR.py 
```
---

# Acknowledgements

[StegaINR](https://github.com/twinlj77/StegaINR) models are used to implement StegaINR4MIH. 