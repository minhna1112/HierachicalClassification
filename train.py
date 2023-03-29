from trainer import Trainer


kwargs = {"batch_size": 16,
        "split_ratio": 0.7,
        "output_types" : "probs",
        "freeze" :  True,
        "w":  0.1,
        "workers": 8}

trainer = Trainer(**kwargs)

loop_idx = 0
for w in range(1, 10): # 9 run , each corresponding to a value of w
    trainer.w = float(w)/10. # w run from 0.1 to 0.9
    trainer.loop(10, loop_idx) # Each run contain 10 epochs of loop
    loop_idx +=1
