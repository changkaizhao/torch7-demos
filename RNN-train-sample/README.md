# RNN training example

Train an RNN to recognize a sequence abba in a random ababbaabbbaaababab sequence of two characters.

Version 1. `main.lua`: Send sequecne of 4 to train but test on each character at a time.

```
th main.lua
```

The results will be not optimal (you're using `1` hidden layer with `d = 2` neurons), and you'll be able to appreciate a wide variety of combinations.
True positives, true negatives, false positives and false negatives.
Try to use more neurons or more hidden layers to improve performace (switch from `d = 2` neurons to `3`, and all will work perfectly, even with just one hidden layer).

Version 2. `train-er-rnn.lua`: is based on Element Research [rnn package](https://github.com/Element-Research/rnn/blob/master/examples/sequence-to-one.lua).

```
th train-er-rnn.lua
```
