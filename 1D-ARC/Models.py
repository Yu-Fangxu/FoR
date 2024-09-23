import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 12/800 solved (1,5%)
# if t.sameIOShapes:
class Model3K(nn.Module):
    def __init__(self, ch=10, padVal=-1):
        super(Model3K, self).__init__()
        self.conv3 = nn.Conv2d(ch, ch, kernel_size=3)
        self.pad1 = nn.ConstantPad2d(1, padVal)

    def forward(self, x, steps=1):
        for _ in range(steps):
            x = self.conv3(self.pad1(x))
            # x = torch.softmax(x, dim=1)
        return x


class OneConvModel(nn.Module):
    def __init__(self, ch=10, kernel=3, padVal=-1):
        super(OneConvModel, self).__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=kernel, bias=0)
        self.pad = nn.ConstantPad2d(int((kernel - 1) / 2), padVal)

    def forward(self, x, steps=1):
        if x.dtype != torch.float32:
            x = x.float()  # Convert to float32 if necessary
        self.conv = self.conv.float()
        # print("Input type after conversion (if needed):", x.dtype)
        for _ in range(steps):
            x = self.conv(self.pad(x))
        return x


#
class LinearModel(nn.Module):
    def __init__(self, inSize, outSize, ch):
        super(LinearModel, self).__init__()
        self.inSize = inSize
        self.outSize = outSize
        self.ch = ch
        self.fc = nn.Linear(inSize[0] * inSize[1] * ch, outSize[0] * outSize[1] * ch)

    def forward(self, x):
        x = x.view(1, self.inSize[0] * self.inSize[1] * self.ch)
        x = self.fc(x)
        x = x.view(1, self.ch, self.outSize[0] * self.outSize[1])
        return x


class SimpleLinearModel(nn.Module):
    def __init__(self, inSize, outSize):
        super(SimpleLinearModel, self).__init__()
        self.fc = nn.Linear(inSize, outSize)

    def forward(self, x):
        x = self.fc(x)
        return x


class LinearModelDummy(nn.Module):  # (dummy = 2 channels)
    def __init__(self, inSize, outSize):
        super(LinearModelDummy, self).__init__()
        self.inSize = inSize
        self.outSize = outSize
        self.fc = nn.Linear(
            inSize[0] * inSize[1] * 2, outSize[0] * outSize[1] * 2, bias=0
        )

    def forward(self, x):
        x = x.view(1, self.inSize[0] * self.inSize[1] * 2)
        x = self.fc(x)
        x = x.view(1, 2, self.outSize[0] * self.outSize[1])
        return x


class ColorAndCellCorrespondence(nn.Module):
    def __init__(self, inSize, outSize, ch):
        super(ColorAndCellCorrespondence, self).__init__()
        self.inSize = inSize
        self.outSize = outSize
        self.ch = ch
        self.fc = nn.Linear(inSize[0] * inSize[1] * ch, outSize[0] * outSize[1] * ch)

    def forward(self, x):
        x = x.view(1, self.inSize[0] * self.inSize[1] * self.ch)
        x = self.fc(x)
        # if self.ch == 1:
        #    x = x.view(1, self.outSize[0], self.outSize[1])
        # else:
        #    x = x.view(1, self.ch, self.outSize[0], self.outSize[1])
        # x = torch.softmax(x, dim=1)
        return x


class CAModel(nn.Module):
    def __init__(self, num_states):
        super(CAModel, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(num_states, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_states, kernel_size=1),
        )

    def forward(self, x, steps=1):
        for _ in range(steps):
            x = self.transition(torch.softmax(x, dim=1))
        return x


class TripleConvModel(nn.Module):
    def __init__(self, ch=10):
        super(TripleConvModel, self).__init__()
        self.conv3 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(ch, ch, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(ch, ch, kernel_size=7, padding=3)
        self.convO1 = nn.Conv2d(ch, ch, kernel_size=1)
        self.convO2 = nn.Conv2d(ch, ch, kernel_size=1)
        self.convO3 = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, x, steps=1):
        for _ in range(steps):
            o1 = self.conv3(x)
            o2 = self.conv5(x)
            o3 = self.conv7(x)
            x = self.convO1(o1) + self.convO2(o2) + self.convO3(o3)
            x = torch.softmax(x, dim=1)
        return x


class OneFilterConvModel(nn.Module):
    def __init__(self, ch=10):
        super(TripleConvModel, self).__init__()
        self.conv3 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(ch, ch, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(ch, ch, kernel_size=7, padding=3)

    def forward(self, x, steps=1):
        for _ in range(steps):
            o1 = self.conv3(x)
            o2 = self.conv5(x)
            o3 = self.conv7(x)
            x = (o1 + o2 + o3) / 3
            x = torch.softmax(x, dim=1)
        return x


from itertools import combinations_with_replacement as cwr
from itertools import product


# 2/800 solved (0,25%)
# if t.sameIOShapes:
def color1_2_1(task):
    train = task.trainSamples
    test = task.testSamples
    inColors = set()
    outColors = set()
    for s in train:
        for key in list(s.inMatrix.colorCount.keys()):
            inColors.add(key)
        for key in list(s.outMatrix.colorCount.keys()):
            outColors.add(key)
    for s in test:
        for key in list(s.inMatrix.colorCount.keys()):
            inColors.add(key)
    inColors = list(inColors)
    outColors = list(outColors)
    # combinations = list(cwr(outColors, len(inColors)))
    combinations = list(product(outColors, repeat=len(inColors)))
    for c in combinations:
        valid = True
        for s in train:
            for i in range(s.inMatrix.shape[0]):
                for j in range(s.inMatrix.shape[1]):
                    if c[inColors.index(s.inMatrix.m[i, j])] != s.outMatrix.m[i, j]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            ret = []
            for s in test:
                im = s.inMatrix.m
                result = np.zeros(im.shape)
                for i in range(im.shape[0]):
                    for j in range(im.shape[1]):
                        result[i, j] = c[inColors.index(im[i, j])]
                ret.append(result)
            return ret
    return []


def pixelCorrespondence(t):
    """
    Returns a dictionary. Keys are positions of the output matrix. Values are
    the pixel in the input matrix it corresponds to.
    Function only valid if t.sameInSahpe and t.sameOutShape
    """
    pixelsColoredAllSamples = []
    # In which positions does each color appear?
    for s in t.trainSamples:
        pixelsColored = [[] for i in range(10)]
        m = s.inMatrix.m
        for i, j in np.ndindex(t.inShape):
            pixelsColored[m[i, j]].append((i, j))
        pixelsColoredAllSamples.append(pixelsColored)

    # For each pixel in output matrix, find correspondent pixel in input matrix
    pixelMap = {}
    for i, j in np.ndindex(t.outShape):
        candidates = set()
        for s in range(t.nTrain):
            m = t.trainSamples[s].outMatrix.m
            if len(candidates) == 0:
                candidates = set(pixelsColoredAllSamples[s][m[i, j]])
            else:
                candidates = set(pixelsColoredAllSamples[s][m[i, j]]) & candidates
            if len(candidates) == 0:
                return {}
        pixelMap[(i, j)] = next(iter(candidates))

    return pixelMap


# %% LSTM
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
