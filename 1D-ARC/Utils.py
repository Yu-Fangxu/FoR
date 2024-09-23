import numpy as np
import operator
import copy
import Models
import Task
import torch
import torch.nn as nn
from itertools import product, permutations, combinations, combinations_with_replacement
from functools import partial
from collections import Counter, defaultdict
# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

from importlib import import_module
from typing import Any, Dict, Iterable, List, Union



def transform_to_function(input_str: str, function_name: str) -> str:
    header = f"def {function_name}(I):\n"
    indented_content = "    " + input_str.strip().replace("\n", "\n    ")
    footer = "\n    return O"
    return header + indented_content + footer


def get_tokenizer(config):
    tokenizer_class = get_class(config.data.dataloader.tokenizer.cls)
    tokenizer = tokenizer_class.from_pretrained(
        config.data.dataloader.tokenizer.name, cache_dir=config.model.cache_dir
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = config.data.dataloader.tokenizer.pad_token_id
    if not tokenizer.eos_token_id:
        tokenizer.eos_token_id = config.data.dataloader.tokenizer.eos_token_id
    if (
        tokenizer.pad_token_id != config.data.dataloader.tokenizer.pad_token_id
        or tokenizer.eos_token_id != config.data.dataloader.tokenizer.eos_token_id
    ):
        raise Exception("Mismatch in tokenizer")
    return tokenizer


def get_class(class_str: str) -> Any:
    """
    A little util to import a class from any package, withot the need for an explicit import
    statement in the header. Useful when your classes can come from many different modules (like HF models).
    Can actually work also with modules which are not classes, like functions.
    :param class_str: The name of the class, inclusive of namespace from the root of the
     module is comes from (e.g. torch.nn.Linear, NOT just Linear
    :return: The class itself.
    """
    package = import_module(".".join(class_str.split(".")[:-1]))
    return getattr(package, class_str.split(".")[-1])


def get_grid_size(grid):
    num_rows = len(grid)
    num_columns = len(grid[0]) if num_rows > 0 else 0
    return (num_rows, num_columns)


def get_num_pixels(grid):
    num_rows = len(grid)
    num_columns = len(grid[0]) if num_rows > 0 else 0
    return num_rows if num_columns == 0 else num_rows * num_columns

def identityM(matrix):
    """
    Function that, given Matrix, returns its corresponding numpy.ndarray m
    """
    if isinstance(matrix, np.ndarray):
        return matrix.copy()
    else:
        return matrix.m.copy()

def correctFixedColors(inMatrix, x, fixedColors, onlyChangedInColors):
    """
    Given an input matrix (inMatrix), an output matrix (x) and a set of colors
    that should not change between the input and the output (fixedColors),
    this function returns a copy of x, but correcting the pixels that 
    shouldn't have changed back into the original, unchanged color.
    
    inMatrix and x are required to have the same shape.
    """
    m = x.copy()
    for i,j in np.ndindex(m.shape):
        if inMatrix[i,j] in fixedColors:
            m[i,j] = inMatrix[i,j]
        if m[i,j] in onlyChangedInColors:
            m[i,j] = inMatrix[i,j]
    return m
                
def incorrectPixels(m1, m2):
    """
    Returns the number of incorrect pixels (0 is best)
    """
    if m1.shape != m2.shape:
        return 1000
    return np.sum(m1!=m2)

def deBackgroundizeMatrix(m, color):
    """
    Given a matrix m and a color, this function returns a matrix whose elements
    are 0 or 1, depending on whether the corresponding pixel is of the given
    color or not.
    """
    return np.uint8(m == color)

def relDicts(colors):
    """
    Given a list of colors (numbers from 0 to 9, no repetitions allowed), this
    function returns two dictionaries giving the relationships between the
    color and its index in the list.
    It's just a way to map the colors to list(range(nColors)).
    """
    rel = {}
    for i in range(len(colors)):
        rel[i] = colors[i]
    invRel = {v: k for k,v in rel.items()}
    for i in range(len(colors)):
        rel[i] = [colors[i]]
    return rel, invRel

def dummify(x, nChannels, rel=None):
    """
    Given a matrix and a relationship given by relDicts, this function returns
    a nColors x shape(x) matrix consisting only of ones and zeros. For each
    channel (corresponding to a color), each element will be 1 if in the 
    original matrix x that pixel is of the corresponding color.
    If rel is not specified, it is expected that the values of x range from
    0 to nChannels-1.
    """
    img = np.full((nChannels, x.shape[0], x.shape[1]), 0, dtype=np.uint8)
    if rel==None:
        for i in range(nChannels):
            img[i] = x==i
    else:
        for i in range(len(rel)):
            img[i] = np.isin(x,rel[i])
    return img

def dummifyColor(x, color):
    """
    Given a matrix x and a color, this function returns a 2-by-shape(x) matrix
    of ones and zeros. In one channel, the elements will be 1 if the pixel is
    of the given color. In the other channel, they will be 1 otherwise.
    """
    img = np.full((2, x.shape[0], x.shape[1]), 0, dtype=np.uint8)
    img[0] = x!=color
    img[1] = x==color
    return img

def updateBestFunction(t, f, bestScore, bestFunction, checkPerfect=False, prevScore=None):
    """
    Given a task t, a partial function f, a best score and a best function, 
    this function executes f to all the matrices in t.trainSamples. If the
    resulting score is lower than bestScore, then it returns f and the new
    best score. Otherwise, it returns bestFunction again.
    """
    fun = copy.deepcopy(f)
    score = 0
    if checkPerfect:
        changedPixels = 0
    for sample in t.trainSamples:
        pred = fun(sample.inMatrix)
        score += incorrectPixels(sample.outMatrix.m, pred)
        if checkPerfect:
            changedPixels += incorrectPixels(pred, sample.inMatrix.m)
    if score < bestScore:
        bestScore = score
        bestFunction = fun
    if checkPerfect:
        if changedPixels != 0 and prevScore - changedPixels == score: 
            isPerfect = True
        else:
            isPerfect = False
        return fun, score, isPerfect
    return bestFunction, bestScore

# %% Symmetrize

# if t.lrSymmetric or t.udSymmetric or t.d1Symmetric:
# if len(t.changingColors) == 1:
def symmetrize(matrix, axis, color=None, outColor=None, refColor=None):
    """
    Given a matrix and a color, this function tries to turn pixels of that
    given color into some other one in order to make the matrix symmetric.
    "axis" is a list or set specifying the symmetry axis (lr, ud, d1 or d2).
    """
    # Left-Right
    def LRSymmetrize(m):
        width = m.shape[1] - 1
        for i in range(m.shape[0]):
            for j in range(int(m.shape[1] / 2)):
                if m[i,j] != m[i,width-j]:
                    if color==None:
                        if m[i,j]==refColor and m[i,width-j]!=refColor:
                            m[i,width-j] = outColor
                        elif m[i,j]!=refColor and m[i,width-j]==refColor:
                            m[i,j] = outColor
                    else:
                        if m[i,j] == color:
                            m[i,j] = m[i,width-j]
                        elif m[i,width-j]==color:
                            m[i,width-j] = m[i,j]
        return m
    
    # Up-Down
    def UDSymmetrize(m):
        height = m.shape[0] - 1
        for i in range(int(m.shape[0] / 2)):
            for j in range(m.shape[1]):
                if m[i,j] != m[height-i,j]:
                    if color==None:
                        if m[i,j]==refColor and m[height-i,j]!=refColor:
                            m[height-i,j] = outColor
                        elif m[i,j]!=refColor and m[height-i,j]==refColor:
                            m[i,j] = outColor
                    else:
                        if m[i,j] == color:
                            m[i,j] = m[height-i,j]
                        elif m[height-i,j]==color:
                            m[height-i,j] = m[i,j]
        return m

    # Main diagonal
    def D1Symmetrize(m):
        for i,j in np.ndindex(m.shape):
            if m[i,j] != m[j,i]:
                if color==None:
                    if m[i,j]==refColor and m[j,i]!=refColor:
                        m[j,i] = outColor
                    elif m[i,j]!=refColor and m[j,i]==refColor:
                        m[i,j] = outColor
                else:
                    if m[i,j] == color:
                        m[i,j] = m[j,i]
                    elif m[j,i]==color:
                        m[j,i] = m[i,j]
        return m
    
    def D2Symmetrize(matrix):
        for i,j in np.ndindex(m.shape):
            if m[i,j] != m[m.shape[0]-j-1, m.shape[1]-i-1]:
                if color==None:
                    if m[i,j]==refColor and m[m.shape[0]-j-1, m.shape[1]-i-1]!=refColor:
                        m[m.shape[0]-j-1, m.shape[1]-i-1] = outColor
                    elif m[i,j]!=refColor and m[m.shape[0]-j-1, m.shape[1]-i-1]==refColor:
                        m[i,j] = outColor
                else:
                    if m[i,j] == color:
                        m[i,j] = m[m.shape[0]-j-1, m.shape[1]-i-1]
                    elif m[m.shape[0]-j-1, m.shape[1]-i-1]==color:
                        m[m.shape[0]-j-1, m.shape[1]-i-1] = m[i,j]
        return m
    
    m = matrix.m.copy()
    while True:
        prevMatrix = m.copy()
        if "lr" in axis:
            m = LRSymmetrize(m)
        if "ud" in axis:
            m = UDSymmetrize(m)
        if "d1" in axis:
            m = D1Symmetrize(m)
        if "d2" in axis:
            m = D2Symmetrize(m)
        if np.array_equal(prevMatrix, m):
            break
            
    return m

# %% Color symmetric pixels (task 653)

def colorSymmetricPixels(matrix, inColor, outColor, axis, includeAxis=False):
    m = matrix.m.copy()
    if axis=="lr":
        for i,j in np.ndindex((m.shape[0], int(m.shape[1]/2))):
            if m[i,j]==inColor and m[i,m.shape[1]-1-j]==inColor:
                m[i,j] = outColor
                m[i,m.shape[1]-1-j] = outColor
        if includeAxis and ((m.shape[1]%2)==1):
            j = int(m.shape[1]/2)
            for i in range(m.shape[0]):
                if m[i,j]==inColor:
                    m[i,j] = outColor
    if axis=="ud":
        for i,j in np.ndindex((int(m.shape[0]/2), m.shape[1])):
            if m[i,j]==inColor and m[m.shape[0]-1-i,j]==inColor:
                m[i,j] = outColor
                m[m.shape[0]-1-i,j] = outColor
        if includeAxis and ((m.shape[0]%2)==1):
            i = int(m.shape[0]/2)
            for j in range(m.shape[1]):
                if m[i,j]==inColor:
                    m[i,j] = outColor
    if axis=="d1":
        for i in range(m.shape[0]):
            for j in range(i):
                if m[i,j]==inColor and m[j,i]==inColor:
                    m[i,j] = outColor
                    m[j,i] = outColor
        if includeAxis:
            for i in range(m.shape[0]):
                if m[i,i]==inColor:
                    m[i,i] = outColor
    if axis=="d2":
        for i in range(m.shape[0]):
            for j in range(m.shape[0]-i-1):
                if m[i,j]==inColor and m[m.shape[1]-j-1,m.shape[0]-i-1]==inColor:
                    m[i,j] = outColor
                    m[m.shape[1]-j-1,m.shape[0]-i-1] = outColor
        if includeAxis:
            for i in range(m.shape[0]):
                if m[i, m.shape[0]-i-1]==inColor:
                   m[i, m.shape[0]-i-1] = outColor
                
    return m

def getBestColorSymmetricPixels(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    
    for cic in t.commonChangedInColors:
        for coc in t.commonChangedOutColors:
            f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                        axis="lr", includeAxis=True)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                        axis="lr", includeAxis=False)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                        axis="ud", includeAxis=True)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                        axis="ud", includeAxis=False)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            if all([s.inMatrix.shape[0]==s.inMatrix.shape[1] for s in t.trainSamples+t.testSamples]):
                f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                            axis="d1", includeAxis=True)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                            axis="d1", includeAxis=False)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                            axis="d2", includeAxis=True)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                f = partial(colorSymmetricPixels, inColor=cic, outColor=coc, \
                            axis="d2", includeAxis=False)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                
    return bestFunction

# %% Train and predict models  
"""
def trainCNNDummyCommonColors(t, commonColors, k, pad):
    nChannels = len(commonColors)+2
    model = Models.OneConvModel(nChannels, k, pad)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    for e in range(100): # numEpochs   
        optimizer.zero_grad()
        loss = 0.0
        for s in t.trainSamples:
            for c in s.colors:
                if c not in commonColors:
                    itColors = commonColors + [c]
                    rel, invRel = relDicts(itColors)
                    firstCC = True
                    for cc in s.colors:
                        if cc not in itColors:
                            if firstCC:
                                rel[nChannels-1] = [cc]
                                firstCC = False
                            else:
                                rel[nChannels-1].append(cc)
                            invRel[cc] = nChannels-1
                    x = dummify(s.inMatrix.m, nChannels, rel)
                    x = torch.tensor(x).unsqueeze(0).float()
                    y = s.outMatrix.m.copy()
                    for i,j in np.ndindex(y.shape):
                        y[i,j] = invRel[y[i,j]]
                    y = torch.tensor(y).unsqueeze(0).long()
                    y_pred = model(x)
                    loss += criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    return model

@torch.no_grad()
def predictCNNDummyCommonColors(matrix, model, commonColors):
    m = matrix.m.copy()
    nChannels = len(commonColors)+2
    pred = np.zeros(m.shape)
    for c in matrix.colors:
        if c not in commonColors:
            itColors = commonColors + [c]
            rel, invRel = relDicts(itColors)
            firstCC = True
            for cc in matrix.colors:
                if cc not in itColors:
                    if firstCC:
                        rel[nChannels-1] = [cc]
                        firstCC = False
                    else:
                        rel[nChannels-1].append(cc)
            x = dummify(m, nChannels, rel)
            x = torch.tensor(x).unsqueeze(0).float()
            x = model(x).argmax(1).squeeze(0).numpy()
            for i,j in np.ndindex(m.shape):
                if m[i,j] == c:
                    pred[i,j] = rel[x[i,j]][0]
    return pred
"""

# def trainCNNDummyColor(t, k, pad):
#     """
#     This function trains a CNN with only one convolution of filter k and with
#     padding values equal to pad.
#     The training samples will have two channels: the background color and any
#     other color. The training loop loops through all the non-background colors
#     of each sample, treating them independently.
#     This is useful for tasks like number 3.
#     """
#     model = Models.OneConvModel(2, k, pad)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
#     criterion = nn.CrossEntropyLoss()
#     for e in range(50): # numEpochs            
#         optimizer.zero_grad()
#         loss = 0.0
#         for s in t.trainSamples:
#             for c in s.colors:
#                 if c != t.backgroundColor:
#                     x = dummifyColor(s.inMatrix.m, c)
#                     x = torch.tensor(x).unsqueeze(0).float()
#                     y = deBackgroundizeMatrix(s.outMatrix.m, c)
#                     y = torch.tensor(y).unsqueeze(0).long()
#                     y_pred = model(x)
#                     loss += criterion(y_pred, y)
#         loss.backward()
#         optimizer.step()
#     return model

# @torch.no_grad()
# def predictCNNDummyColor(matrix, model):
#     """
#     Predict function for a model trained using trainCNNDummyColor.
#     """
#     m = matrix.m.copy()
#     pred = np.ones(m.shape, dtype=np.uint8) * matrix.backgroundColor
#     for c in matrix.colors:
#         if c != matrix.backgroundColor:
#             x = dummifyColor(m, c)
#             x = torch.tensor(x).unsqueeze(0).float()
#             x = model(x).argmax(1).squeeze(0).numpy()
#             for i,j in np.ndindex(m.shape):
#                 if x[i,j] != 0:
#                     pred[i,j] = c
#     return pred

# def trainCNN(t, commonColors, nChannels, k=5, pad=0):
#     """
#     This function trains a CNN model with kernel k and padding value pad.
#     It is required that all the training samples have the same number of colors
#     (adding the colors in the input and in the output).
#     It is also required that the output matrix has always the same shape as the
#     input matrix.
#     The colors are tried to be order in a specific way: first the colors that
#     are common to every sample (commonColors), and then the others.
#     """
#     model = Models.OneConvModel(nChannels, k, pad)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
#     #losses = np.zeros(100)
#     for e in range(100):
#         optimizer.zero_grad()
#         loss = 0.0
#         for s in t.trainSamples:
#             sColors = commonColors.copy()
#             for c in s.colors:
#                 if c not in sColors:
#                     sColors.append(c)
#             rel, invRel = relDicts(sColors)
#             x = dummify(s.inMatrix.m, nChannels, rel)
#             x = torch.tensor(x).unsqueeze(0).float()
#             y = s.outMatrix.m.copy()
#             for i,j in np.ndindex(y.shape):
#                 y[i,j] = invRel[y[i,j]]
#             y = torch.tensor(y).unsqueeze(0).long()
#             y_pred = model(x)
#             loss += criterion(y_pred, y)
#         loss.backward()
#         optimizer.step()
#         #losses[e] = loss
#     return model#, losses

# @torch.no_grad()
# def predictCNN(matrix, model, commonColors, nChannels):
#     """
#     Predict function for a model trained using trainCNN.
#     """
#     m = matrix.m.copy()
#     pred = np.zeros(m.shape, dtype=np.uint8)
#     sColors = commonColors.copy()
#     for c in matrix.colors:
#         if c not in sColors:
#             sColors.append(c)
#     rel, invRel = relDicts(sColors)
#     if len(sColors) > nChannels:
#         return m
#     x = dummify(m, nChannels, rel)
#     x = torch.tensor(x).unsqueeze(0).float()
#     x = model(x).argmax(1).squeeze(0).numpy()
#     for i,j in np.ndindex(m.shape):
#         if x[i,j] not in rel.keys():
#             pred[i,j] = x[i,j]
#         else:
#             pred[i,j] = rel[x[i,j]][0]
#     return pred

# def getBestCNN(t):
#     """
#     This function returns the best CNN with only one convolution, after trying
#     different kernel sizes and padding values.
#     There are as many channels as total colors or the minimum number of
#     channels that is necessary.
#     """
#     kernel = [3,5,7]
#     pad = [0,-1]    
#     bestScore = 100000
#     for k, p in product(kernel, pad):
#         cc = list(range(10))
#         model = trainCNN(t, commonColors=cc, nChannels=10, k=k, pad=p)
#         score = sum([incorrectPixels(predictCNN(t.trainSamples[s].inMatrix, model, cc, 10), \
#                                      t.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
#         if score < bestScore:
#             bestScore=score
#             ret = partial(predictCNN, model=model, commonColors=cc, nChannels=10)
#             if score==0:
#                 return ret
#     return ret

# def getBestSameNSampleColorsCNN(t):
#     kernel = [3,5,7]
#     pad = [0,-1]    
#     bestScore = 100000
#     for k, p in product(kernel, pad):
#         cc = list(t.commonSampleColors)
#         nc = t.trainSamples[0].nColors
#         model = trainCNN(t, commonColors=cc, nChannels=nc, k=k, pad=p)
#         score = sum([incorrectPixels(predictCNN(t.trainSamples[s].inMatrix, model, cc, nc), \
#                                      t.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
#         if score < bestScore:
#             bestScore=score
#             ret = partial(predictCNN, model=model, commonColors=cc, nChannels=nc)
#             if score==0:
#                 return ret
            
#     return ret

# %% CNN learning the output
    
def getNeighbourColors(m, i, j, border=0):
    """
    Given a matrix m and a position i,j, this function returns a list of the
    values of the neighbours of (i,j).
    """
    x = []
    y = m[i-1,j] if i>0 else border
    x.append(y)
    y = m[i,j-1] if j>0 else border
    x.append(y)
    y = m[i+1,j] if i<m.shape[0]-1 else border
    x.append(y)
    y = m[i,j+1] if j<m.shape[1]-1 else border
    x.append(y)
    return x

def getDNeighbourColors(m, i, j, kernel=3, border=0):
    """
    Given a matrix m and a position i,j, this function returns a list of the
    values of the diagonal neighbours of (i,j).
    """
    x = []
    y = m[i-1,j-1] if (i>0 and j>0) else border
    x.append(y)
    y = m[i+1,j-1] if (i<m.shape[0]-1 and j>0) else border
    x.append(y)
    y = m[i-1,j+1] if (i>0 and j<m.shape[1]-1) else border
    x.append(y)
    y = m[i+1,j+1] if (i<m.shape[0]-1 and j<m.shape[1]-1) else border
    x.append(y)
    
    if kernel==5:
        y = m[i-2,j-2] if (i>1 and j>1) else border
        x.append(y)
        y = m[i-1,j-2] if (i>0 and j>1) else border
        x.append(y)
        y = m[i,j-2] if j>1 else border
        x.append(y)
        y = m[i+1,j-2] if (i<m.shape[0]-1 and j>1) else border
        x.append(y)
        y = m[i+2,j-2] if (i<m.shape[0]-2 and j>1) else border
        x.append(y)
        y = m[i+2,j-1] if (i<m.shape[0]-2 and j>0) else border
        x.append(y)
        y = m[i+2,j] if i<m.shape[0]-2 else border
        x.append(y)
        y = m[i+2,j+1] if (i<m.shape[0]-2 and j<m.shape[1]-1) else border
        x.append(y)
        y = m[i+2,j+2] if (i<m.shape[0]-2 and j<m.shape[1]-2) else border
        x.append(y)
        y = m[i+1,j+2] if (i<m.shape[0]-1 and j<m.shape[1]-2) else border
        x.append(y)
        y = m[i,j+2] if j<m.shape[1]-2 else border
        x.append(y)
        y = m[i-1,j+2] if (i>0 and j<m.shape[1]-2) else border
        x.append(y)
        y = m[i-2,j+2] if (i>1 and j<m.shape[1]-2) else border
        x.append(y)
        y = m[i-2,j+1] if (i>1 and j<m.shape[1]-1) else border
        x.append(y)
        y = m[i-2,j] if i>1 else border
        x.append(y)
        y = m[i-2,j-1] if (i>1 and j>0) else border
        x.append(y)
    return x

def getAllNeighbourColors(m, i, j, kernel=3, border=0):
    return getNeighbourColors(m,i,j,border) + getDNeighbourColors(m,i,j,kernel,border)

def colorNeighbours(mIn, mOut ,i, j):
    if i>0:
        mIn[i-1,j] = mOut[i-1,j]
    if j>0:
        mIn[i,j-1] = mOut[i,j-1]
    if i<mIn.shape[0]-1:
        mIn[i+1,j] = mOut[i+1,j]
    if j<mIn.shape[1]-1:
        mIn[i,j+1] = mOut[i,j+1]
        
def colorDNeighbours(mIn, mOut, i, j):
    colorNeighbours(mIn, mOut ,i, j)
    if i>0 and j>0:
        mIn[i-1,j-1] = mOut[i-1,j-1]
    if i<mIn.shape[0]-1 and j>0:
        mIn[i+1,j-1] = mOut[i+1,j-1]
    if i>0 and j<mIn.shape[1]-1:
        mIn[i-1,j+1] = mOut[i-1,j+1]
    if i<mIn.shape[0]-1 and j<mIn.shape[1]-1:
        mIn[i+1,j+1] = mOut[i+1,j+1]
     
# if len(t.changedInColors)==1 (the background color, where everything evolves)
# 311/800 tasks satisfy this condition
# Do I need inMatrix.nColors+fixedColors to be equal for every sample?
def evolve(t, kernel=3, border=0, includeRotations=False):
    def evolveInputMatrices(mIn, mOut, changeCIC=False):
        reference = [m.copy() for m in mIn]
        for m in range(len(mIn)):
            if changeCIC:
                for i,j in np.ndindex(mIn[m].shape):
                    if mIn[m][i,j] not in set.union(fixedColors, changedOutColors): 
                        colorDNeighbours(mIn[m], mOut[m], i, j)
                        break
            else:
                for i,j in np.ndindex(mIn[m].shape):
                    if referenceIsFixed and reference[m][i,j] in fixedColors:
                        colorDNeighbours(mIn[m], mOut[m], i, j)
                    elif reference[m][i,j] in changedOutColors:
                        colorDNeighbours(mIn[m], mOut[m], i, j)
                    
    nColors = t.trainSamples[0].nColors
    
    if not t.allEqual(t.sampleColors):
        sampleRel = []
        sampleInvRel = []
        commonColors = t.orderedColors
        for s in t.trainSamples:
            colors = commonColors.copy()
            for i,j in np.ndindex(s.inMatrix.shape):
                if s.inMatrix.m[i,j] not in colors:
                    colors.append(s.inMatrix.m[i,j])
                    if len(colors) == nColors:
                        break
            if len(colors) != nColors:
                for i,j in np.ndindex(s.outMatrix.shape):
                    if s.outMatrix.m[i,j] not in colors:
                        colors.append(s.outMatrix.m[i,j])
                        if len(colors) == nColors:
                            break
            rel, invRel = relDicts(colors)
            sampleRel.append(rel)
            sampleInvRel.append(invRel)
            
        fixedColors = set()
        for c in t.fixedColors:
            fixedColors.add(sampleInvRel[0][c])
        changedOutColors = set()
        for c in t.commonChangedOutColors:
            changedOutColors.add(sampleInvRel[0][c])
        for c in range(len(commonColors), nColors):
            changedOutColors.add(c)
    else:
        fixedColors = t.fixedColors
        changedOutColors = t.commonChangedOutColors
        
    referenceIsFixed = t.trainSamples[0].inMatrix.nColors == len(fixedColors)+1
    
    outMatrices = [s.outMatrix.m.copy() for s in t.trainSamples]
    referenceOutput = [s.inMatrix.m.copy() for s in t.trainSamples]
    
    if includeRotations:
        for i in range(1,4):
            for m in range(t.nTrain):
                outMatrices.append(np.rot90(outMatrices[m].copy(), i))
                referenceOutput.append(np.rot90(referenceOutput[m].copy(), i))
                    
    
    if not t.allEqual(t.sampleColors):
        for m in range(len(outMatrices)):
            for i,j in np.ndindex(outMatrices[m].shape):
                outMatrices[m][i,j] = sampleInvRel[m%t.nTrain][outMatrices[m][i,j]]
                referenceOutput[m][i,j] = sampleInvRel[m%t.nTrain][referenceOutput[m][i,j]]
    
    colorFromNeighboursK2 = {}
    colorFromNeighboursK3 = {}
    colorFromNeighboursK5 = {}
    for i in range(10):
        referenceInput = [m.copy() for m in referenceOutput]
        evolveInputMatrices(referenceOutput, outMatrices)
        if np.all([np.array_equal(referenceInput[m], referenceOutput[m]) for m in range(len(referenceInput))]):
            evolveInputMatrices(referenceOutput, outMatrices, True)
        for m in range(len(outMatrices)):
            for i,j in np.ndindex(referenceInput[m].shape):
                if referenceInput[m][i,j] != referenceOutput[m][i,j]:
                    neighbourColors = tuple(getNeighbourColors(referenceInput[m],i,j,border))
                    colorFromNeighboursK2[neighbourColors] = referenceOutput[m][i,j]
                    neighbourColors = tuple(getAllNeighbourColors(referenceInput[m],i,j,3,border))
                    colorFromNeighboursK3[neighbourColors] = referenceOutput[m][i,j]
                    neighbourColors = tuple(getAllNeighbourColors(referenceInput[m],i,j,5,border))
                    colorFromNeighboursK5[neighbourColors] = referenceOutput[m][i,j]
       
    colorFromNeighboursK2 = {k:v for k,v in colorFromNeighboursK2.items() if \
                             not all([x not in set.union(changedOutColors, fixedColors) for x in k])}
    colorFromNeighboursK3 = {k:v for k,v in colorFromNeighboursK3.items() if \
                             not all([x not in set.union(changedOutColors, fixedColors) for x in k])}
    colorFromNeighboursK5 = {k:v for k,v in colorFromNeighboursK5.items() if \
                             not all([x not in set.union(changedOutColors, fixedColors) for x in k])}
                
    colorfromNeighboursK3Background = {}
    for m in outMatrices:
        for i,j in np.ndindex(m.shape):
            if m[i,j] in t.commonChangedInColors:
                neighbourColors = tuple(getAllNeighbourColors(m,i,j,3,border))
                colorfromNeighboursK3Background[neighbourColors] = m[i,j]
        
    
    return [colorFromNeighboursK2, colorFromNeighboursK3,\
            colorFromNeighboursK5, colorfromNeighboursK3Background]
        

def applyEvolve(matrix, cfn, nColors, changedOutColors=set(), fixedColors=set(),\
                changedInColors=set(), referenceIsFixed=False, commonColors=set(),\
                kernel=None, border=0, nIterations=1000):
        
    def colorPixel(m,newM,i,j):
        if newM[i,j] not in cic and colorAroundCIC==False:
            return
        if kernel==None:
            tup3 = tuple(getAllNeighbourColors(m,i,j,3,border))
            tup5 = tuple(getAllNeighbourColors(m,i,j,5,border))
            #tup = getMostSimilarTuple(tup)
            if tup3 in cfn[1].keys():
                if tup3 in cfn[3].keys():
                    if tup5 in cfn[2].keys():
                        newM[i,j] = cfn[2][tup5]
                else:
                    newM[i,j] = cfn[1][tup3] 
            elif tup5 in cfn[2].keys():
                newM[i,j] = cfn[2][tup5]
        elif kernel==2:
            tup2 = tuple(getNeighbourColors(m,i,j,border))
            if tup2 in cfn[0].keys():
                newM[i,j] = cfn[0][tup2]
        elif kernel==3:
            tup3 = tuple(getAllNeighbourColors(m,i,j,3,border))
            if tup3 in cfn[1].keys():
                newM[i,j] = cfn[1][tup3]
        elif kernel==5:
            tup5 = tuple(getAllNeighbourColors(m,i,j,5,border))
            if tup5 in cfn[2].keys():
                newM[i,j] = cfn[2][tup5]
        
    def colorPixelsAround(m,newM,i,j):
        if i>0:
            colorPixel(m,newM,i-1,j)
        if j>0:
            colorPixel(m,newM,i,j-1)
        if i<m.shape[0]-1:
            colorPixel(m,newM,i+1,j)
        if j<m.shape[1]-1:
            colorPixel(m,newM,i,j+1)
        if i>0 and j>0:
            colorPixel(m,newM,i-1,j-1)
        if i<m.shape[0]-1 and j>0:
            colorPixel(m,newM,i+1,j-1)
        if i>0 and j<m.shape[1]-1:
            colorPixel(m,newM,i-1,j+1)
        if i<m.shape[0]-1 and j<m.shape[1]-1:
            colorPixel(m,newM,i+1,j+1)
    
    m = matrix.m.copy()
    
    if len(commonColors) > 0:
        colors = list(commonColors.copy())
        for i,j in np.ndindex(m.shape):
            if m[i,j] not in colors:
                colors.append(m[i,j])
        rel, invRel = relDicts(colors)
        
        for i,j in np.ndindex(m.shape):
            m[i,j] = invRel[m[i,j]]
            
        fc = set()
        for c in fixedColors:
            fc.add(invRel[c])        
        coc = set()
        for c in changedOutColors:
            coc.add(invRel[c])
        for c in range(len(commonColors), nColors):
            coc.add(c)
        cic = set()
        for c in changedInColors:
            cic.add(invRel[c])
    else:
        fc = fixedColors
        coc = changedOutColors
        cic = changedInColors
    
    it = 0
    colorAroundCIC=False
    while it<nIterations:
        it += 1
        newM = m.copy()
        #seen = np.zeros(m.shape, dtype=np.bool)
        if colorAroundCIC:
            for i,j in np.ndindex(m.shape):
                colorPixelsAround(m,newM,i,j)
            colorAroundCIC=False
            m=newM.copy()
            continue
        for i,j in np.ndindex(m.shape):
            if referenceIsFixed and m[i,j] in fixedColors:
                colorPixelsAround(m,newM,i,j)
            elif m[i,j] in coc:
                colorPixelsAround(m,newM,i,j)
        if np.array_equal(newM,m):
            if it==1:
                colorAroundCIC=True
            else:
                break
        m = newM.copy()
        
    if len(commonColors) > 0:
        for i,j in np.ndindex(m.shape):
            if m[i,j] in rel.keys():
                m[i,j] = rel[m[i,j]][0]
            else:
                m[i,j] = rel[0][0] # Patch for bug in task 22
        
    return m
    
def getBestEvolve(t, cfn):
    nColors = t.trainSamples[0].nColors
    fc = t.fixedColors
    cic = t.commonChangedInColors
    coc = t.commonChangedOutColors
    refIsFixed = t.trainSamples[0].inMatrix.nColors == len(fc)+1
    
    bestScore = 1000
    bestFunction = None
    
    if t.allEqual(t.sampleColors):
        f = partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                    fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                    kernel=None, border=0)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
        f =  partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                     fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                     kernel=5, border=0)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction

    else:
        f = partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                    fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                    kernel=None, border=0, commonColors=t.orderedColors)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction    
        
        f =  partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                     fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                     kernel=5, border=0, commonColors=t.orderedColors)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
    cfn = evolve(t, includeRotations=True)
    if t.allEqual(t.sampleColors):
        f = partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                    fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                    kernel=None, border=0)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction    
        
        f =  partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                     fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                     kernel=5, border=0)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction

    else:
        f = partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                    fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                    kernel=None, border=0, commonColors=t.orderedColors)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction    
        
        f =  partial(applyEvolve, cfn=cfn, nColors=nColors, changedOutColors=coc,\
                     fixedColors=fc, changedInColors=cic, referenceIsFixed=refIsFixed,\
                     kernel=5, border=0, commonColors=t.orderedColors)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
    return bestFunction
    
# To be solved: 23,57,59,65,83,93,118,135,140,147,167,189,198,201,231,236,247,
# 298,322,357,429,449,457,505,577,585,605,693,703,731,748,749,793,797
    
class EvolvingLine():
    def __init__(self, color, direction, position, cic, source=None, \
                 colorRules=None, stepSize=None, fixedDirection=True, turning=False,\
                 turnAfterNSteps=[None, None], stepIncrease=None, \
                 alternateStepIncrease=False):
        """
        cic = changedInColors
        """
        self.source = source # Task.Shape
        self.color = color
        self.direction = direction
        self.position = position
        self.cic = cic
        self.colorRules = colorRules
        self.fixedDirection = fixedDirection
        self.dealWith = {}
        self.stepSize = stepSize
        self.turning = turning
        # left=10, right=11, top=12, bot=13
        for color in range(14): # 10 colors + 4 borders
            self.dealWith[color] = 'stop'
        if type(colorRules)==str:
            for i in range(10):
                if i not in cic:
                    self.dealWith[i] = colorRules
        elif colorRules!=None:
            for cr in colorRules:
                self.dealWith[cr[0]] = cr[1]    
        self.dealWith[self.color] = "skip"
        self.maxSteps = turnAfterNSteps.copy() # [number of maximum steps, direction]
        self.stepIncrease = stepIncrease
        self.step = -1
        self.alternateStepIncrease = alternateStepIncrease
        self.increaseStep = False
        
    def draw(self, m, direction=None):
        # If we reached the maximum number of steps, turn
        if self.maxSteps[0]!=None:
            if self.maxSteps[0]==self.step:
                if self.maxSteps[1]=="stop":
                    return
                
                #self.turning=True
                
                if self.direction=='u' and self.maxSteps[1]=='r' or\
                self.direction=='d' and self.maxSteps[1]=='l':
                    direction = 'r'
                    if not self.fixedDirection:
                        self.direction = 'r'
                        
                elif self.direction=='u' and self.maxSteps[1]=='l' or\
                self.direction=='d' and self.maxSteps[1]=='r':
                    direction = 'l'
                    if not self.fixedDirection:
                        self.direction = 'l'   
                        
                elif self.direction=='r' and self.maxSteps[1]=='l' or\
                self.direction=='l' and self.maxSteps[1]=='r':
                    direction = 'u'
                    if not self.fixedDirection:
                        self.direction = 'u'
                        
                elif self.direction=='l' and self.maxSteps[1]=='l' or\
                self.direction=='r' and self.maxSteps[1]=='r':
                    direction = 'd'
                    if not self.fixedDirection:
                        self.direction = 'd' 
                        
                if self.stepIncrease!=None:
                    if self.increaseStep:
                        self.maxSteps[0]+=self.stepIncrease
                        if self.alternateStepIncrease:
                            self.increaseStep=False
                    else:
                        self.increaseStep=True
                        
                self.step = -1
                    
        self.step += 1
        
        if direction==None:
            direction=self.direction
                    
        # Left
        if direction=='l':
            if self.position[1]==0:
                if not self.turning:
                    self.turning=True
                    self.dealWithColor(10, m)
                return
            newColor = m[self.position[0], self.position[1]-1]
            if newColor in self.cic:
                if self.turning:
                    self.turning=False
                m[self.position[0], self.position[1]-1] = self.color
                self.position[1] -= 1
                self.draw(m)
            else:
                if not self.turning:
                    self.turning=True
                    self.dealWithColor(newColor, m)
    
        # Right
        if direction=='r':
            if self.position[1]==m.shape[1]-1:
                if not self.turning:
                    self.turning=True
                    self.dealWithColor(11, m)
                return
            newColor = m[self.position[0], self.position[1]+1]
            if newColor in self.cic:
                if self.turning:
                    self.turning=False
                m[self.position[0], self.position[1]+1] = self.color
                self.position[1] += 1
                self.draw(m)
            else:
                if not self.turning:
                    self.turning=True
                    self.dealWithColor(newColor, m)
                
        # Up
        if direction=='u':
            if self.position[0]==0:
                if not self.turning:
                    self.turning=True
                    self.dealWithColor(12, m)
                return
            newColor = m[self.position[0]-1, self.position[1]]
            if newColor in self.cic:
                if self.turning:
                    self.turning=False
                m[self.position[0]-1, self.position[1]] = self.color
                self.position[0] -= 1
                self.draw(m)
            else:
                if not self.turning:
                    self.turning=True
                    self.dealWithColor(newColor, m)
        
        # Down
        if direction=='d':
            if self.position[0]==m.shape[0]-1:
                if not self.turning:
                    self.turning=True
                    self.dealWithColor(13, m)
                return
            newColor = m[self.position[0]+1, self.position[1]]
            if newColor in self.cic:
                if self.turning:
                    self.turning=False
                m[self.position[0]+1, self.position[1]] = self.color
                self.position[0] += 1
                self.draw(m)
            else:
                if not self.turning:
                    self.turning=True
                    self.dealWithColor(newColor, m)
        
    def dealWithColor(self, color, m):
        if self.dealWith[color] == "stop":
            return
        
        if self.dealWith[color] == "convert":
            if self.direction=='l':
                if self.position[1]!=0:
                    self.color = color
                    self.position[1]-=1
                    self.draw(m)
                else:
                    return
            if self.direction=='r':
                if self.position[1]!=m.shape[1]-1:
                    self.color = color
                    self.position[1]+=1
                    self.draw(m)
                else:
                    return
            if self.direction=='u':
                if self.position[0]!=0:
                    self.color = color
                    self.position[0]-=1
                    self.draw(m)
                else:
                    return
            if self.direction=='d':
                if self.position[0]!=m.shape[0]-1:
                    self.color = color
                    self.position[0]+=1
                    self.draw(m)
                else:
                    return
            
        if self.dealWith[color] == "split":
            if self.direction=='l' or self.direction=='r':
                if self.position[0]!=0:
                    l1 = EvolvingLine(self.color, self.direction, self.position.copy(), self.cic,\
                                      colorRules=self.colorRules, fixedDirection=self.fixedDirection, \
                                      turning=True)
                    if self.fixedDirection==False:
                        l1.direction='u'
                    l1.draw(m, direction='u')
                if self.position[0]!=m.shape[0]-1:
                    l2 = EvolvingLine(self.color, self.direction, self.position.copy(), self.cic,\
                                      colorRules=self.colorRules, fixedDirection=self.fixedDirection, \
                                      turning=True)
                    if self.fixedDirection==False:
                        l2.direction='d'
                    l2.draw(m, direction='d')
            if self.direction=='u' or self.direction=='d':
                if self.position[1]!=0:
                    l1 = EvolvingLine(self.color, self.direction, self.position.copy(), self.cic,\
                                      colorRules=self.colorRules, fixedDirection=self.fixedDirection, \
                                      turning=True)
                    if self.fixedDirection==False:
                        l1.direction='l'
                    l1.draw(m, direction='l')
                if self.position[1]!=m.shape[1]-1:
                    l2 = EvolvingLine(self.color, self.direction, self.position.copy(), self.cic,\
                                      colorRules=self.colorRules, fixedDirection=self.fixedDirection, \
                                      turning=True)
                    if self.fixedDirection==False:
                        l2.direction='r'
                    l2.draw(m, direction='r')
                    
        if self.dealWith[color] == "skip":
            if self.direction=='l':
                if self.position[1]!=0:
                    self.position[1]-=1
                    self.draw(m)
                else:
                    return
            if self.direction=='r':
                if self.position[1]!=m.shape[1]-1:
                    self.position[1]+=1
                    self.draw(m)
                else:
                    return
            if self.direction=='u':
                if self.position[0]!=0:
                    self.position[0]-=1
                    self.draw(m)
                else:
                    return
            if self.direction=='d':
                if self.position[0]!=m.shape[0]-1:
                    self.position[0]+=1
                    self.draw(m)
                else:
                    return
                    
        # Left
        if self.dealWith[color] == 'l':
            if self.direction=='u':
                if self.position[1]!=0:
                    if not self.fixedDirection:
                        self.direction = 'l'
                    self.draw(m, direction='l')
                return
            if self.direction=='d':
                if self.position[1]!=m.shape[1]-1:
                    if not self.fixedDirection:
                        self.direction = 'r'
                    self.draw(m, direction='r')
                return
            if self.direction=='l':
                if self.position[0]!=m.shape[0]-1:
                    if not self.fixedDirection:
                        self.direction = 'd'
                    self.draw(m, direction='d')
                return
            if self.direction=='r':
                if self.position[0]!=0:
                    if not self.fixedDirection:
                        self.direction = 'u'
                    self.draw(m, direction='u')
                return
            
        # Right
        if self.dealWith[color] == 'r':
            if self.direction=='u':
                if self.position[1]!=m.shape[1]-1:
                    if not self.fixedDirection:
                        self.direction = 'r'
                    self.draw(m, direction='r')
                return
            if self.direction=='d':
                if self.position[1]!=0:
                    if not self.fixedDirection:
                        self.direction = 'l'
                    self.draw(m, direction='l')
                return
            if self.direction=='l':
                if self.position[0]!=0:
                    if not self.fixedDirection:
                        self.direction = 'u'
                    self.draw(m, direction='u')
                return
            if self.direction=='r':
                if self.position[0]!=m.shape[0]-1:
                    if not self.fixedDirection:
                        self.direction = 'd'
                    self.draw(m, direction='d')
                return            
        
def detectEvolvingLineSources(t):
    sources = set()
    if len(t.commonChangedOutColors)==1:
        coc = next(iter(t.commonChangedOutColors))
    else:
        coc = None
    possibleSourceColors = set.intersection(t.commonChangedOutColors, t.commonInColors)
    if len(possibleSourceColors) == 0:
        possibleSourceColors = set(t.fixedColors)
    if len(possibleSourceColors) != 0:
        firstIt = True
        for sample in t.trainSamples:
            sampleSources = set()
            for color in possibleSourceColors:
                if coc==None:
                    targetColor=color
                else:
                    targetColor=coc
                for shape in sample.inMatrix.shapes:
                    if shape.color==color and shape.nPixels==1:                        
                        # First special case: Corners
                        if shape.position==(0,0):
                            if sample.outMatrix.m[1][0]==targetColor and sample.outMatrix.m[0][1]==targetColor:
                                sampleSources.add((color, "away"))
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                                sampleSources.add((color, 'l'))
                                sampleSources.add((color, 'r'))
                            elif sample.outMatrix.m[1][0]==targetColor:
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                            elif sample.outMatrix.m[0][1]==targetColor:
                                sampleSources.add((color, 'l'))
                                sampleSources.add((color, 'r'))
                        elif shape.position==(0,sample.inMatrix.shape[1]-1):
                            if sample.outMatrix.m[1][sample.outMatrix.shape[1]-1]==targetColor and sample.outMatrix.m[0][sample.outMatrix.shape[1]-2]==targetColor:
                                sampleSources.add((color, "away"))
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                                sampleSources.add((color, 'l'))
                                sampleSources.add((color, 'r'))
                            elif sample.outMatrix.m[1][sample.outMatrix.shape[1]-1]==targetColor:
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                            elif sample.outMatrix.m[0][sample.outMatrix.shape[1]-2]==targetColor:
                                sampleSources.add((color, 'l'))
                                sampleSources.add((color, 'r'))
                        elif shape.position==(sample.inMatrix.shape[0]-1,0):
                            if sample.outMatrix.m[sample.outMatrix.shape[0]-2][0]==targetColor and sample.outMatrix.m[sample.outMatrix.shape[0]-1][1]==targetColor:
                                sampleSources.add((color, "away"))
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                                sampleSources.add((color, 'l'))
                                sampleSources.add((color, 'r'))
                            elif sample.outMatrix.m[sample.outMatrix.shape[0]-2][0]==targetColor:
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                            elif sample.outMatrix.m[sample.outMatrix.shape[0]-1][1]==targetColor:
                                sampleSources.add((color, 'l'))
                                sampleSources.add((color, 'r'))
                        elif shape.position==(sample.inMatrix.shape[0]-1,sample.inMatrix.shape[1]-1):
                            if sample.outMatrix.m[sample.outMatrix.shape[0]-2][sample.outMatrix.shape[1]-1]==targetColor and sample.outMatrix.m[sample.outMatrix.shape[0]-1][sample.outMatrix.shape[1]-2]==targetColor:
                                sampleSources.add((color, "away"))
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                                sampleSources.add((color, 'l'))
                                sampleSources.add((color, 'r'))
                            elif sample.outMatrix.m[sample.outMatrix.shape[0]-2][sample.outMatrix.shape[1]-1]==targetColor:
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                            elif sample.outMatrix.m[sample.outMatrix.shape[0]-1][sample.outMatrix.shape[1]-2]==targetColor:
                                sampleSources.add((color, 'l'))
                                sampleSources.add((color, 'r'))
                        
                        # Second special case: Border but not corner
                        elif shape.position[0]== 0:
                            if sample.outMatrix.m[1,shape.position[1]]==targetColor:
                                sampleSources.add((color,"away"))
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                            if sample.outMatrix.m[0,shape.position[1]-1]==targetColor:
                                sampleSources.add((color, 'l'))
                            if sample.outMatrix.m[0,shape.position[1]+1]==targetColor:
                                sampleSources.add((color, 'r'))
                        elif shape.position[0]== sample.inMatrix.shape[0]-1:
                            if sample.outMatrix.m[sample.inMatrix.shape[0]-2,shape.position[1]]==targetColor:
                                sampleSources.add((color,"away"))
                                sampleSources.add((color,'u'))
                                sampleSources.add((color, 'd'))
                            if sample.outMatrix.m[sample.inMatrix.shape[0]-1,shape.position[1]-1]==targetColor:
                                sampleSources.add((color, 'l'))
                            if sample.outMatrix.m[sample.inMatrix.shape[0]-1,shape.position[1]+1]==targetColor:
                                sampleSources.add((color, 'r'))
                        elif shape.position[1]== 0:
                            if sample.outMatrix.m[shape.position[0],1]==targetColor:
                                sampleSources.add((color,"away"))
                                sampleSources.add((color,'r'))
                                sampleSources.add((color, 'l'))
                            if sample.outMatrix.m[shape.position[0]-1,0]==targetColor:
                                sampleSources.add((color, 'u'))
                            if sample.outMatrix.m[shape.position[0]+1,0]==targetColor:
                                sampleSources.add((color, 'd'))
                        elif shape.position[1]== sample.inMatrix.shape[1]-1:
                            if sample.outMatrix.m[shape.position[0],sample.inMatrix.shape[1]-2]==targetColor:
                                sampleSources.add((color,"away"))
                                sampleSources.add((color,'r'))
                                sampleSources.add((color, 'l'))
                            if sample.outMatrix.m[shape.position[0]-1,sample.inMatrix.shape[1]-1]==targetColor:
                                sampleSources.add((color, 'u'))
                            if sample.outMatrix.m[shape.position[0]+1,sample.inMatrix.shape[1]-1]==targetColor:
                                sampleSources.add((color, 'd'))
                                
                        # Third case: Not border
                        else:
                            if sample.outMatrix.m[shape.position[0]+1, shape.position[1]]==targetColor:
                                sampleSources.add((color, 'd'))
                            if sample.outMatrix.m[shape.position[0]-1, shape.position[1]]==targetColor:
                                sampleSources.add((color, 'u'))
                            if sample.outMatrix.m[shape.position[0], shape.position[1]+1]==targetColor:
                                sampleSources.add((color, 'r'))
                            if sample.outMatrix.m[shape.position[0], shape.position[1]-1]==targetColor:
                                sampleSources.add((color, 'l'))
            if firstIt:
                sources = sampleSources
                firstIt = False
            else:
                sources = set.intersection(sources, sampleSources) 
                
    return sources

def getBestEvolvingLines(t):
    if any([s.inMatrix.shape[0]==1 or s.inMatrix.shape[1]==1 for s in t.trainSamples]):
        return partial(identityM)
    
    sources = detectEvolvingLineSources(t)
    
    fixedColorsList = list(t.fixedColors2)
    cic=t.commonChangedInColors
    #cic = [color for color in list(range(10)) if color not in fixedColorsList]
    if len(t.commonChangedOutColors)==1:
        coc = next(iter(t.commonChangedOutColors))
    else:
        coc = None
    
    bestScore = 1000
    bestFunction = partial(identityM)
    
    mergeColors = t.commonOutColors-t.totalInColors
    if len(mergeColors) == 1:
        mergeColor = next(iter(mergeColors))
    else:
        mergeColor = None
    
    for actions in combinations_with_replacement(["stop", 'l', 'r', "split", "skip"],\
                                                 len(t.fixedColors2)):
        rules = []
        for c in range(len(fixedColorsList)):
            rules.append([fixedColorsList[c], actions[c]])
            
        f = partial(drawEvolvingLines, sources=sources, rules=rules, cic=cic, \
                    fixedDirection=True, coc=coc, mergeColor=mergeColor)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        f = partial(drawEvolvingLines, sources=sources, rules=rules, cic=cic, \
                    fixedDirection=False, coc=coc, mergeColor=mergeColor)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if coc!=None:
            f = partial(drawEvolvingLines, sources=sources, rules=rules, cic=cic, \
                        fixedDirection=True, coc=coc, mergeColor=mergeColor)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            f = partial(drawEvolvingLines, sources=sources, rules=rules, cic=cic, \
                        fixedDirection=False, coc=coc, mergeColor=mergeColor)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
    f = partial(drawEvolvingLines, sources=sources, rules="convert", cic=cic, \
                    fixedDirection=False, coc=coc, mergeColor=mergeColor)  
    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        
    for asi in [True, False]:
        for stepIncrease in [None, 1, 2]:
            for steps in [1, 2, 3, 4]:
                for direction in ['r', 'l']:
                    f = partial(drawEvolvingLines, sources=sources, rules="stop", cic=cic, \
                    fixedDirection=False, alternateStepIncrease=asi, \
                    stepIncrease=stepIncrease, turnAfterNSteps=[steps, direction], mergeColor=mergeColor) 
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            
    return bestFunction

def mergeMatrices(matrices, backgroundColor, mergeColor=None):
    """
    All matrices are required to have the same shape.
    """
    result = np.zeros(matrices[0].shape, dtype=np.uint8)
    if mergeColor==None:
        for i,j in np.ndindex(matrices[0].shape):
            done=False
            for m in matrices:
                if m[i,j]!=backgroundColor:
                    result[i,j] = m[i,j]
                    done=True
                    break
            if not done:
                result[i,j] = backgroundColor
    else:
        for i,j in np.ndindex(matrices[0].shape):
            colors = set()
            for m in matrices:
                if m[i,j]!=backgroundColor:
                    colors.add(m[i,j])
            if len(colors)==0:
                result[i,j] = backgroundColor
            elif len(colors)==1:
                result[i,j] = next(iter(colors))
            else:
                result[i,j] = mergeColor
    return result
        
def drawEvolvingLines(matrix, sources, rules, cic, fixedDirection, coc=None, \
                      stepIncrease=None, alternateStepIncrease=False, \
                      turnAfterNSteps=[None, None], mergeColor=None):
    if len(sources)==0:
        return matrix.m.copy()
    fd = fixedDirection
    matrices = []
    for source in sources:
        newM = matrix.m.copy()
        for i,j in np.ndindex(matrix.shape):
            if matrix.m[i,j]==source[0]:
                if source[1]=="away":
                    if i==0:
                        if coc==None:
                            line = EvolvingLine(source[0], 'd', [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                                stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                                turnAfterNSteps=turnAfterNSteps)
                        else:
                            line = EvolvingLine(coc, 'd', [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                                stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                                turnAfterNSteps=turnAfterNSteps)
                    elif i==matrix.m.shape[0]-1:
                        if coc==None:
                            line = EvolvingLine(source[0], 'u', [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                                stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                                turnAfterNSteps=turnAfterNSteps)
                        else:
                            line = EvolvingLine(coc, 'u', [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                                stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                                turnAfterNSteps=turnAfterNSteps)
                    elif j==0:
                        if coc==None:
                            line = EvolvingLine(source[0], 'r', [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                                stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                                turnAfterNSteps=turnAfterNSteps)
                        else:
                            line = EvolvingLine(coc, 'r', [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                                stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                                turnAfterNSteps=turnAfterNSteps)
                    elif j==matrix.m.shape[1]-1:
                        if coc==None:
                            line = EvolvingLine(source[0], 'l', [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                                stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                                turnAfterNSteps=turnAfterNSteps)
                        else:
                            line = EvolvingLine(coc, 'l', [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                                stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                                turnAfterNSteps=turnAfterNSteps)
                    else:
                        return matrix.m.copy()
                else:
                    if coc==None:
                        line = EvolvingLine(source[0], source[1], [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                            stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                            turnAfterNSteps=turnAfterNSteps)
                    else:
                        line = EvolvingLine(coc, source[1], [i,j], cic, colorRules=rules, fixedDirection=fd,\
                                            stepIncrease=stepIncrease, alternateStepIncrease=alternateStepIncrease,\
                                            turnAfterNSteps=turnAfterNSteps)
                line.draw(newM)
        matrices.append(newM)
    m = mergeMatrices(matrices, next(iter(cic)), mergeColor)
    return m

# %% Crossed coordinates
    
def paintCrossedCoordinates(matrix, refColor, outColor, fixedColors=set()):
    m = matrix.m.copy()
    xCoord = set()
    yCoord = set()
    
    for i,j in np.ndindex(matrix.shape):
        if m[i,j]==refColor:
            xCoord.add(i)
            yCoord.add(j)
    
    for i in xCoord:
        for j in yCoord:
            if matrix.m[i,j] not in fixedColors:
                m[i,j] = outColor
    
    return m
    

# %% Linear Models

# If input always has the same shape and output always has the same shape
# And there is always the same number of colors in each sample    
# def trainLinearModel(t, commonColors, nChannels):
#     """
#     This function trains a linear model.
#     It is required that all the training samples have the same number of colors
#     (adding the colors in the input and in the output).
#     It is also required that all the input matrices have the same shape, and
#     all the output matrices have the same shape.
#     The colors are tried to be order in a specific way: first the colors that
#     are common to every sample (commonColors), and then the others.
#     """
#     model = Models.LinearModel(t.inShape, t.outShape, nChannels)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
#     for e in range(100):
#         optimizer.zero_grad()
#         loss = 0.0
#         for s in t.trainSamples:
#             sColors = commonColors.copy()
#             for c in s.colors:
#                 if c not in sColors:
#                     sColors.append(c)
#             rel, invRel = relDicts(sColors)
#             x = dummify(s.inMatrix.m, nChannels, rel)
#             x = torch.tensor(x).unsqueeze(0).float()
#             y = s.outMatrix.m.copy()
#             for i,j in np.ndindex(y.shape):
#                 y[i,j] = invRel[y[i,j]]
#             y = torch.tensor(y).unsqueeze(0).view(1,-1).long()
#             y_pred = model(x)
#             loss += criterion(y_pred, y)
#         loss.backward()
#         optimizer.step()
#     return model

# @torch.no_grad()
# def predictLinearModel(matrix, model, commonColors, nChannels, outShape):
#     """
#     Predict function for a model trained using trainLinearModel.
#     """
#     m = matrix.m.copy()
#     pred = np.zeros(outShape, dtype=np.uint8)
#     sColors = commonColors.copy()
#     for c in matrix.colors:
#         if c not in sColors:
#             sColors.append(c)
#     rel, invRel = relDicts(sColors)
#     if len(sColors) > nChannels:
#         return
#     x = dummify(m, nChannels, rel)
#     x = torch.tensor(x).unsqueeze(0).float()
#     x = model(x).argmax(1).squeeze(0).view(outShape).numpy()
#     for i,j in np.ndindex(outShape):
#         if x[i,j] not in rel.keys():
#             pred[i,j] = x[i,j]
#         else:
#             pred[i,j] = rel[x[i,j]][0]
#     return pred

# def trainLinearDummyModel(t):
#     """
#     This function trains a linear model.
#     The training samples will have two channels: the background color and any
#     other color. The training loop loops through all the non-background colors
#     of each sample, treating them independently.
#     """
#     model = Models.LinearModelDummy(t.inShape, t.outShape)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
#     for e in range(100):
#         optimizer.zero_grad()
#         loss = 0.0
#         for s in t.trainSamples:
#             for c in s.colors:
#                 if c != t.backgroundColor:
#                     x = dummifyColor(s.inMatrix.m, c)
#                     x = torch.tensor(x).unsqueeze(0).float()
#                     y = deBackgroundizeMatrix(s.outMatrix.m, c)
#                     y = torch.tensor(y).unsqueeze(0).long()
#                     y = y.view(1, -1)
#                     y_pred = model(x)
#                     loss += criterion(y_pred, y)
#         loss.backward()
#         optimizer.step()
#     return model
    
# @torch.no_grad()
# def predictLinearDummyModel(matrix, model, outShape, backgroundColor):
#     """
#     Predict function for a model trained using trainLinearDummyModel.
#     """
#     m = matrix.m.copy()
#     pred = np.zeros(outShape, dtype=np.uint8)
#     for c in matrix.colors:
#         if c != backgroundColor:
#             x = dummifyColor(m, c)
#             x = torch.tensor(x).unsqueeze(0).float()
#             x = model(x).argmax(1).squeeze().view(outShape).numpy()
#             for i,j in np.ndindex(outShape):
#                 if x[i,j] != 0:
#                     pred[i,j] = c
#     return pred

# def trainLinearModelShapeColor(t):
#     """
#     For trainLinearModelShapeColor we need to have the same shapes in the input
#     and in the output, and in the exact same positions. The training loop loops
#     through all the shapes of the task, and its aim is to predict the final
#     color of each shape.
#     The features of the linear model are:
#         - One feature per color in the task. Value of 1 if the shape has that
#         color, 0 otherwise.
#         - Several features representing the number of pixels of the shape.
#         Only one of these features can be equal to 1, the rest will be equal
#         to 0.
#         - 5 features to encode the number of holes of the shape (0,1,2,3 or 4)
#         - Feature encoding whether the shape is a square or not.
#         - Feature encoding whether the shape is a rectangle or not.
#         - Feature encoding whether the shape touches the border or not.
#     """
#     inColors = set.union(*t.changedInColors+t.changedOutColors) - t.unchangedColors
#     colors = list(inColors) + list(set.union(*t.changedInColors+t.changedOutColors) - inColors)
#     rel, invRel = relDicts(list(colors))
#     shapePixelNumbers = t.shapePixelNumbers
#     _,nPixelsRel = relDicts(shapePixelNumbers)
#     # inFeatures: [colors that change], [number of pixels]+1, [number of holes] (0-4),
#     # isSquare, isRectangle, isBorder
#     nInFeatures = len(inColors) + len(shapePixelNumbers) + 1 + 5 + 3
#     model = Models.SimpleLinearModel(nInFeatures, len(colors))
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
#     num_epochs = 80
#     trainShapes = []
#     for s in t.trainSamples:
#         for shapeI in range(s.inMatrix.nShapes):
#             trainShapes.append((s.inMatrix.shapes[shapeI],\
#                                 s.outMatrix.shapes[shapeI].color))
#         for e in range(num_epochs):
#             optimizer.zero_grad()
#             loss = 0.0
#             for s,label in trainShapes:
#                 inFeatures = torch.zeros(nInFeatures)
#                 if s.color in inColors:
#                     inFeatures[invRel[s.color]] = 1
#                     inFeatures[len(inColors)+nPixelsRel[s.nPixels]] = 1
#                     inFeatures[len(inColors)+len(shapePixelNumbers)+1+min(s.nHoles, 4)] = 1
#                     inFeatures[nInFeatures-1] = int(s.isSquare)
#                     inFeatures[nInFeatures-2] = int(s.isRectangle)
#                     inFeatures[nInFeatures-1] = s.isBorder
#                     #inFeatures[nInFeatures-4] = s.nHoles
#                     #inFeatures[t.nColors+5] = s.position[0].item()
#                     #inFeatures[t.nColors+6] = s.position[1].item()
#                     y = torch.tensor(invRel[label]).unsqueeze(0).long()
#                     x = inFeatures.unsqueeze(0).float()
#                     y_pred = model(x)
#                     loss += criterion(y_pred, y)
#             if loss == 0:
#                 continue
#             loss.backward()
#             optimizer.step()
#             for p in model.parameters():
#                 p.data.clamp_(min=0.05, max=1)
#     return model

# @torch.no_grad()
# def predictLinearModelShapeColor(matrix, model, colors, unchangedColors, shapePixelNumbers):
#     """
#     Predict function for a model trained using trainLinearModelShapeColor.
#     """
#     inColors = colors - unchangedColors
#     colors = list(inColors) + list(colors - inColors)
#     rel, invRel = relDicts(list(colors))
#     _,nPixelsRel = relDicts(shapePixelNumbers)
#     nInFeatures = len(inColors) + len(shapePixelNumbers) + 1 + 5 + 3
#     pred = matrix.m.copy()
#     for shape in matrix.shapes:
#         if shape.color in inColors:
#             inFeatures = torch.zeros(nInFeatures)
#             inFeatures[invRel[shape.color]] = 1
#             if shape.nPixels not in nPixelsRel.keys():
#                 inFeatures[len(inColors)+len(shapePixelNumbers)] = 1
#             else:
#                 inFeatures[len(inColors)+nPixelsRel[shape.nPixels]] = 1
#             inFeatures[len(inColors)+len(shapePixelNumbers)+1+min(shape.nHoles, 4)] = 1
#             inFeatures[nInFeatures-1] = int(shape.isSquare)
#             inFeatures[nInFeatures-2] = int(shape.isRectangle)
#             inFeatures[nInFeatures-3] = shape.isBorder
#             #inFeatures[nInFeatures-4] = shape.nHoles
#             #inFeatures[nColors+5] = shape.position[0].item()
#             #inFeatures[nColors+6] = shape.position[1].item()
#             x = inFeatures.unsqueeze(0).float()
#             y = model(x).squeeze().argmax().item()
#             pred = changeColorShapes(pred, [shape], rel[y][0])
#     return pred

# %% LSTM
# def prepare_sequence(seq, to_ix):
#     """
#     Utility function for LSTM.
#     """
#     idxs = [to_ix[w] for w in seq]
#     return torch.tensor(idxs, dtype=torch.long)

# def trainLSTM(t, inColors, colors, inRel, outRel, reverse, order):
#     """
#     This function tries to train a model that colors shapes according to a
#     sequence.
#     """
#     EMBEDDING_DIM = 10
#     HIDDEN_DIM = 10
#     model = Models.LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(inColors), len(colors))
#     loss_function = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     num_epochs = 150
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         loss = 0.0
#         for s in t.trainSamples:
#             inShapes = [shape for shape in s.inMatrix.shapes if shape.color in inColors]
#             inSeq = sorted(inShapes, key=lambda x: (x.position[order[0]], x.position[order[1]]), reverse=reverse)
#             inSeq = [shape.color for shape in inSeq]
#             outShapes = [shape for shape in s.outMatrix.shapes if shape.color in colors]
#             targetSeq = sorted(outShapes, key=lambda x: (x.position[order[0]], x.position[order[1]]), reverse=reverse)
#             targetSeq = [shape.color for shape in targetSeq]
#             inSeq = prepare_sequence(inSeq, inRel)
#             targetSeq = prepare_sequence(targetSeq, outRel)
#             tag_scores = model(inSeq)
#             loss += loss_function(tag_scores, targetSeq)
#         loss.backward()
#         optimizer.step()
#     return model
    
# @torch.no_grad()
# def predictLSTM(matrix, model, inColors, colors, inRel, rel, reverse, order):
#     """
#     Predict function for a model trained using trainLSTM.
#     """
#     m = matrix.m.copy()
#     inShapes = [shape for shape in matrix.shapes if shape.color in inColors]
#     if len(inShapes)==0:
#         return m
#     sortedShapes = sorted(inShapes, key=lambda x: (x.position[order[0]], x.position[order[1]]), reverse=reverse)
#     inSeq = [shape.color for shape in sortedShapes]
#     inSeq = prepare_sequence(inSeq, inRel)
#     pred = model(inSeq).argmax(1).numpy()
#     for shapeI in range(len(sortedShapes)):
#         m = changeColorShapes(m, [sortedShapes[shapeI]], rel[pred[shapeI]][0])
#     return m
             
# def getBestLSTM(t):
#     """
#     This function tries to find out which one is the best-fitting LSTM model
#     for the task t. The LSTM models try to change the color of shapes to fit
#     sequences. Examples are tasks 175, 331, 459 or 594.
#     4 LSTM models are trained, considering models that order shapes by X
#     coordinage, models that order them by Y coordinate, and considering both
#     directions of the sequence (normal and reverse).
#     """
#     colors = set.union(*t.changedInColors+t.changedOutColors)
#     inColors = colors - t.unchangedColors
#     if len(inColors) == 0:
#         return partial(identityM)
#     _,inRel = relDicts(list(inColors))
#     colors = list(inColors) + list(colors - inColors)
#     rel, outRel = relDicts(colors)
    
#     for s in t.trainSamples:
#         inShapes = [shape for shape in s.inMatrix.shapes if shape.color in inColors]
#         outShapes = [shape for shape in s.outMatrix.shapes if shape.color in colors]
#         if len(inShapes) != len(outShapes) or len(inShapes) == 0:
#             return partial(identityM)
            
#     reverse = [True, False]
#     order = [(0,1), (1,0)]    
#     bestScore = 1000
#     for r, o in product(reverse, order):        
#         model = trainLSTM(t, inColors=inColors, colors=colors, inRel=inRel,\
#                           outRel=outRel, reverse=r, order=o)
        
#         score = 0
#         for s in t.trainSamples:
#             m = predictLSTM(s.inMatrix, model, inColors, colors, inRel, rel, r, o)
#             score += incorrectPixels(m, s.outMatrix.m)
#         if score < bestScore:
#             bestScore=score
#             ret = partial(predictLSTM, model=model, inColors=inColors,\
#                           colors=colors, inRel=inRel, rel=rel, reverse=r, order=o) 
#             if bestScore==0:
#                 return ret
#     return ret

# %% Other utility functions

def insertShape(matrix, shape):
    """
    Given a matrix (numpy.ndarray) and a Task.Shape, this function returns the
    same matrix but with the shape inserted.
    """
    m = matrix.copy()
    shapeM = shape.m.copy()
    for i,j in np.ndindex(shape.shape):
        if shapeM[i,j] != 255:
            if shape.position[0]+i<matrix.shape[0] and shape.position[1]+j<matrix.shape[1]\
                    and shape.position[0]+i >= 0 and shape.position[1]+j >= 0:
                m[tuple(map(operator.add, (i,j), shape.position))] = shapeM[i,j]
    return m

def deleteShape(matrix, shape, backgroundColor):
    """
    Given a matrix (numpy.ndarray) and a Task.Shape, this function substitutes
    the shape by the background color of the matrix.
    """
    m = matrix.copy()
    for c in shape.pixels:
        m[tuple(map(operator.add, c, shape.position))] = backgroundColor
    return m

def symmetrizeSubmatrix(matrix, ud=False, lr=False, rotation=False, newColor=None, subShape=None):
    """
    Given a Task.Matrix, make the non-background part symmetric
    """
    m = matrix.m.copy()
    # print("m init", m)
    bC = matrix.backgroundColor
    if np.all(m == bC):
        return m
    x1, x2, y1, y2 = 0, m.shape[0]-1, 0, m.shape[1]-1
    while x1 <= x2 and np.all(m[x1,:] == bC):
        x1 += 1
    while x2 >= x1 and np.all(m[x2,:] == bC):
        x2 -= 1
    while y1 <= y2 and np.all(m[:,y1] == bC):
        y1 += 1
    while y2 >= y1 and np.all(m[:,y2] == bC):
        y2 -= 1
    subMat = m[x1:x2+1,y1:y2+1].copy()
    
    if subShape == None:
        symList = []
        if ud:
            symList.append(np.flipud(subMat.copy()))
            if lr:
                symList.append(np.fliplr(subMat.copy()))
                symList.append(np.fliplr(np.flipud(subMat.copy())))
            elif lr:
                symList.append(np.fliplr(subMat.copy()))
        elif rotation:
            for x in range(1,4):
                symList.append(np.rot90(subMat.copy(),x))
        for newSym in symList:
            score, bestScore = 0, 0
            bestX, bestY = 0, 0
            if (m.shape[0] >= newSym.shape[0]) and (m.shape[1] >= newSym.shape[1]):
                for i, j in np.ndindex((m.shape[0]-newSym.shape[0]+1,m.shape[1]-newSym.shape[1]+1)):
                    score = np.count_nonzero(np.logical_or(m[i:i+newSym.shape[0],j:j+newSym.shape[1]] == newSym, newSym == bC))
                    if score > bestScore:
                        bestX, bestY = i, j
                        bestScore = score
                for i, j in np.ndindex(newSym.shape):
                    if newSym[i,j] != bC:
                        if newColor == None:
                            m[bestX+i, bestY+j] = newSym[i,j]
                        else:
                            if bestX+i >= m.shape[0] or bestY+j >= m.shape[1]:
                                continue

                            if m[bestX+i,bestY+j] == bC:
                                m[bestX+i, bestY+j] = newColor

    else:
        found = False
        for x in range(subMat.shape[0]-subShape.shape[0]+1):
            for y in range(subMat.shape[1]-subShape.shape[1]+1):
                #if np.all(m[x1+x:x1+x+subShape.shape[0],y1+y:y1+y+subShape.shape[1]]==subShape.m):
                if np.all(np.equal(m[x1+x:x1+x+subShape.shape[0],y1+y:y1+y+subShape.shape[1]] == bC,subShape.m == 255)):
                    found = True   
                    break
            if found:
                break
        if not found:
            return m
        if ud and lr:
            if 2*x+x1+subShape.shape[0] > m.shape[0] or 2*y+y1+subShape.shape[0]> m.shape[1]:
                return m
            for i in range(subMat.shape[0]):
                for j in range(subMat.shape[1]):
                    if subMat[i][j] != bC:
                        m[2*x+x1+subShape.shape[0]-i-1,y1+j] = subMat[i,j]
                        m[x1+i,2*y+y1+subShape.shape[0]-j-1] = subMat[i,j]
                        m[2*x+x1+subShape.shape[0]-i-1,2*y+y1+subShape.shape[0]-j-1] = subMat[i,j]
        elif rotation:
            if x1+y+x+subShape.shape[0] > m.shape[0] or y1+x+y+subShape.shape[1] > m.shape[1]\
                or y1+y-x+subMat.shape[0] >= m.shape[0] or x1+x-y+subMat.shape[1] >= m.shape[1]\
                or x1+2*x+subShape.shape[0] > m.shape[0] or y1+2*y+subShape.shape[0] > m.shape[1]:
                return m
            for i in range(subMat.shape[0]):
                for j in range(subMat.shape[1]):
                    if subMat[i,j] != bC:
                        m[x1+x+subShape.shape[0]+y-j-1,y1+y-x+i] = subMat[i,j]
                        m[x1+x-y+j,y1+y+subShape.shape[0]+x-i-1] = subMat[i,j]
                        m[x1+2*x+subShape.shape[0]-i-1,y1+2*y+subShape.shape[0]-j-1] = subMat[i,j]        
    return m

def getBestSymmetrizeSubmatrix(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    rotation, lr, ud, newColor = False, False, False, None
    croppedSamples = [cropAllBackground(s.outMatrix).copy() for s in t.trainSamples]
    if all(np.all(np.flipud(m)==m) for m in croppedSamples):
        lr = True
    if all(np.all(np.fliplr(m)==m) for m in croppedSamples):    
        ud = True    
    if all(m.shape[0]==m.shape[1] for m in croppedSamples):
        if all(np.all(np.rot90(m)==m) for m in croppedSamples):
            rotation = True
        elif hasattr(t,'colorChanges') and all(np.all((m==t.backgroundColor) == (np.rot90(m)==t.backgroundColor)) for m in croppedSamples):
            rotation = True
            newColor = list(t.colorChanges)[0][1]
    for sh in t.commonInDShapes:
        bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeSubmatrix,\
                                                lr=lr,ud=ud,rotation=rotation,subShape=sh, newColor=newColor), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeSubmatrix,lr=lr,\
                                                ud=ud,rotation=rotation, newColor=newColor), bestScore, bestFunction)
    return bestFunction

def colorMap(matrix, cMap):
    """
    cMap is a dict of color changes. Each input color can map to one and only
    one output color. Only valid if t.sameIOShapes.
    """
    m = matrix.m.copy()
    for i,j in np.ndindex(m.shape):
        if m[i,j] in cMap.keys(): # Otherwise, it means m[i,j] unchanged
            m[i,j] = cMap[matrix.m[i,j]]
    return m

def changeColorShapes(matrix, shapes, color):
    """
    Given a matrix (numpy.ndarray), a list of Task.Shapes (they are expected to
    be present in the matrix) and a color, this function returns the same
    matrix, but with the shapes of the list having the given color.
    """
    if len(shapes) == 0:
        return matrix
    m = matrix.copy()
    if color not in list(range(10)):
        return m
    for s in shapes:
        for c in s.pixels:
            m[tuple(map(operator.add, c, s.position))] = color
    return m

def changeShapes(m, inColor, outColor, bigOrSmall=None, isBorder=None):
    """
    Given a Task.Matrix, this function changes the Task.Shapes of the matrix
    that have color inColor to having the color outColor, if they satisfy the
    given conditions bigOrSmall (is the shape the smallest/biggest one?) and
    isBorder.
    """
    return changeColorShapes(m.m.copy(), m.getShapes(inColor, bigOrSmall, isBorder), outColor)

def paintShapesInHalf(matrix, shapeColor, color, half, diagonal=False, middle=None):
    """
    Half can be 'u', 'd', 'l' or 'r'.
    """
    m = matrix.m.copy()
    if diagonal:
        shapesToPaint = [shape for shape in matrix.dShapes if shape.color==shapeColor]
    else:
        shapesToPaint = [shape for shape in matrix.shapes if shape.color==shapeColor]
    
    for shape in shapesToPaint:
        if (shape.shape[0]%2)==0 or middle!=None:
            if middle==True:
                iLimit=int((shape.shape[0]+1)/2)
            else:
                iLimit=int(shape.shape[0]/2)
            if half=='u':
                for i,j in np.ndindex((iLimit, shape.shape[1])):
                    if shape.m[i,j]==shape.color:
                        m[shape.position[0]+i, shape.position[1]+j] = color
            if half=='d':
                for i,j in np.ndindex((iLimit, shape.shape[1])):
                    if shape.m[shape.shape[0]-1-i,j]==shape.color:
                        m[shape.position[0]+shape.shape[0]-1-i, shape.position[1]+j] = color
        if (shape.shape[1]%2)==0 or middle!=None:
            if middle==True:
                jLimit=int((shape.shape[1]+1)/2)
            else:
                jLimit=int(shape.shape[1]/2)
            if half=='l':
                for i,j in np.ndindex((shape.shape[0], jLimit)):
                    if shape.m[i,j]==shape.color:
                        m[shape.position[0]+i, shape.position[1]+j] = color
            if half=='r':
                for i,j in np.ndindex((shape.shape[0], jLimit)):
                    if shape.m[i,shape.shape[1]-1-j]==shape.color:
                        m[shape.position[0]+i, shape.position[1]+shape.shape[1]-1-j] = color
            
    return m

def getBestPaintShapesInHalf(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    for half in ['u', 'd', 'l', 'r']:
        for cic in t.commonChangedInColors:
            for coc in t.commonChangedOutColors:
                for middle, diagonal in product([None, True, False], [True, False]):
                    f = partial(paintShapesInHalf, shapeColor=cic, color=coc,\
                                half=half, diagonal=diagonal, middle=middle)
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                    if bestScore==0:
                        return bestFunction
    return bestFunction

# %% Paint shape from border color (506,753,754)

def paintShapeFromBorderColor(matrix, shapeColors, fixedColors, diagonals=False):
    m = matrix.m.copy()
    shapesToChange = [shape for shape in matrix.shapes if shape.color in shapeColors]
    
    for shape in shapesToChange:
        neighbourColors = Counter()
        for i,j in np.ndindex(shape.shape):
            if shape.m[i,j]!=255:
                if diagonals:
                    neighbourColors += Counter(getAllNeighbourColors(matrix.m,\
                                               shape.position[0]+i, shape.position[1]+j,\
                                               kernel=3, border=-1))
                else:
                    neighbourColors += Counter(getNeighbourColors(matrix.m,\
                                               shape.position[0]+i, shape.position[1]+j,\
                                               border=-1))
        for color in fixedColors|shapeColors|set([-1]):
            neighbourColors.pop(color, None)
        if len(neighbourColors)>0:
            newColor = max(neighbourColors.items(), key=operator.itemgetter(1))[0]
            m = changeColorShapes(m, [shape], newColor)
    return m

def getBestPaintShapeFromBorderColor(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    shapeColors = t.commonChangedInColors
    fixedColors = t.fixedColors
    for diagonals in [True, False]:
        f = partial(paintShapeFromBorderColor, shapeColors=shapeColors,\
                    fixedColors=fixedColors, diagonals=diagonals)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
    return bestFunction

# %% Things with features
def isFixedShape(shape, fixedShapeFeatures):
    return shape.hasFeatures(fixedShapeFeatures)

def hasFeatures(candidate, reference):
    if all([i==False for i in reference]):
        return False
    for i in range(len(reference)):
        if reference[i] and not candidate[i]:
            return False
    return True

# %% Change Colors with features

def getClosestFixedShapeColor(shape, fixedShapes):
    def getDistance(x1, x2):
        x, y = sorted((x1, x2))
        if x[0] <= x[1] < y[0] and all( y[0] <= y[1] for y in (x1,x2)):
            return y[0] - x[1]
        return 0
    
    color = 0
    minDistance = 1000
    for fs in fixedShapes:
        xDist = getDistance([fs.position[0],fs.position[0]+fs.shape[0]-1], \
                            [shape.position[0],shape.position[0]+shape.shape[0]-1])
        yDist = getDistance([fs.position[1],fs.position[1]+fs.shape[1]-1], \
                            [shape.position[1],shape.position[1]+shape.shape[1]-1])
        
        if xDist+yDist < minDistance:
            minDistance = xDist+yDist
            color = fs.color
    return color

def getShapeFeaturesForColorChange(t, fixedShapeFeatures=None, fixedColors=None,\
                                   predict=False):  
    shapeFeatures = []
    
    if predict:
        matrices = [t]
    else:
        matrices = [s.inMatrix for s in t.trainSamples]
                  
    for m in range(len(matrices)):
        # Smallest and biggest shapes:
        biggestShape = 0
        smallestShape = 1000
        for shape in matrices[m].shapes:
            if shape.color not in fixedColors:
                if shape.nPixels>biggestShape:
                    biggestShape=shape.nPixels
                if shape.nPixels<smallestShape:
                    smallestShape=shape.nPixels
                    
        # Closest fixed shape
        if predict:
            fixedShapes = []
            for shape in matrices[0].shapes:
                if shape.hasFeatures(fixedShapeFeatures):
                    fixedShapes.append(shape)
        else:
            fixedShapes = t.trainSamples[m].fixedShapes
        
        for shape in matrices[m].shapes:
            shFeatures = []
            for c in range(10):
                shFeatures.append(shape.color==c)
            shFeatures.append(shape.isBorder)
            shFeatures.append(not shape.isBorder)
            shFeatures.append(shape.lrSymmetric)
            shFeatures.append(shape.udSymmetric)
            shFeatures.append(shape.d1Symmetric)
            shFeatures.append(shape.d2Symmetric)
            shFeatures.append(shape.isSquare)
            shFeatures.append(shape.isRectangle)
            for nPix in range(1,30):
                shFeatures.append(shape.nPixels==nPix)
            for nPix in range(1,6):
                shFeatures.append(shape.nPixels>nPix)
            for nPix in range(2,7):
                shFeatures.append(shape.nPixels<nPix)
            shFeatures.append((shape.nPixels%2)==0)
            shFeatures.append((shape.nPixels%2)==1)
            for h in range(5):
                shFeatures.append(shape.nHoles==h)
            shFeatures.append(shape.nPixels==biggestShape)
            shFeatures.append(shape.nPixels==smallestShape)
            #shFeatures.append(self.isUniqueShape(sh))
            #shFeatures.append(not self.isUniqueShape(sh))
            closestFixedShapeColor = getClosestFixedShapeColor(shape, fixedShapes)
            for c in range(10):
                shFeatures.append(closestFixedShapeColor==c)
                
            shapeFeatures.append(shFeatures)
            
    return shapeFeatures

def getColorChangesWithFeatures(t):
    shapeFeatures = getShapeFeaturesForColorChange(t, fixedColors=t.fixedColors)
    colorChangesWithFeatures = {}
    nFeatures = len(shapeFeatures[0])
    trueList = []
    falseList = []
    for i in range(nFeatures):
        trueList.append(True)
        falseList.append(False)
    for c in t.totalOutColors:
        colorChangesWithFeatures[c] = trueList
    changeCounter = Counter() # How many shapes change to color c?
    # First, initialise the colorChangesWithFeatures. For every outColor, we
    # detect which features are True for all the shapes that change to that color.
    shapeCounter = 0            
    for sample in t.trainSamples:
        for shape in sample.inMatrix.shapes:
            shapeChanges = False
            for i,j in np.ndindex(shape.shape):
                if shape.m[i,j]!=255:
                    if sample.outMatrix.m[shape.position[0]+i,shape.position[1]+j]!=shape.m[i,j]:
                        color = sample.outMatrix.m[shape.position[0]+i,shape.position[1]+j]
                        shapeChanges=True
                        break
            if shapeChanges:
                changeCounter[color] += 1
                colorChangesWithFeatures[color] = \
                [colorChangesWithFeatures[color][x] and shapeFeatures[shapeCounter][x]\
                 for x in range(nFeatures)]
            shapeCounter += 1
    # Now, there might be more True values than necessary in a certain entry
    # of colorChangesWithFeatures. Therefore, we try to determine the minimum
    # number of necessary True features.
    for c in t.totalOutColors:
        if colorChangesWithFeatures[c] == trueList:
            continue
        trueIndices = [i for i, x in enumerate(colorChangesWithFeatures[c]) if x]
        # First, check if only one feature is enough
        goodIndices = []
        for index in trueIndices:
            trueCount = 0
            featureList = falseList.copy()
            featureList[index] = True
            for sf in shapeFeatures:
                if hasFeatures(sf, featureList):
                    trueCount += 1
            # If the true count matches the number of changed shapes, we're done!
            if trueCount == changeCounter[c]:
                goodIndices.append(index)
        if len(goodIndices) > 0:
            featureList = falseList.copy()
            for index in goodIndices:
                featureList[index] = True
            colorChangesWithFeatures[c] = featureList
        # If we're not done, then check with combinations of 2 features
        else:
            for i,j in combinations(trueIndices, 2):
                trueCount = 0
                featureList = falseList.copy()
                featureList[i] = True
                featureList[j] = True
                for sf in shapeFeatures:
                    if hasFeatures(sf, featureList):
                        trueCount += 1
                # If the true count matches the number of changed shapes, we're done!
                if trueCount == changeCounter[c]:
                    colorChangesWithFeatures[c] = featureList
                    break   
                    
    return colorChangesWithFeatures

def changeShapesWithFeatures(matrix, ccwf, fixedColors, fixedShapeFeatures):
    """
    ccwp stands for 'color change with properties'. It's a dictionary. Its keys
    are integers encoding the color of the output shape, and its values are the
    properties that the input shape has to satisfy in order to execute the
    color change.
    """
    featureList = getShapeFeaturesForColorChange(matrix, fixedColors=fixedColors,\
                                                 fixedShapeFeatures=fixedShapeFeatures,\
                                                 predict=True)
    m = matrix.m.copy()
    sortedCcwf = {k: v for k, v in sorted(ccwf.items(), key=lambda item: sum(item[1]))}
    for color in sortedCcwf.keys():
        for sh in range(len(matrix.shapes)):
            if (matrix.shapes[sh].color in fixedColors):# or \
            #(matrix.shapes[sh].hasFeatures(fixedShapeFeatures)):
                continue
            if hasFeatures(featureList[sh], ccwf[color]):
                m = changeColorShapes(m, [matrix.shapes[sh]], color)
                #break
    return m


# %% Change pixels with features

def pixelRecolor(t):
    """
    if t.sameIOShapes
    """
    Input = [s.inMatrix.m for s in t.trainSamples]
    Output = [s.outMatrix.m for s in t.trainSamples]
        
    Best_Dict = -1
    Best_Q1 = -1
    Best_Q2 = -1
    Best_v = -1
    
    # v ranges from 0 to 3. This gives an extra flexibility of measuring distance from any of the 4 corners
    Pairs = []
    for t in range(15):
        for Q1 in range(1,8):
            for Q2 in range(1,8):
                if Q1+Q2 == t:
                    Pairs.append((Q1,Q2))
                    
    for Q1, Q2 in Pairs:
        for v in range(4):
            if Best_Dict != -1:
                continue
            possible = True
            Dict = {}
            
            for x, y in zip(Input, Output):
                n = len(x)
                k = len(x[0])
                for i in range(n):
                    for j in range(k):
                        if v == 0 or v ==2:
                            p1 = i%Q1
                        else:
                            p1 = (n-1-i)%Q1
                        if v == 0 or v ==3:
                            p2 = j%Q2
                        else :
                            p2 = (k-1-j)%Q2
                        color1 = x[i][j]
                        color2 = y[i][j]
                        if color1 != color2:
                            rule = (p1, p2, color1)
                            if rule not in Dict:
                                Dict[rule] = color2
                            elif Dict[rule] != color2:
                                possible = False
            if possible:
                
                # Let's see if we actually solve the problem
                for x, y in zip(Input, Output):
                    n = len(x)
                    k = len(x[0])
                    for i in range(n):
                        for j in range(k):
                            if v == 0 or v ==2:
                                p1 = i%Q1
                            else:
                                p1 = (n-1-i)%Q1
                            if v == 0 or v ==3:
                                p2 = j%Q2
                            else :
                                p2 = (k-1-j)%Q2
                           
                            color1 = x[i][j]
                            rule = (p1,p2,color1)
                            
                            if rule in Dict:
                                color2 = 0 + Dict[rule]
                            else:
                                color2 = 0 + y[i][j]
                            if color2 != y[i][j]:
                                possible = False 
                if possible:
                    Best_Dict = Dict
                    Best_Q1 = Q1
                    Best_Q2 = Q2
                    Best_v = v
        
    if Best_Dict == -1:
        return [-1]#meaning that we didn't find a rule that works for the traning cases
    else:
        return [Best_Dict, Best_v, Best_Q1, Best_Q2]
    
def executePixelRecolor(matrix, Best_Dict, Best_v, Best_Q1, Best_Q2):
    m = np.zeros(matrix.shape, dtype = np.uint8)
    for i,j in np.ndindex(matrix.shape):
        if Best_v == 0 or Best_v ==2:
            p1 = i%Best_Q1
        else:
            p1 = (matrix.shape[0]-1-i)%Best_Q1
        if Best_v == 0 or Best_v ==3:
            p2 = j%Best_Q2
        else :
            p2 = (matrix.shape[1]-1-j)%Best_Q2
       
        color1 = matrix.m[i,j]
        rule = (p1, p2, color1)
        if (p1, p2, color1) in Best_Dict:
            m[i][j] = 0 + Best_Dict[rule]
        else:
            m[i][j] = 0 + color1
 
    return m
    


def doRulesWithReference(m, reference, rules):
    for i,j in np.ndindex(m.shape):
        y = (m[i,j], reference[i,j])
        if y in rules.keys():
            m[i,j] = rules[y]
    return m    

def doPixelMod2Row(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesRow = np.ones(m.shape[1], dtype=np.uint8)
    for i in range(m.shape[0]):
        if i%2 == 0:
            reference[i,:] = onesRow.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod3Row(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesRow = np.ones(m.shape[1], dtype=np.uint8)
    twosRow = np.full(m.shape[1], 2, dtype=np.uint8)
    for i in range(m.shape[0]):
        if i%3 == 0:
            reference[i,:] = onesRow.copy()
        elif i%3 == 1:
            reference[i,:] = twosRow.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod2RowReverse(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesRow = np.ones(m.shape[1], dtype=np.uint8)
    for i in range(m.shape[0]):
        if i%2 == 0:
            reference[m.shape[0]-i-1,:] = onesRow.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod3RowReverse(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesRow = np.ones(m.shape[1], dtype=np.uint8)
    twosRow = np.full(m.shape[1], 2, dtype=np.uint8)
    for i in range(m.shape[0]):
        if i%3 == 0:
            reference[m.shape[0]-i-1,:] = onesRow.copy()
        elif i%3 == 1:
            reference[m.shape[0]-i-1,:] = twosRow.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod2Col(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesCol = np.ones(m.shape[0], dtype=np.uint8)
    for j in range(m.shape[1]):
        if j%2 == 0:
            reference[:,j] = onesCol.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod3Col(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesCol = np.ones(m.shape[0], dtype=np.uint8)
    twosCol = np.full(m.shape[0], 2, dtype=np.uint8)
    for j in range(m.shape[1]):
        if j%3 == 0:
            reference[:,j] = onesCol.copy()
        elif j%3 == 1:
            reference[:,j] = twosCol.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod2ColReverse(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesCol = np.ones(m.shape[0], dtype=np.uint8)
    for j in range(m.shape[1]):
        if j%2 == 0:
            reference[:,m.shape[1]-j-1] = onesCol.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod3ColReverse(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    onesCol = np.ones(m.shape[0], dtype=np.uint8)
    twosCol = np.full(m.shape[0], 2, dtype=np.uint8)
    for j in range(m.shape[1]):
        if j%3 == 0:
            reference[:,m.shape[1]-j-1] = onesCol.copy()
        elif j%3 == 1:
            reference[:,m.shape[1]-j-1] = twosCol.copy()
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod2Alternate(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        reference[i,j] = (i+j)%2
    m = doRulesWithReference(m, reference, rules)
    return m

def doPixelMod3Alternate(matrix, rules):
    m = matrix.m.copy()
    reference = np.zeros(m.shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        reference[i,j] = (i+j)%3
    m = doRulesWithReference(m, reference, rules)
    return m

def getPixelChangeCriteria(t):
    # Row
    # Mod 2
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesRow = np.ones(sample.inMatrix.shape[1], dtype=np.uint8)
        for i in range(sample.inMatrix.shape[0]):
            if i%2 == 0:
                reference[i,:] = onesRow.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod2Row, rules=x)
    # Mod 3
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesRow = np.ones(sample.inMatrix.shape[1], dtype=np.uint8)
        twosRow = np.full(sample.inMatrix.shape[1], 2, dtype=np.uint8)
        for i in range(sample.inMatrix.shape[0]):
            if i%3 == 0:
                reference[i,:] = onesRow.copy()
            elif i%3 == 1:
                reference[i,:] = twosRow.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod3Row, rules=x)
    
    # Row Reverse
    # Mod 2
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesRow = np.ones(sample.inMatrix.shape[1], dtype=np.uint8)
        for i in range(sample.inMatrix.shape[0]):
            if i%2 == 0:
                reference[sample.inMatrix.shape[0]-i-1,:] = onesRow.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod2RowReverse, rules=x)
    # Mod 3
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesRow = np.ones(sample.inMatrix.shape[1], dtype=np.uint8)
        twosRow = np.full(sample.inMatrix.shape[1], 2, dtype=np.uint8)
        for i in range(sample.inMatrix.shape[0]):
            if i%3 == 0:
                reference[sample.inMatrix.shape[0]-i-1,:] = onesRow.copy()
            elif i%3 == 1:
                reference[sample.inMatrix.shape[0]-i-1,:] = twosRow.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod3RowReverse, rules=x)

    # Col
    # Mod 2
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesCol = np.ones(sample.inMatrix.shape[0], dtype=np.uint8)
        for j in range(sample.inMatrix.shape[1]):
            if j%2 == 0:
                reference[:,j] = onesCol.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod2Col, rules=x)
    # Mod 3
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesCol = np.ones(sample.inMatrix.shape[0], dtype=np.uint8)
        twosCol = np.full(sample.inMatrix.shape[0], 2, dtype=np.uint8)
        for j in range(sample.inMatrix.shape[1]):
            if j%3 == 0:
                reference[:,j] = onesCol.copy()
            elif j%3 == 1:
                reference[:,j] = twosCol.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod3Col, rules=x)
    
    # Col Reverse
    # Mod 2
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesCol = np.ones(sample.inMatrix.shape[0], dtype=np.uint8)
        for j in range(sample.inMatrix.shape[1]):
            if j%2 == 0:
                reference[:,sample.inMatrix.shape[1]-j-1] = onesCol.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod2ColReverse, rules=x)
    # Mod 3
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        onesCol = np.ones(sample.inMatrix.shape[0], dtype=np.uint8)
        twosCol = np.full(sample.inMatrix.shape[0], 2, dtype=np.uint8)
        for j in range(sample.inMatrix.shape[1]):
            if j%3 == 0:
                reference[:,sample.inMatrix.shape[1]-j-1] = onesCol.copy()
            elif j%3 == 1:
                reference[:,sample.inMatrix.shape[1]-j-1] = twosCol.copy()
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod3ColReverse, rules=x)
    
    # Alternate
    # Mod2
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        for i,j in np.ndindex(sample.inMatrix.shape):
            reference[i,j] = (i+j)%2
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod2Alternate, rules=x)
    # Mod3
    x = {}
    for sample in t.trainSamples:
        reference = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
        for i,j in np.ndindex(sample.inMatrix.shape):
            reference[i,j] = (i+j)%3
        for i,j in np.ndindex(reference.shape):
            y = (sample.inMatrix.m[i,j], reference[i,j])
            if y in x.keys():
                x[y].add(sample.outMatrix.m[i,j])
            else:
                x[y] = set([sample.outMatrix.m[i,j]])
    x = {k:next(iter(v)) for k,v in x.items() if len(v)==1}
    x = {k:v for k,v in x.items() if v not in t.fixedColors}
    if len(x)>0:
        return partial(doPixelMod3Alternate, rules=x)
    
    return 0    

# %% Surround Shape

def surroundShape(matrix, shape, color, fixedColors, nSteps = None, forceFull=False, \
                  stepIsShape=False, stepIsNHoles=False):

    m = matrix.copy()
    shapeMatrix = shape.m.copy()
    
    if nSteps==None:
        if stepIsShape:
            nSteps = int(shape.shape[0]/2)
        elif stepIsNHoles:
            nSteps = shape.nHoles
        else:
            nSteps = 15
    
    step = 0
    
    while step<nSteps:
        step += 1
        if forceFull:
            if shape.position[0]-step<0 or shape.position[0]+shape.shape[0]+step>matrix.shape[0] or\
            shape.position[1]-step<0 or shape.position[1]+shape.shape[1]+step>matrix.shape[1]:
                step -= 1
                break
            
            done = False
            for i in range(shape.position[0]-step, shape.position[0]+shape.shape[0]+step):
                if matrix[i, shape.position[1]-step] in fixedColors:
                    step -= 1
                    done = True
                    break
                if matrix[i, shape.position[1]+shape.shape[1]+step-1] in fixedColors:
                    step -= 1
                    done = True
                    break
            if done:
                break
            for j in range(shape.position[1]-step, shape.position[1]+shape.shape[1]+step):
                if matrix[shape.position[0]-step, j] in fixedColors:
                    step -= 1
                    done = True
                    break
                if matrix[shape.position[0]+shape.shape[0]+step-1, j] in fixedColors:
                    step -= 1
                    done = True
                    break
            if done:
                break
        
        row = np.full(shapeMatrix.shape[1], -1, dtype=np.uint8)
        col = np.full(shapeMatrix.shape[0]+2, -1, dtype=np.uint8)
        newM = shapeMatrix.copy() 
        newM = np.vstack([row,newM,row])
        newM = np.column_stack([col,newM,col])
        
        for i in range(newM.shape[0]):
            for j in range(newM.shape[1]):
                if newM[i,j] != 255:
                    newM[i, j-1] = color
                    break
            for j in reversed(range(newM.shape[1])):
                if newM[i,j] != 255:
                    newM[i, j+1] = color
                    break
                    
        for j in range(newM.shape[1]):
            for i in range(newM.shape[0]):
                if newM[i,j] != 255:
                    newM[i-1, j] = color
                    break
            for i in reversed(range(newM.shape[0])):
                if newM[i,j] != 255:
                    newM[i+1, j] = color
                    break
                    
        shapeMatrix = newM.copy()
                    
    for i,j in np.ndindex(shapeMatrix.shape):
        if shape.position[0]-step+i<0 or shape.position[0]-step+i>=matrix.shape[0] or \
        shape.position[1]-step+j<0 or shape.position[1]-step+j>=matrix.shape[1]:
            continue
        if shapeMatrix[i,j] != 255:
            m[shape.position[0]-step+i, shape.position[1]-step+j] = shapeMatrix[i,j]
        
    return m
    
def surroundAllShapes(matrix, shapeColor, surroundColor, fixedColors, nSteps=None,\
                      forceFull=False, stepIsShape=False, stepIsNHoles=False):
    m = matrix.m.copy()
    shapesToSurround = [s for s in matrix.shapes if s.color == shapeColor]
    if stepIsShape:
        shapesToSurround = [s for s in shapesToSurround if s.isSquare]
    for s in shapesToSurround:
        m = surroundShape(m, s, surroundColor, fixedColors, nSteps=nSteps,\
                          forceFull=forceFull, stepIsShape=stepIsShape,\
                          stepIsNHoles=stepIsNHoles)
    return m

def getBestSurroundShapes(t):    
    bestScore = 1000
    bestFunction = partial(identityM)
    
    for fc in t.fixedColors:
        for coc in t.commonChangedOutColors:
            f = partial(surroundAllShapes, shapeColor=fc, surroundColor=coc, \
                        fixedColors=t.fixedColors, forceFull=True)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            
            f = partial(surroundAllShapes, shapeColor=fc, surroundColor=coc, \
                            fixedColors=t.fixedColors, stepIsShape=True)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            
            f = partial(surroundAllShapes, shapeColor=fc, surroundColor=coc, \
                            fixedColors=t.fixedColors, forceFull=True, stepIsShape=True)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            
            f = partial(surroundAllShapes, shapeColor=fc, surroundColor=coc, \
                            fixedColors=t.fixedColors, forceFull=False, stepIsNHoles=True)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            
            f = partial(surroundAllShapes, shapeColor=fc, surroundColor=coc, \
                            fixedColors=t.fixedColors, forceFull=True, stepIsNHoles=True)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            
            for nSteps in range(1,4):
                f = partial(surroundAllShapes, shapeColor=fc, surroundColor=coc, \
                            fixedColors=t.fixedColors, nSteps=nSteps)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                
                f = partial(surroundAllShapes, shapeColor=fc, surroundColor=coc, \
                            fixedColors=t.fixedColors, nSteps=nSteps, forceFull=True)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
            
    return bestFunction
        
# %% Extend Color

def extendColor(matrix, direction, cic, fixedColors, color=None, sourceColor=None,\
                deleteExtensionColors=set(), deleteIfBorder=False, \
                breakAtFixedColor=False, mergeColor=None):
    #m = matrix.m.copy()
    
    if sourceColor==None:
        sourceColor=color
        
    matrices = []
    
    # Vertical
    if direction=='all' or direction=='hv' or direction=='v' or direction=='u':
        for j in range(matrix.shape[1]):
            colorCells=False
            start = matrix.shape[0]-1
            for i in reversed(range(matrix.shape[0])):
                if color==None:
                    if matrix.m[i,j] not in (fixedColors|cic):
                        sourceColor = matrix.m[i,j]
                if matrix.m[i,j]==sourceColor:
                    m = matrix.m.copy()
                    colorCells=True
                    start = i
                if colorCells and matrix.m[i,j] in fixedColors and breakAtFixedColor:
                    break
                if colorCells and matrix.m[i,j] in cic:
                    if color==None:
                        m[i,j] = sourceColor
                    else:
                        m[i,j] = color
                if colorCells and ((matrix.m[i,j] in deleteExtensionColors) or \
                                   i==0 and deleteIfBorder):
                    currentI = i
                    for i in range(currentI, start):
                        m[i,j] = matrix.m[i,j]
                    break
            if color==None:
                sourceColor=None
            if colorCells:
                matrices.append(m)
    if direction=='all' or direction=='hv' or direction=='v' or direction=='d':
        for j in range(matrix.shape[1]):
            colorCells=False
            start = 0
            for i in range(matrix.shape[0]):
                if color==None:
                    if matrix.m[i,j] not in (fixedColors|cic):
                        sourceColor = matrix.m[i,j]
                if matrix.m[i,j]==sourceColor:
                    m = matrix.m.copy()
                    colorCells=True
                    start = i
                if colorCells and matrix.m[i,j] in fixedColors and breakAtFixedColor:
                    break
                if colorCells and matrix.m[i,j] in cic:
                    if color==None:
                        m[i,j] = sourceColor
                    else:
                        m[i,j] = color
                if colorCells and ((matrix.m[i,j] in deleteExtensionColors) or \
                                   i==m.shape[0]-1 and deleteIfBorder):
                    currentI = i+1
                    for i in reversed(range(start, currentI)):
                        m[i,j] = matrix.m[i,j]
                    break
            if color==None:
                sourceColor=None
            if colorCells:
                matrices.append(m)
             
    # Horizontal
    if direction=='all' or direction=='hv' or direction=='h' or direction=='l':
        for i in range(matrix.shape[0]):
            colorCells=False
            start = matrix.shape[1]-1
            for j in reversed(range(matrix.shape[1])):
                if color==None:
                    if matrix.m[i,j] not in (fixedColors|cic):
                        sourceColor = matrix.m[i,j]
                if matrix.m[i,j]==sourceColor:
                    m = matrix.m.copy()
                    colorCells=True
                    start = j
                if colorCells and matrix.m[i,j] in fixedColors and breakAtFixedColor:
                    break
                if colorCells and matrix.m[i,j] in cic:
                    if color==None:
                        m[i,j] = sourceColor
                    else:
                        m[i,j] = color
                if colorCells and ((matrix.m[i,j] in deleteExtensionColors) or \
                                   j==0 and deleteIfBorder):
                    currentJ = j
                    for j in range(currentJ, start):
                        m[i,j] = matrix.m[i,j]
                    break
            if color==None:
                sourceColor=None
            if colorCells:
                matrices.append(m)
    if direction=='all' or direction=='hv' or direction=='h' or direction=='r':
        for i in range(matrix.shape[0]):
            colorCells=False
            start = 0
            for j in range(matrix.shape[1]):
                if color==None:
                    if matrix.m[i,j] not in (fixedColors|cic):
                        sourceColor = matrix.m[i,j]
                if matrix.m[i,j]==sourceColor:
                    m = matrix.m.copy()
                    colorCells=True
                    start = j
                if colorCells and matrix.m[i,j] in fixedColors and breakAtFixedColor:
                    break
                if colorCells and matrix.m[i,j] in cic:
                    if color==None:
                        m[i,j] = sourceColor
                    else:
                        m[i,j] = color
                if colorCells and ((matrix.m[i,j] in deleteExtensionColors) or \
                                   j==m.shape[1]-1 and deleteIfBorder):
                    currentJ = j+1
                    for j in reversed(range(start, currentJ)):
                        m[i,j] = matrix.m[i,j]
                    break
            if color==None:
                sourceColor=None
            if colorCells:
                matrices.append(m)
              
    if len(matrices)>0:
        m = mergeMatrices(matrices, next(iter(cic)), mergeColor=mergeColor)
    else:
        m = matrix.m.copy()
                
    # Diagonal
    if direction=='all' or  direction=='diag' or direction=='d1' or direction=='d2':
        if direction=='diag' or direction=='all':
            directions = ['d1', 'd2']
        else:
            directions = [direction]
        for direction in directions:
            for transpose in [True, False]:
                if transpose and direction=='d1':
                    matrix.m = np.rot90(matrix.m, 2).T
                    m = np.rot90(m, 2).T
                if transpose and direction=='d2':
                    matrix.m = matrix.m.T
                    m = m.T
                if direction=='d2':
                    matrix.m = np.fliplr(matrix.m)
                    m = np.fliplr(m)
                for i in range(-matrix.shape[0]+1, matrix.shape[1]):
                    diag = np.diagonal(matrix.m, i)
                    colorCells=False
                    for j in range(len(diag)):
                        if color==None:
                            if i<=0:
                                if matrix.m[-i+j,j] not in (fixedColors|cic):
                                    sourceColor = matrix.m[-i+j,j]
                            else:
                                if matrix.m[j,i+j] not in (fixedColors|cic):
                                    sourceColor = matrix.m[j,i+j]
                        if i<=0:
                            if matrix.m[-i+j,j]==sourceColor:
                                colorCells=True
                            if colorCells and matrix.m[-i+j,j] in fixedColors and breakAtFixedColor:
                                break
                            if colorCells and matrix.m[-i+j,j] in cic:
                                if color==None:
                                    m[-i+j,j] = sourceColor
                                else:
                                    m[-i+j,j] = color
                            if colorCells and ((matrix.m[-i+j,j] in deleteExtensionColors) or \
                                               j==len(diag)-1 and deleteIfBorder):
                                for j in range(len(diag)):
                                    m[-i+j,j] = matrix.m[-i+j,j]
                                break
                        else:
                            if matrix.m[j,i+j]==sourceColor:
                                colorCells=True
                            if colorCells and matrix.m[j,i+j] in fixedColors and breakAtFixedColor:
                                break
                            if colorCells and matrix.m[j,i+j] in cic:
                                if color==None:
                                    m[j,i+j] = sourceColor
                                else:
                                    m[j,i+j] = color
                            if colorCells and ((matrix.m[j,i+j] in deleteExtensionColors) or \
                                               j==len(diag)-1 and deleteIfBorder):
                                for j in range(len(diag)):
                                    m[j,i+j] = matrix.m[j,i+j]
                                break
                    if color==None:
                        sourceColor=None
                if direction=='d2':
                    matrix.m = np.fliplr(matrix.m)
                    m = np.fliplr(m)
                if transpose and direction=='d2':
                    matrix.m = matrix.m.T
                    m = m.T
                if transpose and direction=='d1':
                    matrix.m = np.rot90(matrix.m, 2).T
                    m = np.rot90(m, 2).T

    return m    

def getBestExtendColor(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    
    mergeColors = t.commonOutColors-t.totalInColors
    if len(mergeColors) == 1:
        mergeColor = next(iter(mergeColors))
    else:
        mergeColor = None
    
    cic = t.commonChangedInColors
    if len(cic)==0:
        return bestFunction
    fixedColors = t.fixedColors
    for d in ['r', 'l', 'h', 'u', 'd', 'v', 'hv', 'd1', 'd2', 'diag', 'all']:
        for dib,bafc in product([True, False], [True, False]):
            f = partial(extendColor, direction=d, cic=cic, fixedColors=fixedColors,\
                        deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            f = partial(extendColor, direction=d, cic=cic, fixedColors=fixedColors,\
                        deleteExtensionColors=fixedColors,\
                        deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            for coc in t.commonChangedOutColors:    
                f = partial(extendColor, color=coc, direction=d, cic=cic, fixedColors=fixedColors,\
                            deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                f = partial(extendColor, color=coc, direction=d, cic=cic, fixedColors=fixedColors,\
                            deleteExtensionColors=fixedColors,\
                            deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                for fc in t.fixedColors:
                    f = partial(extendColor, color=coc, direction=d, cic=cic, sourceColor=fc, \
                                fixedColors=fixedColors,\
                                deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                    if bestScore==0:
                        return bestFunction
                    f = partial(extendColor, color=coc, direction=d, cic=cic, sourceColor=fc, \
                                fixedColors=fixedColors, deleteExtensionColors=fixedColors,\
                                deleteIfBorder=dib, breakAtFixedColor=bafc, mergeColor=mergeColor)
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                    if bestScore==0:
                        return bestFunction
                
    return bestFunction

# %% Fill rectangleInside (task 525)
    
def fillRectangleInside(matrix, rectangleColor, fillColor):
    m = matrix.m.copy()
    for shape in matrix.shapes:
        if shape.isRectangle and shape.color==rectangleColor:
            if shape.shape[0] > 2 and shape.shape[1] > 2:
                rect = np.full((shape.shape[0]-2, shape.shape[1]-2), fillColor, dtype=np.uint8)
                m[shape.position[0]+1:shape.position[0]+shape.shape[0]-1,\
                  shape.position[1]+1:shape.position[1]+shape.shape[1]-1] = rect
    return m
            

# %% Color longest line
def colorLongestLines(matrix, cic, coc, direction):
    """
    cic stands for "changedInColor"
    coc stands for "changedOutColor"
    direction can be one of 4 strings: 'v', 'h', 'hv', 'd' (vertical,
    horizontal, diagonal)
    It is assumed t.sameIOShapes
    """    
    m = matrix.m.copy()
    
    longest=0
    positions = set()
    if direction=='h':
        for i in range(m.shape[0]):
            count = 0
            for j in range(m.shape[1]):
                if m[i,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add((i,j))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add((i,m.shape[1]-1))
        for pos in positions:
            for j in range(pos[1]-longest, pos[1]):
                m[pos[0],j] = coc
        return m                
        
    elif direction=='v':
        for j in range(m.shape[1]):
            count = 0
            for i in range(m.shape[0]):
                if m[i,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add((i,j))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add((m.shape[0]-1,j))
        for pos in positions:
            for i in range(pos[0]-longest, pos[0]):
                m[i,pos[1]] = coc
        return m 
                        
    elif direction=='hv':
        longestH = 0
        longestV = 0
        positionsH = set()
        positionsV = set()
        for i in range(m.shape[0]):
            count = 0
            for j in range(m.shape[1]):
                if m[i,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longestH:
                        if count > longestH:
                            positionsH = set()
                        longestH = count
                        positionsH.add((i,j))
                    count = 0
            if count >= longestH:
                if count > longestH:
                    positionsH = set()
                longestH = count
                positionsH.add((i,m.shape[1]-1))
        for j in range(m.shape[1]):
            count = 0
            for i in range(m.shape[0]):
                if m[i,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longestV:
                        if count > longestV:
                            positionsV = set()
                        longestV = count
                        positionsV.add((i,j))
                    count = 0
            if count >= longestV:
                if count > longestV:
                    positionsV = set()
                longestV = count
                positionsV.add((m.shape[0]-1,j))
        for pos in positionsH:
            for j in range(pos[1]-longestH, pos[1]):
                m[pos[0],j] = coc
        for pos in positionsV:
            for i in range(pos[0]-longestV, pos[0]):
                m[i,pos[1]] = coc
        return m
    
    elif direction=='d':
        # Direction of main diagonal
        for i in reversed(range(m.shape[0])):
            count = 0
            jLimit = min(m.shape[1], m.shape[0]-i)
            for j in range(jLimit):
                if m[i+j,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add(((i+j-1,j-1), 'main'))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add(((i+jLimit-1,jLimit-1), 'main'))
        for j in range(1, m.shape[1]):
            count = 0
            iLimit = min(m.shape[0], m.shape[1]-j)
            for i in range(iLimit):
                if m[i,j+i]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add(((i-1,j+i-1), 'main'))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add(((iLimit-1,j+iLimit-1), 'main'))
                
        # Direction of counterdiagonal
        for i in range(m.shape[0]):
            count = 0
            jLimit = min(m.shape[1], i+1)
            for j in range(jLimit):
                if m[i-j,j]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add(((i-j+1, j-1), 'counter'))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add(((i-jLimit+1,jLimit-1), 'counter'))
        for j in range(m.shape[1]):
            count = 0
            iLimit = min(m.shape[0], m.shape[1]-j)
            for i in range(iLimit):
                if m[m.shape[0]-i-1,j+i]==cic:
                    if count!=0:
                        count += 1
                    else:
                        count = 1
                else:
                    if count >= longest:
                        if count > longest:
                            positions = set()
                        longest = count
                        positions.add(((m.shape[0]-i,j+i-1), 'counter'))
                    count = 0
            if count >= longest:
                if count > longest:
                    positions = set()
                longest = count
                positions.add(((m.shape[0]-iLimit,j+iLimit-1), 'counter'))
        
        # Draw the lines
        for pos in positions:
            if pos[1]=='main':
                for x in range(longest):
                    m[pos[0][0]-x, pos[0][1]-x] = coc
            else:
                for x in range(longest):
                    m[pos[0][0]+x, pos[0][1]-x] = coc
        return m
    return m
    
# %% Move shapes    

def moveShape(matrix, shape, background, direction, until = -1, nSteps = 100, \
              keepOriginal=False):
    """
    'direction' can be l, r, u, d, ul, ur, dl, dr
    (left, right, up, down, horizontal, vertical, diagonal1, diagonal2)
    'until' can be a color or -1, which will be interpreted as border
    If 'until'==-2, then move until the shape encounters anything
    """
    m = matrix.copy()
    if not keepOriginal:
        m = changeColorShapes(m, [shape], background)
    s = copy.deepcopy(shape)
    if nSteps=="shapeX":
        nSteps = shape.shape[0]
    if nSteps=="shapeY":
        nSteps = shape.shape[1]
    step = 0
    while True and step != nSteps:
        step += 1
        for c in s.pixels:
            pos = (s.position[0]+c[0], s.position[1]+c[1])
            if direction == "l":
                newPos = (pos[0], pos[1]-1)
            if direction == "r":
                newPos = (pos[0], pos[1]+1)
            if direction == "u":
                newPos = (pos[0]-1, pos[1])
            if direction == "d":
                newPos = (pos[0]+1, pos[1])
            if direction == "ul":
                newPos = (pos[0]-1, pos[1]-1)
            if direction == "ur":
                newPos = (pos[0]-1, pos[1]+1)
            if direction == "dl":
                newPos = (pos[0]+1, pos[1]-1)
            if direction == "dr":
                newPos = (pos[0]+1, pos[1]+1)
                
            if newPos[0] not in range(m.shape[0]) or \
            newPos[1] not in range(m.shape[1]):
                if until != -1 and until != -2:
                    return matrix.copy()
                else:
                    return insertShape(m, s)
            if until == -2 and m[newPos] != background:
                return insertShape(m, s)
            if m[newPos] == until:
                return insertShape(m, s)
            
        if direction == "l":
            s.position = (s.position[0], s.position[1]-1)
        if direction == "r":
            s.position = (s.position[0], s.position[1]+1)
        if direction == "u":
            s.position = (s.position[0]-1, s.position[1])
        if direction == "d":
            s.position = (s.position[0]+1, s.position[1])
        if direction == "ul":
            s.position = (s.position[0]-1, s.position[1]-1)
        if direction == "ur":
            s.position = (s.position[0]-1, s.position[1]+1)
        if direction == "dl":
            s.position = (s.position[0]+1, s.position[1]-1)
        if direction == "dr":
            s.position = (s.position[0]+1, s.position[1]+1)
      
    return insertShape(m, s) 
    
def moveAllShapes(matrix, background, direction, until, nSteps=100, color=None, \
                  keepOriginal=False):
    """
    direction can be l, r, u, d, ul, ur, dl, dr, h, v, d1, d2, all, any
    """
    if color==None or color=="multiColor":
        shapesToMove = matrix.multicolorShapes
    elif color=="diagonalMultiColor":
        shapesToMove=matrix.multicolorDShapes
    elif color=="singleColor":
        shapesToMove = [s for s in matrix.shapes if s.color!=background]
    elif color=="diagonalSingleColor":
        shapesToMove = [s for s in matrix.dShapes if s.color!=background]
    else:
        shapesToMove = [s for s in matrix.shapes if s.color in color]
    if direction == 'l':
        shapesToMove.sort(key=lambda x: x.position[1])
    if direction == 'r':
        shapesToMove.sort(key=lambda x: x.position[1]+x.shape[1], reverse=True)
    if direction == 'u':
        shapesToMove.sort(key=lambda x: x.position[0])  
    if direction == 'd':
        shapesToMove.sort(key=lambda x: x.position[0]+x.shape[0], reverse=True)
    m = matrix.m.copy()
    if len(shapesToMove) > 15:
        return m
    for s in shapesToMove:
        newMatrix = m.copy()
        if direction == "any":
            for d in ['l', 'r', 'u', 'd', 'ul', 'ur', 'dl', 'dr']:
                newMatrix = moveShape(m, s, background, d, until)
                if not np.all(newMatrix == m):
                    return newMatrix
                    break
        else:
            m = moveShape(m, s, background, direction, until, nSteps, keepOriginal)
    return m
    
def moveShapeToClosest(matrix, shape, background, until=None, diagonals=False, restore=True):
    """
    Given a matrix (numpy.ndarray) and a Task.Shape, this function moves the
    given shape until the closest shape with the color given by "until".
    """
    m = matrix.copy()
    s = copy.deepcopy(shape)
    m = deleteShape(m, shape, background)
    if until==None:
        if hasattr(shape, "color"):
            until=shape.color
        else:
            return matrix
    if until not in m:
        return matrix
    nSteps = 0
    while True:
        for c in s.pixels:
            pixelPos = tuple(map(operator.add, c, s.position))
            if nSteps <= pixelPos[0] and m[pixelPos[0]-nSteps, pixelPos[1]] == until:
                while nSteps>=0 and m[pixelPos[0]-nSteps, pixelPos[1]]!=background:
                    nSteps-=1
                s.position = (s.position[0]-nSteps, s.position[1])
                return insertShape(m, s)
            if pixelPos[0]+nSteps < m.shape[0] and m[pixelPos[0]+nSteps, pixelPos[1]] == until:
                while nSteps>=0 and m[pixelPos[0]+nSteps, pixelPos[1]]!=background:
                    nSteps-=1
                s.position = (s.position[0]+nSteps, s.position[1])
                return insertShape(m, s)
            if nSteps <= pixelPos[1] and m[pixelPos[0], pixelPos[1]-nSteps] == until:
                while nSteps>=0 and m[pixelPos[0], pixelPos[1]-nSteps]!=background:
                    nSteps-=1
                s.position = (s.position[0], s.position[1]-nSteps)
                return insertShape(m, s)
            if pixelPos[1]+nSteps < m.shape[1] and m[pixelPos[0], pixelPos[1]+nSteps] == until:
                while nSteps>=0 and m[pixelPos[0], pixelPos[1]+nSteps]!=background:
                    nSteps-=1
                s.position = (s.position[0], s.position[1]+nSteps)
                return insertShape(m, s)
            if diagonals:
                if nSteps <= pixelPos[0] and nSteps <= pixelPos[1] and \
                m[pixelPos[0]-nSteps, pixelPos[1]-nSteps] == until:
                    s.position = (s.position[0]-nSteps+1, s.position[1]-nSteps+1)
                    return insertShape(m, s)
                if nSteps <= pixelPos[0] and pixelPos[1]+nSteps < m.shape[1] and \
                m[pixelPos[0]-nSteps, pixelPos[1]+nSteps] == until:
                    s.position = (s.position[0]-nSteps+1, s.position[1]+nSteps-1)
                    return insertShape(m, s)
                if pixelPos[0]+nSteps < m.shape[0] and nSteps <= pixelPos[1] and \
                m[pixelPos[0]+nSteps, pixelPos[1]-nSteps] == until:
                    s.position = (s.position[0]+nSteps-1, s.position[1]-nSteps+1)
                    return insertShape(m, s)
                if pixelPos[0]+nSteps < m.shape[0] and pixelPos[1]+nSteps < m.shape[1] and \
                m[pixelPos[0]+nSteps, pixelPos[1]+nSteps] == until:
                    s.position = (s.position[0]+nSteps-1, s.position[1]+nSteps-1)
                    return insertShape(m, s)
        nSteps += 1
        if nSteps > m.shape[0] and nSteps > m.shape[1]:
            if restore:
                return matrix
            else:
                return m
        
def moveAllShapesToClosest(matrix, background, colorsToMove=None, until=None, \
                           diagonals=False, restore=True, fixedShapeFeatures=None):
    """
    This function moves all the shapes with color "colorsToMove" until the
    closest shape with color "until".
    """
    m = matrix.m.copy()
    if len(matrix.shapes) > 25:
        return m
    fixedShapes = []
    if until == None:
        colorsToMove = []
        for shape in matrix.shapes:
            if hasFeatures(shape.boolFeatures, fixedShapeFeatures):
                fixedShapes.append(shape)
                colorsToMove.append(shape.color)
    elif colorsToMove==None:
        colorsToMove = matrix.colors - set([background, until])
    else:
        colorsToMove = [colorsToMove]
    for ctm in colorsToMove:
        for shape in matrix.shapes:
            if shape not in fixedShapes:
                if shape.color == ctm:
                    m = moveShapeToClosest(m, shape, background, until, diagonals, restore)
    return m

def getBestMoveShapes(t, candidate):
    """
    This functions tries to find, for a given task t, the best way to move
    shapes.
    """
    directions = ['l', 'r', 'u', 'd', 'ul', 'ur', 'dl', 'dr', 'any']
    bestScore = 1000
    bestFunction = partial(identityM)
    
    doSingleColor = all([len(s.inMatrix.shapes) < 30 for s in t.trainSamples])
    doDSingleColor = all([len(s.inMatrix.dShapes) < 15 for s in t.trainSamples])
    doMulticolor = all([len(s.inMatrix.multicolorShapes) < 15 for s in t.trainSamples])
        
    # Move all shapes in a specific direction, until a non-background thing is touched
    for d in directions:
        if doSingleColor:
            f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                        direction=d, color="singleColor")
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
        if doDSingleColor:
            f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                        direction=d, color="diagonalSingleColor")
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
        if doMulticolor:
            f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                        direction=d, color="multiColor")
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
        f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                    direction=d, color="diagonalMultiColor")
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                    direction=d, color="diagonalMultiColor", nSteps="shapeX")
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                    direction=d, color="diagonalMultiColor", nSteps="shapeY")
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        if doDSingleColor:
            f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                        direction=d, color="diagonalSingleColor", nSteps="shapeX")
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
            f = partial(moveAllShapes, background=t.backgroundColor, until=-2,\
                        direction=d, color="diagonalSingleColor", nSteps="shapeY")
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
        
    if doSingleColor:
        colorsToChange = list(t.colors - t.fixedColors - set({t.backgroundColor}))
        ctc = [[c] for c in colorsToChange]# + [colorsToChange] # Also all colors
        for c in ctc:
            for d in directions:
                moveUntil = colorsToChange + [-1] + [-2] #Border, any
                for u in moveUntil:
                    f = partial(moveAllShapes, color=c, background=t.backgroundColor,\
                                    direction=d, until=u)
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                    if bestScore==0:
                        return bestFunction
                for nSteps in range(1, 5):
                    f = partial(moveAllShapes, color=c, background=t.backgroundColor,\
                                direction=d, until=-2, nSteps=nSteps)
                    bestFunction, bestScore, isPerfect = updateBestFunction(t, f, bestScore, bestFunction,\
                                                                            checkPerfect=True, prevScore = candidate.score)
                    if isPerfect:
                        return bestFunction
                    #print(c, d, nSteps, bestScore)
                    if bestScore==0:
                        return bestFunction
                    f = partial(moveAllShapes, color=c, background=t.backgroundColor,\
                                direction=d, until=-2, nSteps=nSteps, keepOriginal=True)
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                    if bestScore==0:
                        return bestFunction
                
    
        if t.backgroundColor != -1 and hasattr(t, 'fixedColors'):
            colorsToMove = set(range(10)) - set([t.backgroundColor]) - t.fixedColors
            for ctm in colorsToMove:
                for uc in t.unchangedColors:
                    f = partial(moveAllShapesToClosest, colorsToMove=ctm,\
                                     background=t.backgroundColor, until=uc)
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                    if bestScore==0:
                        return bestFunction
                    
                    f = partial(moveAllShapesToClosest, colorsToMove=ctm,\
                                background=t.backgroundColor, until=uc, diagonals=True)
                    bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                    if bestScore==0:
                        return bestFunction
                
        if all([len(sample.fixedShapes)>0 for sample in t.trainSamples]):
            f = partial(moveAllShapesToClosest, background=t.backgroundColor,\
                        fixedShapeFeatures = t.fixedShapeFeatures)
            bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
            if bestScore==0:
                return bestFunction
        
    return bestFunction

def getBestMoveShapesNoRestore(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    
    if t.backgroundColor != -1 and hasattr(t, 'fixedColors'):
        colorsToMove = set(range(10)) - set([t.backgroundColor]) - t.fixedColors
        for ctm in colorsToMove:
            for uc in t.unchangedColors:
                f = partial(moveAllShapesToClosest, colorsToMove=ctm,\
                                 background=t.backgroundColor, until=uc, restore=False)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
                
                f = partial(moveAllShapesToClosest, colorsToMove=ctm,\
                            background=t.backgroundColor, until=uc, diagonals=True, restore=False)
                bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
                if bestScore==0:
                    return bestFunction
    
    if all([len(sample.fixedShapes)>0 for sample in t.trainSamples]):        
        f = partial(moveAllShapesToClosest, background=t.backgroundColor,\
                    fixedShapeFeatures = t.fixedShapeFeatures, restore=False)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)  
        if bestScore==0:
            return bestFunction    
        
    return bestFunction
    

# %% Complete rectangles
def completeRectangles(matrix, sourceColor, newColor):
    """
    It is assumed that the background is clear.
    """
    m = matrix.m.copy()
    for s in matrix.multicolorDShapes:
        if hasattr(s, 'color') and s.color==sourceColor:
            newShape = copy.deepcopy(s)
            newShape.m[newShape.m==255] = newColor
            m = insertShape(m, newShape)
    return m

# %% Delete shapes
# Like that this only solves task 96. It's clearly not general enough.
def deletePixels(matrix, diagonals=False):
    """
    Given a matrix, this functions deletes all the pixels. This means that all
    the dShapes consisting of only 1 pixel are converted to the color that
    surrounds most of that pixel.
    """
    m = matrix.m.copy()
    if m.shape[0]==1 and m.shape[1]==1:
        return m
    if diagonals:
        shapes = matrix.dShapes
    else:
        shapes = matrix.shapes
    for s in shapes:
        if s.nPixels==1:
            surrColors = Counter()
            if s.position[0]>0:
                surrColors[m[s.position[0]-1, s.position[1]]] += 1
            if s.position[1]<m.shape[1]-1:
                surrColors[m[s.position[0], s.position[1]+1]] += 1
            if s.position[0]<m.shape[0]-1:
                surrColors[m[s.position[0]+1, s.position[1]]] += 1
            if s.position[1]>0:
                surrColors[m[s.position[0], s.position[1]-1]] += 1
            if len(set(surrColors.values()))==1:
                if s.position[0]>0 and s.position[1]>0:
                    surrColors[m[s.position[0]-1, s.position[1]-1]] += 1
                if s.position[0]>0 and s.position[1]<m.shape[1]-1:
                    surrColors[m[s.position[0]-1, s.position[1]+1]] += 1
                if s.position[0]<m.shape[0]-1 and s.position[1]<m.shape[1]-1:
                    surrColors[m[s.position[0]+1, s.position[1]+1]] += 1
                if s.position[0]<m.shape[0]-1 and s.position[1]>0:
                    surrColors[m[s.position[0]+1, s.position[1]-1]] += 1
            
            m[s.position[0],s.position[1]] = max(surrColors.items(), key=operator.itemgetter(1))[0]
    return m

# %% Connect Pixels

def connectPixels(matrix, pixelColor=None, connColor=None, fixedColors=set(),\
                  allowedChanges={}, lineExclusive=False, diagonal=False):
    """
    Given a matrix, this function connects all the pixels that have the same
    color. This means that, for example, all the pixels between two red pixels
    will also be red.
    If "pixelColor" is specified, then only the pixels with the specified color
    will be connected.
    Ìf "connColor" is specified, the color used to connect the pixels will be
    the one given by this parameter.
    If there are any colors in the "fixedColors" set, then it is made sure that
    they remain unchanged.
    "allowedChanges" is a dictionary determining which color changes are
    allowed. It's exclusive with the options unchangedColors and connColor.
    """
    m = matrix.copy()
    # Row
    for i in range(m.shape[0]):
        lowLimit = 0
        while lowLimit < m.shape[1] and matrix[i, lowLimit] != pixelColor:
            lowLimit += 1
        lowLimit += 1
        upLimit = m.shape[1]-1
        while upLimit > lowLimit and matrix[i, upLimit] != pixelColor:
            upLimit -= 1
        if upLimit > lowLimit:
            if lineExclusive:
                for j in range(lowLimit, upLimit):
                    if matrix[i,j] == pixelColor:
                        lowLimit = upLimit
                        break
            for j in range(lowLimit, upLimit):
                if connColor != None:
                    if matrix[i,j] != pixelColor and matrix[i,j] not in fixedColors:
                        m[i,j] = connColor
                else:
                    if matrix[i,j] in allowedChanges.keys():
                        m[i,j] = allowedChanges[matrix[i,j]]
       
    # Column             
    for j in range(m.shape[1]):
        lowLimit = 0
        while lowLimit < m.shape[0] and matrix[lowLimit, j] != pixelColor:
            lowLimit += 1
        lowLimit += 1
        upLimit = m.shape[0]-1
        while upLimit > lowLimit and matrix[upLimit, j] != pixelColor:
            upLimit -= 1
        if upLimit > lowLimit:
            if lineExclusive:
                for i in range(lowLimit, upLimit):
                    if matrix[i,j] == pixelColor:
                        lowLimit = upLimit
                        break
            for i in range(lowLimit, upLimit):
                if connColor != None:
                    if matrix[i,j] != pixelColor and matrix[i,j] not in fixedColors:
                        m[i,j] = connColor
                else:
                    if matrix[i,j] in allowedChanges.keys():
                        m[i,j] = allowedChanges[matrix[i,j]]
    
    # Diagonal         
    if diagonal:
        for d in [1, 2]:
            if d==2:
                matrix = np.fliplr(matrix)
                m = np.fliplr(m)
            for i in range(-matrix.shape[0]+1, matrix.shape[1]):
                diag = np.diagonal(matrix, i)
                lowLimit = 0
                while lowLimit < len(diag) and diag[lowLimit] != pixelColor:
                    lowLimit += 1
                lowLimit += 1
                upLimit = len(diag)-1
                while upLimit > lowLimit and diag[upLimit] != pixelColor:
                    upLimit -= 1
                if upLimit > lowLimit:
                    if lineExclusive:
                        for j in range(lowLimit, upLimit):
                            if i<=0:
                                if matrix[-i+j,j] == pixelColor:
                                    lowLimit = upLimit
                                    break
                            else:
                                if matrix[j,i+j] == pixelColor:
                                    lowLimit = upLimit
                                    break
                    for j in range(lowLimit, upLimit):
                        if i<=0:
                            if connColor != None:
                                if matrix[-i+j,j] != pixelColor and matrix[-i+j,j] not in fixedColors:
                                    m[-i+j,j] = connColor
                            else:
                                if matrix[-i+j,j] in allowedChanges.keys():
                                    m[-i+j,j] = allowedChanges[matrix[-i+j,j]]
                        else:
                            if connColor != None:
                                if matrix[j,i+j] != pixelColor and matrix[j,i+j] not in fixedColors:
                                    m[j,i+j] = connColor
                            else:
                                if matrix[j,i+j] in allowedChanges.keys():
                                    m[j,i+j] = allowedChanges[matrix[j,i+j]]
            if d==2:
                matrix = np.fliplr(matrix)
                m = np.fliplr(m)

    return m

def connectAnyPixels(matrix, pixelColor=None, connColor=None, fixedColors=set(),\
                     allowedChanges={}, lineExclusive=False, diagonal=False):
    m = matrix.m.copy()
    if pixelColor==None:
        if connColor==None:
            for c in matrix.colors - set([matrix.backgroundColor]):
                m = connectPixels(m, c, c, lineExclusive=lineExclusive, diagonal=diagonal)
            return m
        else:
            for c in matrix.colors - set([matrix.backgroundColor]):
                m = connectPixels(m, c, connColor, lineExclusive=lineExclusive, diagonal=diagonal)
            return m
    else:
        if len(allowedChanges)>0:
            m = connectPixels(m, pixelColor, allowedChanges=allowedChanges,\
                              lineExclusive=lineExclusive, diagonal=diagonal)
        else:
            m = connectPixels(m, pixelColor, connColor, fixedColors, lineExclusive=lineExclusive,\
                              diagonal=diagonal)
    return m

def rotate(matrix, angle):
    """
    Angle can be 90, 180, 270
    """
    assert angle in [90, 180, 270], "Invalid rotation angle"
    if isinstance(matrix, np.ndarray):
        m = matrix.copy()
    else:
        m = matrix.m.copy()
    return np.rot90(m, int(angle/90))    
    
def mirror(matrix, axis):
    """
    Axis can be lr, up, d1, d2
    """
    if isinstance(matrix, np.ndarray):
        m = matrix.copy()
    else:
        m = matrix.m.copy()
    assert axis in ["lr", "ud", "d1", "d2"], "Invalid mirror axis"
    if axis == "lr":
        return np.fliplr(m)
    if axis == "ud":
        return np.flipud(m)
    if axis == "d1":
        return m.T
    if axis == "d2":
        return m[::-1,::-1].T

def flipShape(matrix, shape, axis, background):
    # Axis can be lr, ud
    m = matrix.copy()
    smallM = np.ones((shape.shape[0], shape.shape[1]), dtype=np.uint8) * background
    for c in shape.pixels:
        smallM[c] = shape.color
    if axis == "lr":
        smallM = np.fliplr(smallM)
    if axis == "ud":
        smallM = np.flipud(smallM)
    for i,j in np.ndindex(smallM.shape):
        m[shape.position[0]+i, shape.position[1]+j] = smallM[i,j]
    return m

def flipAllShapes(matrix, axis, color, background, byColor=False, diagonal=False):#, multicolor=False):
    m = matrix.m.copy()
    if byColor:
        shapesToMirror = [s for s in matrix.shapesByColor if s.color in color]
    #elif multicolor:
    #    if diagonal:
    #        shapesToMirror = [s for s in matrix.multicolorDShapes]
    #    else:
    #        shapesToMirror = [s for s in matrix.multicolorShapes]
    else:
        if diagonal:
            shapesToMirror = [s for s in matrix.dShapes if s.color in color]
        else:
            shapesToMirror = [s for s in matrix.shapes if s.color in color]
    for s in shapesToMirror:
        m = flipShape(m, s, axis, background)
    return m

def getBestFlipAllShapes(t):
    bestFunction = partial(identityM)
    bestScore = 1000
    for d in ["lr", "ud"]:
        #for multicolor in [True, False]:
        for diagonal in [True, False]:
            bestFunction, bestScore = updateBestFunction(t, partial(flipAllShapes, axis=d, color=[1,2,3,4,5,6,7,8,9],\
                                    background=max(t.backgroundColor,0), diagonal=diagonal), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(flipAllShapes, axis=d, color=[1,2,3,4,5,6,7,8,9],\
                                background=max(t.backgroundColor,0), byColor=True), bestScore, bestFunction)
    return bestFunction

def mapPixels(matrix, pixelMap, outShape):
    """
    Given a Task.Matrix as input, this function maps each pixel of that matrix
    to an outputMatrix, given by outShape.
    The dictionary pixelMap determines which pixel in the input matrix maps
    to each pixel in the output matrix.
    """
    inMatrix = matrix.m.copy()
    m = np.zeros(outShape, dtype=np.uint8)
    for i,j in np.ndindex(outShape):
        m[i,j] = inMatrix[pixelMap[i,j]]
    return m

def switchColors(matrix, color1=None, color2=None):
    """
    This function switches the color1 and the color2 in the matrix.
    If color1 and color2 are not specified, then the matrix is expected to only
    have 2 colors, and they will be switched.
    """
    if type(matrix) == np.ndarray:
        m = matrix.copy()
    else:
        m = matrix.m.copy()
    if color1==None or color2==None:
        color1 = m[0,0]
        for i,j in np.ndindex(m.shape):
            if m[i,j]!=color1:
                color2 = m[i,j]
                break
    for i,j in np.ndindex(m.shape):
        if m[i,j]==color1:
            m[i,j] = color2
        else:
            m[i,j] = color1        
    return m

# %% Rotation things
# TODO (task 26)
def makeShapeRotationInvariant(matrix, color):
    m = matrix.m.copy()
    
    return m

# %% Follow row/col patterns
def identifyColor(m, pixelPos, c2c, rowStep=None, colStep=None):
    """
    Utility function for followPattern.
    """
    if colStep!=None and rowStep!=None:
        i = 0
        while i+pixelPos[0] < m.shape[0]:
            j = 0
            while j+pixelPos[1] < m.shape[1]:
                if m[pixelPos[0]+i, pixelPos[1]+j] != c2c:
                    return m[pixelPos[0]+i, pixelPos[1]+j]
                j += colStep
            i += rowStep
        return c2c
    
def identifyColStep(m, c2c):
    """
    Utility function for followPattern.
    """
    colStep = 1
    while colStep < int(m.shape[1]/2)+1:
        isGood = True
        for j in range(colStep):
            for i in range(m.shape[0]):
                block = 0
                colors = set()
                while j+block < m.shape[1]:
                    colors.add(m[i,j+block])
                    block += colStep
                if c2c in colors:
                    if len(colors) > 2:
                        isGood = False
                        break
                else:
                    if len(colors) > 1:
                        isGood = False
                        break
            if not isGood:
                break  
        if isGood:
            return colStep 
        colStep+=1 
    return m.shape[1]

def identifyRowStep(m, c2c):
    """
    Utility function for followPattern.
    """
    rowStep = 1
    while rowStep < int(m.shape[0]/2)+1:
        isGood = True
        for i in range(rowStep):
            for j in range(m.shape[1]):
                block = 0
                colors = set()
                while i+block < m.shape[0]:
                    colors.add(m[i+block,j])
                    block += rowStep
                if c2c in colors:
                    if len(colors) > 2:
                        isGood = False
                        break
                else:
                    if len(colors) > 1:
                        isGood = False
                        break
            if not isGood:
                break  
        if isGood:
            return rowStep 
        rowStep+=1  
    return m.shape[0]            

def followPattern(matrix, rc, colorToChange=None, rowStep=None, colStep=None):
    """
    Given a Task.Matrix, this function turns it into a matrix that follows a
    pattern. This will be made row-wise, column-wise or both, depending on the
    parameter "rc". "rc" can be "row", "column" or "both".
    'colorToChange' is the number corresponding to the only color that changes,
    if any.
    'rowStep' and 'colStep' are only to be given if the rowStep/colStep is the
    same for every train sample.
    """  
    m = matrix.m.copy()
            
    if colorToChange!=None:
        if rc=="col":
            rowStep=m.shape[0]
            if colStep==None:
                colStep=identifyColStep(m, colorToChange)
        if rc=="row":
            colStep=m.shape[1]
            if rowStep==None:
                rowStep=identifyRowStep(m, colorToChange)
        if rc=="both":
            if colStep==None and rowStep==None:
                colStep=identifyColStep(m, colorToChange)
                rowStep=identifyRowStep(m, colorToChange) 
            elif rowStep==None:
                rowStep=m.shape[0]
            elif colStep==None:
                colStep=m.shape[1]                       
        for i,j in np.ndindex((rowStep, colStep)):
            color = identifyColor(m, (i,j), colorToChange, rowStep, colStep)
            k = 0
            while i+k < m.shape[0]:
                l = 0
                while j+l < m.shape[1]:
                    m[i+k, j+l] = color
                    l += colStep
                k += rowStep
            
    return m

# %% Fill the blank
def fillTheBlankParameters(t):
    matrices = []
    for s in t.trainSamples:
        m = s.inMatrix.m.copy()
        blank = s.blankToFill
        m[blank.position[0]:blank.position[0]+blank.shape[0],\
          blank.position[1]:blank.position[1]+blank.shape[1]] = s.outMatrix.m.copy()
        matrices.append(Task.Matrix(m))
        
    x = []
    x.append(all([m.lrSymmetric for m in matrices]))
    x.append(all([m.udSymmetric for m in matrices]))
    x.append(all([m.d1Symmetric for m in matrices]))
    x.append(all([m.d2Symmetric for m in matrices]))
    return x

def fillTheBlank(matrix, params):
    m = matrix.m.copy()
    if len(matrix.blanks) == 0:
        return m
    blank = matrix.blanks[0]
    color = blank.color
    pred = np.zeros(blank.shape, dtype=np.uint8)
    
    # lr
    if params[0]:
        for i,j in np.ndindex(blank.shape):
            if m[blank.position[0]+i, m.shape[1]-1-(blank.position[1]+j)] != color:
                pred[i,j] = m[blank.position[0]+i, m.shape[1]-1-(blank.position[1]+j)]
    # ud
    if params[1]:
        for i,j in np.ndindex(blank.shape):
            if m[m.shape[0]-1-(blank.position[0]+i), blank.position[1]+j] != color:
                pred[i,j] = m[m.shape[0]-1-(blank.position[0]+i), blank.position[1]+j]
    # d1
    if params[2] and m.shape[0]==m.shape[1]:
        for i,j in np.ndindex(blank.shape):
            if m[blank.position[1]+j, blank.position[0]+i] != color:
                pred[i,j] = m[blank.position[1]+j, blank.position[0]+i]
    # d2 (persymmetric matrix)
    if params[3] and m.shape[0]==m.shape[1]:
        for i,j in np.ndindex(blank.shape):
            if m[m.shape[1]-1-(blank.position[1]+j), m.shape[0]-1-(blank.position[0]+i)] != color:
                pred[i,j] = m[m.shape[1]-1-(blank.position[1]+j), m.shape[0]-1-(blank.position[0]+i)]
    
    return pred
    
# %% Operations with more than one matrix

# All the matrices need to have the same shape

def pixelwiseAnd(matrices, falseColor, targetColor=None, trueColor=None):
    """
    This function returns the result of executing the pixelwise "and" operation
    in a list of matrices.
    
    Parameters
    ----------
    matrices: list
        A list of numpy.ndarrays of the same shape
    falseColor: int
        The color of the pixel in the output matrix if the "and" operation is
        false.
    targetColor: int
        The color to be targeted by the "and" operation. For example, if
        targetColor is red, then the "and" operation will be true for a pixel
        if all that pixel is red in all of the input matrices.
        If targetColor is None, then the "and" operation will return true if
        the pixel has the same color in all the matrices, and false otherwise.
    trueColor: int
        The color of the pixel in the output matrix if the "and" operation is
        true.
        If trueColor is none, the output color if the "and" operation is true
        will be the color of the evaluated pixel.
    """
    m = np.zeros(matrices[0].shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        if targetColor == None:
            if all([x[i,j] == matrices[0][i,j] for x in matrices]):
                if trueColor == None:
                    m[i,j] = matrices[0][i,j]
                else:
                    m[i,j] = trueColor
            else:
                m[i,j] = falseColor
        else:
            if all([x[i,j] == targetColor for x in matrices]):
                if trueColor == None:
                    m[i,j] = matrices[0][i,j]
                else:
                    m[i,j] = trueColor
            else:
                m[i,j] = falseColor
    return m

"""
def pixelwiseOr(matrices, falseColor, targetColor=None, trueColor=None, \
                trueValues=None):
    See pixelwiseAnd.
    trueValues is a list with as many elements as matrices.
    m = np.zeros(matrices[0].shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        if targetColor == None:
            isFalse = True
            for x in matrices:
                if x[i,j] != falseColor:
                    isFalse = False
                    if trueColor == None:
                        m[i,j] = x[i,j]
                    else:
                        m[i,j] = trueColor
                    break
            if isFalse:
                m[i,j] = falseColor
        else:
            if any([x[i,j] == targetColor for x in matrices]):
                if trueColor == None:
                    m[i,j] = targetColor
                else:
                    m[i,j] = trueColor
            else:
                m[i,j] = falseColor
    return m
"""

def pixelwiseOr(matrices, falseColor, targetColor=None, trueColor=None, \
                trueValues=None):
    """
    See pixelwiseAnd.
    trueValues is a list with as many elements as matrices.
    """
    m = np.zeros(matrices[0].shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        if targetColor == None:
            trueCount = 0
            index = 0
            for x in matrices:
                if x[i,j] != falseColor:
                    trueCount += 1
                    trueIndex = index
                index += 1
            if trueCount==0:
                m[i,j] = falseColor
            else:
                if trueColor!=None:
                    m[i,j] = trueColor
                elif trueValues!=None:
                    if trueCount==1:
                        m[i,j] = trueValues[trueIndex]
                    else:
                        m[i,j] = matrices[trueIndex][i,j]
                else:
                    m[i,j] = matrices[trueIndex][i,j]
        else:
            if any([x[i,j] == targetColor for x in matrices]):
                if trueColor == None:
                    m[i,j] = targetColor
                else:
                    m[i,j] = trueColor
            else:
                m[i,j] = falseColor
    return m

def pixelwiseXor(m1, m2, falseColor, targetColor=None, trueColor=None, \
                 firstTrue=None, secondTrue=None):
    """
    See pixelwiseAnd. The difference is that the Xor operation only makes sense
    with two input matrices.
    """
    m = np.zeros(m1.shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        if targetColor == None:
            if (m1[i,j] == falseColor) != (m2[i,j] == falseColor):
                if trueColor == None:
                    if firstTrue == None:
                        if m1[i,j] != falseColor:
                            m[i,j] = m1[i,j]
                        else:
                            m[i,j] = m2[i,j]
                    else:
                        if m1[i,j] != falseColor:
                            m[i,j] = firstTrue
                        else:
                            m[i,j] = secondTrue
                else:
                    m[i,j] = trueColor     
            else:
                m[i,j] = falseColor
        else:
            if (m1[i,j] == targetColor) != (m2[i,j] == targetColor):
                if trueColor == None:
                    if firstTrue == None:
                        if m1[i,j] != falseColor:
                            m[i,j] = m1[i,j]
                        else:
                            m[i,j] = m2[i,j]
                    else:
                        if m1[i,j] != falseColor:
                            m[i,j] = firstTrue
                        else:
                            m[i,j] = secondTrue
                else:
                    m[i,j] = trueColor     
            else:
                m[i,j] = falseColor
    return m

def colorSubmatricesWithReference(matrix, reference, firstIsRef=True):
    """
    The shape of matrix has to be a multiple of the shape of reference.
    """
    m = matrix.m.copy()
    factor = (int(m.shape[0]/reference.shape[0]), int(m.shape[0]/reference.shape[0]))
    for i,j in np.ndindex(reference.shape):
        for k,l in np.ndindex(factor):
            m[i*factor[0]+k, j*factor[1]+l] = reference[i,j]
    return m

# %% Downsize and Minimize
    
def getDownsizeFactors(matrix):
    """
    Still unused
    """
    xDivisors = set()
    for x in range(1, matrix.shape[0]):
        if (matrix.shape[0]%x)==0:
            xDivisors.add(x)
    yDivisors = set()
    for y in range(1, matrix.shape[1]):
        if (matrix.shape[1]%y)==0:
            yDivisors.add(y)
    
    downsizeFactors = set()
    for x,y in product(xDivisors, yDivisors):
        downsizeFactors.add((x,y))
 
    return downsizeFactors

def downsize(matrix, newShape, falseColor=None):
    """
    Given a matrix and a shape, this function returns a new matrix with the
    given shape. The elements of the return matrix are given by the colors of 
    each of the submatrices. Each submatrix is only allowed to have the
    background color and at most another one (that will define the output
    color of the corresponding pixel).
    """
    if falseColor==None:
        falseColor = matrix.backgroundColor
    if (matrix.shape[0]%newShape[0])!=0 or (matrix.shape[1]%newShape[1])!=0:
        return matrix.m.copy()
    xBlock = int(matrix.shape[0]/newShape[0])
    yBlock = int(matrix.shape[1]/newShape[1])
    m = np.full(newShape, matrix.backgroundColor, dtype=np.uint8)
    for i,j in np.ndindex(newShape[0], newShape[1]):
        color = -1
        for x,y in np.ndindex(xBlock, yBlock):
            if matrix.m[i*xBlock+x, j*yBlock+y] not in [matrix.backgroundColor, color]:
                if color==-1:
                    color = matrix.m[i*xBlock+x, j*yBlock+y]
                else:
                    return matrix.m.copy()
        if color==-1:
            m[i,j] = falseColor
        else:
            m[i,j] = color
    return m

def minimize(matrix):
    """
    Given a matrix, this function returns the matrix resulting from the
    following operations:
        If two consecutive rows are equal, delete one of them
        If two consecutive columns are equal, delete one of them
    """
    m = matrix.m.copy()
    x = 1
    for i in range(1, matrix.shape[0]):
        if np.array_equal(m[x,:],m[x-1,:]):
            m = np.delete(m, (x), axis=0)
        else:
            x+=1
    x = 1
    for i in range(1, matrix.shape[1]):
        if np.array_equal(m[:,x],m[:,x-1]):
            m = np.delete(m, (x), axis=1)
        else:
            x+=1
    return m
            
        

# %% Operations to extend matrices
    
def extendMatrix(matrix, color, position="tl", xShape=None, yShape=None, isSquare=False, goodDimension=None):
    """
    Given a matrix, xShape(>matrix.shape[0]), yShape(>matrix.shape[1]) and a color,
    this function extends the matrix using the dimensions given by xShape and
    yShape by coloring the extra pixels with the given color.
    If xShape or yShape are not given, then they will be equal to matrix.shape[0]
    of matrix.shape[1], respectively.
    The position of the input matrix in the output matrix can be given by
    specifying "tl", "tr", "bl" or "br" (top-left/top-right/bot-left/bot-right).
    The default is top-left.
    """
    if isSquare:
        if goodDimension=='x':
            xShape = matrix.shape[0]
            yShape = matrix.shape[0]
        if goodDimension=='y':
            xShape = matrix.shape[1]
            yShape = matrix.shape[1]
    if xShape==None:
        xShape = matrix.shape[0]
    if yShape==None:
        yShape = matrix.shape[1]
    m = np.full((xShape, yShape), color, dtype=np.uint8)
    if position=="tl":
        m[0:matrix.shape[0], 0:matrix.shape[1]] = matrix.m.copy()
    elif position=="tr":
        m[0:matrix.shape[0], yShape-matrix.shape[1]:yShape] = matrix.m.copy()
    elif position=="bl":
        m[xShape-matrix.shape[0]:xShape, 0:matrix.shape[1]] = matrix.m.copy()
    else:
        m[xShape-matrix.shape[0]:xShape, yShape-matrix.shape[1]:yShape] = matrix.m.copy()
    
    return m

def getBestExtendMatrix(t):
    """
    If t.backgroundColor!=-1 and if it makes sense.
    """
    if t.backgroundColor==-1:
        totalColorCount = Counter()
        for s in t.trainSamples:
            totalColorCount += s.inMatrix.colorCount
        background = max(totalColorCount.items(), key=operator.itemgetter(1))[0]
    else:
        background=t.backgroundColor
    bestScore = 1000
    bestFunction = partial(identityM)
    
    # Define xShape and yShape:
    xShape = None
    yShape = None
    isSquare=False
    goodDimension=None
    
    # If the outShape is given, easy
    if t.sameOutShape:
        xShape=t.outShape[0]
        yShape=t.outShape[1]
    # If the output xShape is always the same and the yShape keeps constant, pass the common xShape
    elif len(set([s.outMatrix.shape[0] for s in t.trainSamples]))==1:
        if all([s.outMatrix.shape[1]==s.inMatrix.shape[1] for s in t.trainSamples]):
            xShape=t.trainSamples[0].outMatrix.shape[0]  
    # If the output yShape is always the same and the xShape keeps constant, pass the common yShape
    elif len(set([s.outMatrix.shape[1] for s in t.trainSamples]))==1:
        if all([s.outMatrix.shape[0]==s.inMatrix.shape[0] for s in t.trainSamples]):
            yShape=t.trainSamples[0].outMatrix.shape[1] 
    # If the matrix is always squared, and one dimension (x or y) is fixed, do this:
    elif all([s.outMatrix.shape[0]==s.outMatrix.shape[1] for s in t.trainSamples]):
        isSquare=True
        if all([s.outMatrix.shape[1]==s.inMatrix.shape[1] for s in t.trainSamples]):
            goodDimension='y'     
        elif all([s.outMatrix.shape[0]==s.inMatrix.shape[0] for s in t.trainSamples]):
            goodDimension='x'      

    for position in ["tl", "tr", "bl", "br"]:
        f = partial(extendMatrix, color=background, xShape=xShape, yShape=yShape,\
                    position=position, isSquare=isSquare, goodDimension=goodDimension)
        bestFunction, bestScore = updateBestFunction(t, f, bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
    return bestFunction
    
def getFactor(matrix, factor):
    """
    Given a Task.Task.inShapeFactor (that can be a string), this function
    returns its corresponding tuple for the given matrix.
    """
    if factor == "squared":
        f = (matrix.shape[0], matrix.shape[1])
    elif factor == "xSquared":
        f = (matrix.shape[0], 1)
    elif factor == "ySquared":
        f = (1, matrix.shape[1])
    elif factor == "nColors":
        f = (matrix.nColors, matrix.nColors)
    elif factor == "nColors-1":
        f = (matrix.nColors-1, matrix.nColors-1)
    else:
        f = factor
    return f

def multiplyPixels(matrix, factor):
    """
    Factor is a 2-dimensional tuple.
    The output matrix has shape matrix.shape*factor. Each pixel of the input
    matrix is expanded by factor.
    """
    factor = getFactor(matrix, factor)
    m = np.zeros(tuple(s * f for s, f in zip(matrix.shape, factor)), dtype=np.uint8)
    for i,j in np.ndindex(matrix.m.shape):
        for k,l in np.ndindex(factor):
            m[i*factor[0]+k, j*factor[1]+l] = matrix.m[i,j]
    return m

def multiplyMatrix(matrix, factor):
    """
    Copy the matrix "matrix" into every submatrix of the output, which has
    shape matrix.shape * factor.
    """
    factor = getFactor(matrix, factor)
    m = np.zeros(tuple(s * f for s, f in zip(matrix.shape, factor)), dtype=np.uint8)
    for i,j in np.ndindex(factor):
        m[i*matrix.shape[0]:(i+1)*matrix.shape[0], j*matrix.shape[1]:(j+1)*matrix.shape[1]] = matrix.m.copy()
    return m

def matrixTopLeft(matrix, factor, background=0):
    """
    Copy the matrix into the top left corner of the multiplied matrix
    """
    factor = getFactor(matrix, factor)
    m = np.full(tuple(s * f for s, f in zip(matrix.shape, factor)), background, dtype=np.uint8)
    m[0:matrix.shape[0], 0:matrix.shape[1]] = matrix.m.copy()
    return m
    
def matrixBotRight(matrix, factor, background=0):
    """
    Copy the matrix into the bottom right corner of the multiplied matrix
    """
    factor = getFactor(matrix, factor)
    m = np.full(tuple(s * f for s, f in zip(matrix.shape, factor)), background, dtype=np.uint8)
    m[(factor[0]-1)*matrix.shape[0]:factor[0]*matrix.shape[0], \
      (factor[1]-1)*matrix.shape[1]:factor[1]*matrix.shape[1]]
    return m

def getBestMosaic(t):
    """
    Given a task t, this function tries to find the best way to generate a
    mosaic, given that the output shape is always bigger than the input shape
    with a shape factor that makes sense.
    A mosaic is a matrix that takes an input matrix as reference, and then
    copies it many times. The copies can include rotations or mirrorings.
    """
    factor = t.inShapeFactor
    ops = []
    ops.append(partial(identityM))
    ops.append(partial(mirror, axis="lr"))
    ops.append(partial(mirror, axis="ud"))
    ops.append(partial(rotate, angle=180))
    if t.inMatricesSquared:
        ops.append(partial(mirror, axis="d1"))
        ops.append(partial(mirror, axis="d2"))
        ops.append(partial(rotate, angle=90))
        ops.append(partial(rotate, angle=270))
    bestOps = []
    for i in range(factor[0]):
        bestOps.append([])
        for j in range(factor[1]):
            bestScore = 1000
            bestOp = partial(identityM)
            for op in ops:
                score = 0
                for s in t.trainSamples:
                    inM = s.inMatrix.m.copy()
                    outM = s.outMatrix.m[i*inM.shape[0]:(i+1)*inM.shape[0], j*inM.shape[1]:(j+1)*inM.shape[1]]
                    score += incorrectPixels(op(inM),outM)
                if score < bestScore:
                    bestScore = score
                    bestOp = op
                    if score==0:
                        break
            bestOps[i].append(bestOp)
    return bestOps

def generateMosaic(matrix, ops, factor):
    """
    Generates a mosaic from the given matrix using the operations given in the
    list ops. The output matrix has shape matrix.shape*factor.
    """
    m = np.zeros(tuple(s * f for s, f in zip(matrix.shape, factor)), dtype=np.uint8)
    for i in range(factor[0]):
        for j in range(factor[1]):
            m[i*matrix.shape[0]:(i+1)*matrix.shape[0], j*matrix.shape[1]:(j+1)*matrix.shape[1]] = \
            ops[i][j](matrix)
    return m

# Only if the factor is squared
def getBestMultiplyMatrix(t, falseColor):  
    def getFullMatrix(matrix, color):
        return np.full(matrix.shape, color, dtype=np.uint8)
    # Possible operations on the matrix
    ops = []
    ops.append(partial(identityM))
    ops.append(partial(mirror, axis="lr"))
    ops.append(partial(mirror, axis="ud"))
    ops.append(partial(rotate, angle=180))
    if t.inMatricesSquared:
        ops.append(partial(mirror, axis="d1"))
        ops.append(partial(mirror, axis="d2"))
        ops.append(partial(rotate, angle=90))
        ops.append(partial(rotate, angle=270))
    if all([n==2 for n in t.nInColors]):
        ops.append(partial(switchColors))
    
    # Conditions
    def trueCondition(matrix, pixel):
        return True
    def maxColor(matrix, pixel):
        x = [k for k, v in sorted(matrix.colorCount.items(), key=lambda item: item[1], reverse=True)]
        if len(x)<2 or matrix.colorCount[x[0]]!=matrix.colorCount[x[1]]:
            return pixel==max(matrix.colorCount, key=matrix.colorCount.get)
        else:
            return False
    def minColor(matrix,pixel):
        x = [k for k, v in sorted(matrix.colorCount.items(), key=lambda item: item[1])]
        if len(x)<2 or matrix.colorCount[x[-1]]!=matrix.colorCount[x[-2]]:
            return pixel==min(matrix.colorCount, key=matrix.colorCount.get)
        else:
            return False
    def isColor(matrix, pixel, color):
        return pixel==color
    def nonZero(matrix, pixel):
        return pixel!=0
    def zero(matrix, pixel):
        return pixel==0
    conditions = []
    conditions.append(partial(trueCondition))
    conditions.append(partial(maxColor))
    conditions.append(partial(minColor))
    conditions.append(partial(nonZero))
    conditions.append(partial(zero))
    for c in t.colors:
        conditions.append(partial(isColor, color=c))

    bestScore = 1000
    for op, cond in product(ops, conditions):
        score = 0
        for s in t.trainSamples:
            factor = getFactor(s.inMatrix, t.inShapeFactor)
            for i,j in np.ndindex(factor):
                inM = s.inMatrix.m.copy()
                outM = s.outMatrix.m[i*inM.shape[0]:(i+1)*inM.shape[0], j*inM.shape[1]:(j+1)*inM.shape[1]]
                if cond(s.inMatrix, inM[i,j]):
                    score += incorrectPixels(op(inM),outM)
                else:
                    score += incorrectPixels(getFullMatrix(inM, falseColor), outM)
        if score < bestScore:
            bestScore = score
            opCond = (op, cond)
            if score==0:
                return opCond
    return opCond

def doBestMultiplyMatrix(matrix, opCond, falseColor):
    factor = matrix.shape
    m = np.full(tuple(s * f for s, f in zip(matrix.shape, factor)), falseColor, dtype=np.uint8)
    for i,j in np.ndindex(factor):
        if opCond[1](matrix, matrix.m[i,j]):
            m[i*matrix.shape[0]:(i+1)*matrix.shape[0], j*matrix.shape[1]:(j+1)*matrix.shape[1]] = \
            opCond[0](matrix)
    return m

# %% Multiply pixels

def multiplyPixelsAndAnd(matrix, factor, falseColor):
    """
    This function basically is the same as executing the functions
    multiplyPixels, multiplyMatrix, and executing pixelwiseAnd with these two
    matrices as inputs
    """
    factor = getFactor(matrix, factor)
    m = matrix.m.copy()
    multipliedM = multiplyPixels(matrix, factor)
    for i,j in np.ndindex(factor):
        newM = multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]]
        multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]] = pixelwiseAnd([m, newM], falseColor)
    return multipliedM

def multiplyPixelsAndOr(matrix, factor, falseColor):
    """
    This function basically is the same as executing the functions
    multiplyPixels, multiplyMatrix, and executing pixelwiseOr with these two
    matrices as inputs
    """
    factor = getFactor(matrix, factor)
    m = matrix.m.copy()
    multipliedM = multiplyPixels(matrix, factor)
    for i,j in np.ndindex(factor):
        newM = multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]]
        multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]] = pixelwiseOr([m, newM], falseColor)
    return multipliedM

def multiplyPixelsAndXor(matrix, factor, falseColor):
    """
    This function basically is the same as executing the functions
    multiplyPixels, multiplyMatrix, and executing pixelwiseXor with these two
    matrices as inputs
    """
    factor = getFactor(matrix, factor)
    m = matrix.m.copy()
    multipliedM = multiplyPixels(matrix, factor)
    for i,j in np.ndindex(factor):
        newM = multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]]
        multipliedM[i*m.shape[0]:(i+1)*m.shape[0], j*m.shape[1]:(j+1)*m.shape[1]] = pixelwiseXor(m, newM, falseColor)
    return multipliedM

# %% Operations considering all submatrices of task with outShapeFactor
    
def getSubmatrices(m, factor):
    """
    Given a matrix m and a factor, this function returns a list of all the
    submatrices with shape determined by the factor.
    """
    matrices = []
    nRows = int(m.shape[0] / factor[0])
    nCols = int(m.shape[1] / factor[1])
    for i,j in np.ndindex(factor):
        matrices.append(m[i*nRows:(i+1)*nRows, j*nCols:(j+1)*nCols])
    return matrices

def outputIsSubmatrix(t, isGrid=False):
    """
    Given a task t that has outShapeFactor, this function returns true if any
    of the submatrices is equal to the output matrix for every sample.
    """
    for sample in t.trainSamples:
        if isGrid:
            matrices = [c[0].m for c in sample.inMatrix.grid.cellList]
        else:
            matrices = getSubmatrices(sample.inMatrix.m, sample.outShapeFactor)
        anyIsSubmatrix = False
        for m in matrices:
            if np.array_equal(m, sample.outMatrix.m):
                anyIsSubmatrix = True
                break
        if not anyIsSubmatrix:
            return False
    return True

def selectSubmatrixWithMaxColor(matrix, color, outShapeFactor=None, isGrid=False):
    """
    Given a matrix, this function returns the submatrix with most appearances
    of the color given. If the matrix is not a grid, an outShapeFactor must be
    specified.
    """
    if isGrid:
        matrices = [c[0].m for c in matrix.grid.cellList]
    else:
        matrices = getSubmatrices(matrix.m, outShapeFactor)
        
    maxCount = 0
    matricesWithProperty = 0
    bestMatrix = None
    for mat in matrices:
        m = Task.Matrix(mat)
        if color in m.colors:
            if m.colorCount[color]>maxCount:
                bestMatrix = mat.copy()
                maxCount = m.colorCount[color]
                matricesWithProperty = 1
            if m.colorCount[color]==maxCount:
                matricesWithProperty += 1
    if matricesWithProperty!=1:
        return matrix.m.copy()
    else:
        return bestMatrix
    
def selectSubmatrixWithMinColor(matrix, color, outShapeFactor=None, isGrid=False):
    """
    Given a matrix, this function returns the submatrix with least appearances
    of the color given. If the matrix is not a grid, an outShapeFactor must be
    specified.
    """
    if isGrid:
        matrices = [c[0].m for c in matrix.grid.cellList]
    else:
        matrices = getSubmatrices(matrix.m, outShapeFactor)
        
    minCount = 1000
    matricesWithProperty = 0
    bestMatrix = None
    for mat in matrices:
        m = Task.Matrix(mat)
        if color in m.colors:
            if m.colorCount[color]<minCount:
                bestMatrix = mat.copy()
                minCount = m.colorCount[color]
                matricesWithProperty = 1
            elif m.colorCount[color]==minCount:
                matricesWithProperty += 1
    if matricesWithProperty!=1:
        return matrix.m.copy()
    else:
        return bestMatrix
    
def selectSubmatrixWithMostColors(matrix, outShapeFactor=None, isGrid=False):
    """
    Given a matrix, this function returns the submatrix with the most number of
    colors. If the matrix is not a grid, an outShapeFactor must be specified.
    """
    if isGrid:
        matrices = [c[0].m for c in matrix.grid.cellList]
    else:
        matrices = getSubmatrices(matrix.m, outShapeFactor)
        
    maxNColors = 0
    matricesWithProperty = 0
    bestMatrix = None
    for mat in matrices:
        m = Task.Matrix(mat)
        if len(m.colorCount)>maxNColors:
            bestMatrix = mat.copy()
            maxNColors = len(m.colorCount)
            matricesWithProperty = 1
        elif len(m.colorCount)==maxNColors:
            matricesWithProperty += 1
    if matricesWithProperty!=1:
        return matrix.m.copy()
    else:
        return bestMatrix
    
def selectSubmatrixWithLeastColors(matrix, outShapeFactor=None, isGrid=False):
    """
    Given a matrix, this function returns the submatrix with the least number
    of colors. If the matrix is not a grid, an outShapeFactor must be
    specified.
    """
    if isGrid:
        matrices = [c[0].m for c in matrix.grid.cellList]
    else:
        matrices = getSubmatrices(matrix.m, outShapeFactor)
        
    minNColors = 1000
    matricesWithProperty = 0
    bestMatrix = None
    for mat in matrices:
        m = Task.Matrix(mat)
        if len(m.colorCount)<minNColors:
            bestMatrix = mat.copy()
            minNColors = len(m.colorCount)
            matricesWithProperty = 1
        elif len(m.colorCount)==minNColors:
            matricesWithProperty += 1
    if matricesWithProperty!=1:
        return matrix.m.copy()
    else:
        return bestMatrix
        
def getBestSubmatrixPosition(t, outShapeFactor=None, isGrid=False):
    """
    Given a task t, and assuming that all the input matrices have the same
    shape and all the ouptut matrices have the same shape too, this function
    tries to check whether the output matrix is just the submatrix in a given
    position. If that's the case, it returns the position. Otherwise, it
    returns 0.
    """
    iteration = 0
    possiblePositions = []
    for sample in t.trainSamples:
        if isGrid:
            matrices = [c[0].m for c in sample.inMatrix.grid.cellList]
        else:
            matrices = getSubmatrices(sample.inMatrix.m, outShapeFactor)
            
        possiblePositions.append(set())
        for m in range(len(matrices)):
            if np.array_equal(matrices[m], sample.outMatrix.m):
                possiblePositions[iteration].add(m)
        
        iteration += 1
    positions = set.intersection(*possiblePositions)
    if len(positions)==1:
        return next(iter(positions))
    else:
        return 0
                
def selectSubmatrixInPosition(matrix, position, outShapeFactor=None, isGrid=False):
    """
    Given a matrix and a position, this function returns the submatrix that
    appears in the given position (submatrices are either defined by
    outShapeFactor or by the shape of the grid cells).
    """
    if isGrid:
        matrices = [c[0].m for c in matrix.grid.cellList]
    else:
        matrices = getSubmatrices(matrix.m, outShapeFactor)
        
    return matrices[position].copy()

def maxColorFromCell(matrix):
    """
    Only to be called if matrix.isGrid.
    Given a matrix with a grid, this function returns a matrix with the same
    shape as the grid. Every pixel of the matrix will be colored with the 
    color that appears the most in the corresponding cell of the grid.
    """
    m = np.zeros(matrix.grid.shape, dtype=np.uint8)
    for i,j in np.ndindex(matrix.grid.shape):
        color = max(matrix.grid.cells[i][j][0].colorCount.items(), key=operator.itemgetter(1))[0]
        m[i,j] = color
    return m

def colorAppearingXTimes(matrix, times):
    m = np.zeros(matrix.grid.shape, dtype=np.uint8)
    for i,j in np.ndindex(matrix.grid.shape):
        for k,v in matrix.grid.cells[i][j][0].colorCount.items():
            if v==times:
                m[i,j] = k
    return m
        
def pixelwiseAndInSubmatrices(matrix, factor, falseColor, targetColor=None, trueColor=None):
    matrices = getSubmatrices(matrix.m.copy(), factor)
    return pixelwiseAnd(matrices, falseColor, targetColor, trueColor)

def pixelwiseOrInSubmatrices(matrix, factor, falseColor, targetColor=None, trueColor=None, \
                             trueValues=None):
    matrices = getSubmatrices(matrix.m.copy(), factor)
    return pixelwiseOr(matrices, falseColor, targetColor, trueColor, trueValues)

def pixelwiseXorInSubmatrices(matrix, factor, falseColor, targetColor=None, trueColor=None, \
                              firstTrue=None, secondTrue=None):
    matrices = getSubmatrices(matrix.m.copy(), factor)
    return pixelwiseXor(matrices[0], matrices[1], falseColor, targetColor, trueColor, firstTrue, secondTrue)

# %% Grid only functions

# Operations considering all submatrices of a grid

def pixelwiseAndInGridSubmatrices(matrix, falseColor, targetColor=None, trueColor=None):
    matrices = [c[0].m for c in matrix.grid.cellList]
    return pixelwiseAnd(matrices, falseColor, targetColor, trueColor)

def pixelwiseOrInGridSubmatrices(matrix, falseColor, targetColor=None, trueColor=None, \
                                 trueValues=None):
    matrices = [c[0].m for c in matrix.grid.cellList]
    return pixelwiseOr(matrices, falseColor, targetColor, trueColor, trueValues)

def pixelwiseXorInGridSubmatrices(matrix, falseColor, targetColor=None, trueColor=None, \
                                  firstTrue=None, secondTrue=None):
    m1 = matrix.grid.cellList[0][0].m.copy()
    m2 = matrix.grid.cellList[1][0].m.copy()
    return pixelwiseXor(m1, m2, falseColor, targetColor, trueColor, firstTrue, secondTrue)


def paintGridCellWithMostColor(matrix, color, outColor, fixedColors):
    m = matrix.m.copy()
    maxColorPixels = 0
    for i,j in np.ndindex(matrix.grid.shape):
        if color in matrix.grid.cells[i][j][0].colorCount.keys():
            if matrix.grid.cells[i][j][0].colorCount[color]>maxColorPixels:
                maxColorPixels = matrix.grid.cells[i][j][0].colorCount[color]
                bestPosition = (i,j)
    
    position = matrix.grid.cells[bestPosition[0]][bestPosition[1]][1]
    for i,j in np.ndindex(matrix.grid.cellShape):
        if m[i+position[0], j+position[1]] not in fixedColors:
            m[i+position[0], j+position[1]] = outColor
            
    return m

def orderGridCells(matrix, direction, color):
    m = matrix.m.copy()
    cells = [matrix.grid.cellList[i] for i in range(matrix.grid.nCells)]
    cells.sort(key=lambda x: x[0].colorCount[color])
    
    return m

def getBestOrderGridCells(t):
    #bestScore = 1000
    bestFunction = partial(identityM)
    
    
    return bestFunction

# %% crop all shapes
def cropAllShapes(matrix, background, diagonal=False):
    if diagonal:
        shapes = [shape for shape in matrix.dShapes if shape.color!=background]
    else:
        shapes = [shape for shape in matrix.shapes if shape.color!=background]
    shapes = sorted(shapes, key=lambda x: x.nPixels, reverse=True)
    
    if len(shapes)==0:
        return matrix.m.copy()
    
    m = shapes[0].m.copy()
    for i,j in np.ndindex(m.shape):
        if m[i,j]==255:
            m[i,j] = background
    
    outMatrix = Task.Matrix(m)
    if diagonal:
        outShapes = [shape for shape in outMatrix.dShapes if shape.color==background]
    else:
        outShapes = [shape for shape in outMatrix.shapes if shape.color==background]
    
    for s in outShapes:
        if s.color==background:
            for shape in shapes:
                if shape.hasSameShape(s):
                    m = changeColorShapes(m, [s], shape.color)
                    break
    return m

# %% Stuff added by Roderic"
def getLayerDict(t):
    shList = []
    for s in t.trainSamples:
        shList += [sh.shape for sh in s.outMatrix.shapes]
    sc = Counter(shList)
    if len(sc) == 0:
        return None, dict()
    shape = sc.most_common(1)[0][0]
    layerDict = dict()
    for s in t.trainSamples:
        inM = s.inMatrix.m
        outM = s.outMatrix.m
        xR = inM.shape[0]//shape[0]
        yR = inM.shape[1]//shape[1]
        for x in range(xR):
            for y in range(yR):
                if inM[x*shape[0]:(x+1)*shape[0],y*shape[1]:(y+1)*shape[1]].tobytes() in layerDict:
                    if np.all(layerDict[inM[x*shape[0]:(x+1)*shape[0],y*shape[1]:(y+1)*shape[1]].tobytes()] !=\
                                                outM[x*shape[0]:(x+1)*shape[0],y*shape[1]:(y+1)*shape[1]]):
                        return None, dict()
                layerDict[inM[x*shape[0]:(x+1)*shape[0],y*shape[1]:(y+1)*shape[1]].tobytes()]\
                                    = outM[x*shape[0]:(x+1)*shape[0],y*shape[1]:(y+1)*shape[1]] 
    return shape, layerDict

def subMatToLayer(matrix, shapeAndDict):
    shape = shapeAndDict[0]
    layerDict = shapeAndDict[1]
    m = matrix.m.copy()
    if shape is None:
        return m
    xR = m.shape[0]//shape[0]
    yR = m.shape[1]//shape[1]
    for x in range(xR):
        for y in range(yR):
            if m[x*shape[0]:(x+1)*shape[0],y*shape[1]:(y+1)*shape[1]].tobytes() in layerDict:
                m[x*shape[0]:(x+1)*shape[0],y*shape[1]:(y+1)*shape[1]] =\
                        layerDict[m[x*shape[0]:(x+1)*shape[0],y*shape[1]:(y+1)*shape[1]].tobytes()].copy()
            else:
                return m
    return m
        
def getBestFitToFrame(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    crop = False
    if t.outSmallerThanIn:
        crop = True
    for s in [True, False]:
        bestFunction, bestScore = updateBestFunction(t, partial(fitToFrame, crop=crop, includeFrame=True, scale=s), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(fitToFrame, crop=crop, includeFrame=False, scale=s), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(fitToFrame, crop=crop, includeFrame=True, colorMatch=True), bestScore, bestFunction)
    return bestFunction

def fitToFrame(matrix, crop=False, scale=False, includeFrame=True, colorMatch=False):
    m = matrix.m.copy()
    if len(matrix.partialFrames) != 1:
        return m
    frame = matrix.partialFrames[0]
    found = False
    maux = Task.Matrix(deleteShape(m, frame, matrix.backgroundColor))
    for sh in maux.multicolorDShapes:
        if sh != frame and sh.shape[0]>1 and sh.shape[1]>1:
            m = deleteShape(m, sh, matrix.backgroundColor)
            found = True
            break
    if not found or sh.shape[0] > frame.shape[0] or sh.shape[1] > frame.shape[1]:
        return m
    if scale:
        r = min((frame.shape[0]-2)//sh.shape[0], (frame.shape[1]-2)//sh.shape[1])
        newSh = copy.deepcopy(sh)
        newSh.m = np.repeat(np.repeat(sh.m, r, axis=1), r, axis=0)
        newSh.shape = newSh.m.shape
        newSh.pixels = set([(i,j) for i,j in np.ndindex(newSh.m.shape) if newSh.m[i,j]!=255])
    else:
        newSh = copy.deepcopy(sh)
    newSh.position = (frame.position[0] + (frame.shape[0]-newSh.shape[0])//2, frame.position[1] + (frame.shape[1]-newSh.shape[1])//2)
    if colorMatch:
        if len(set(sh.m[:,0]).intersection(matrix.m[frame.position[0]:frame.position[0]+\
               frame.shape[0],frame.position[1]])-set([255])) == 0:
            newSh.m = np.fliplr(newSh.m)
    m = insertShape(m, newSh)
    if crop:
        bC = matrix.backgroundColor
        if np.all(m == bC):
            return m
        x1, x2, y1, y2 = 0, m.shape[0]-1, 0, m.shape[1]-1
        while x1 <= x2 and np.all(m[x1,:] == bC):
            x1 += 1
        while x2 >= x1 and np.all(m[x2,:] == bC):
            x2 -= 1
        while y1 <= y2 and np.all(m[:,y1] == bC):
            y1 += 1
        while y2 >= y1 and np.all(m[:,y2] == bC):
            y2 -= 1
        if includeFrame:
            return(m[x1:x2+1,y1:y2+1])
        elif x1+1<x2 and y1+1<y2:
            return(m[x1+1:x2,y1+1:y2])
    return m

def getBestCountColors(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    outShape = [None]
    sliced = [False]
    if t.sameOutShape:
        outShape.append(t.trainSamples[0].outMatrix.shape)
    if t.outSmallerThanIn or not t.sameIOShapes:
        sliced += [True]
    if t.sameIOShapes:
        outShape=['inShape']
    for byShape in [True,False]:
        for outCol in [t.backgroundColor, t.trainSamples[0].outMatrix.backgroundColor, 0]:
            for rotate in range(-1,2):
                for flip in [True, False]:
                    for ignore in ['max', 'min', None]:
                        for sl in sliced:
                            for oSh in outShape:
                                bestFunction, bestScore = updateBestFunction(t, partial(countColors,\
                                            rotate=rotate,outBackgroundColor=outCol, flip=flip, sliced=sl,\
                                            ignore=ignore, outShape=oSh, byShape=byShape), bestScore, bestFunction)
                                bestFunction, bestScore = updateBestFunction(t, partial(countColors,\
                                            rotate=rotate,outBackgroundColor=outCol, flip=flip, sliced=sl,\
                                            ignore=ignore, outShape=oSh, byShape=byShape, ignoreBackground=False), bestScore, bestFunction)
                                bestFunction, bestScore = updateBestFunction(t, partial(countColors,\
                                            rotate=rotate,outBackgroundColor=outCol, flip=flip, sliced=sl,\
                                            ignore=ignore, outShape=oSh, byShape=byShape, sortByColor=True), bestScore, bestFunction)
    return bestFunction

def countColors(matrix, outBackgroundColor=-1, outShape=None,ignoreBackground=True,\
                ignore=False, sliced=False, rotate=0, flip=False, byShape=False, sortByColor=False):#diagonal, skip, lay
    if byShape:
        cc = Counter([sh.color for sh in matrix.shapes if (sh.color != matrix.backgroundColor or not ignoreBackground)])
        cc = sorted(cc.items(), key=lambda x: x[1], reverse=True)
    else:
        cc = matrix.colorCount
        #add line to sort geographically
        cc = sorted(cc.items(), key=lambda x: x[1], reverse=True)
    if sortByColor:
        cc = sorted(cc, key=lambda x: x[0])
    if outBackgroundColor == -1:
        bC = matrix.backgroundColor
    else:
        bC = outBackgroundColor
    if ignoreBackground:
        cc = [c for c in cc if c[0] != bC]
    if ignore == 'max':
        cc = [cc[i] for i in range(1,len(cc))]
    elif ignore == 'min':
        cc = [c for c in cc if c[1] == max([c[1] for c in cc]) ]
    if len(cc) == 0:
        return matrix.m.copy()
    if outShape == None:
        m = np.full((max([c[1] for c in cc]),len(cc)), fill_value=bC)
        for j in range(len(cc)):
            for i in range(cc[j][1]):
                if i >= m.shape[0] or j >= m.shape[1]:
                    break
                m[i,j] = cc[j][0]
    else:
        if outShape == 'inShape':
            m = np.full(matrix.shape, fill_value=bC)
            if m.shape[0] > m.shape[1]:
                m = np.rot90(m)
            for j in range(len(cc)):
                for i in range(cc[j][1]):
                    if i >= m.shape[0] or j >= m.shape[1]:
                        break
                    m[i,j] = cc[j][0]
        else:
            m = np.full(outShape, fill_value=bC)
            cc = [c[0] for c in cc for j in range(c[1])]
            i = 0
            while i < m.shape[0]:
                j = 0
                while j < m.shape[1]:
                    if i*m.shape[1]+j >= len(cc):
                        break
                    m[i,j] = cc[i*m.shape[1]+j]
                    j += 1 
                i += 1 
    if sliced:
        m = [m[0,:]]
    m = np.rot90(m, rotate)
    if flip:
        m = np.flipud(m)
    if outShape == 'inShape' and m.shape !=  matrix.shape:
        m = np.rot90(m)
    return m

def getBestCountShapes(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    if all([len(s.inMatrix.shapes)>15 for s in t.trainSamples]):
            return bestFunction
    if t.sameIOShapes:
        oSh = 'inShape'
    elif t.sameOutShape:
        oSh = t.trainSamples[0].outMatrix.shape
    else:
        oSh = None
        
    for outC in set([None]).union(set.intersection(*t.outColors)):
        for inC in set([-1]).union(set.intersection(*t.inColors)):
            for sh in [None] + t.commonInShapes:
                for lay in ['h','d']:
                    for skip in [True, False]:
                        bestFunction, bestScore = updateBestFunction(t, partial(countShapes,color=inC,\
                                        outShape=oSh, lay=lay, outColor=outC, shape=sh, skip=skip), bestScore, bestFunction)
    return bestFunction

def countShapes(matrix, color=-1, shape=None, outColor=None, outShape=None, lay='d', skip=False):
    if color < 0:
        shc = [sh for sh in matrix.shapes]
    else:
        shc = [sh for sh in matrix.shapes if sh.color == color]
    if shape != None:
        shc = [sh for sh in shc if sh == shape]
    if outShape == None:
        m = np.full((len(shc),len(shc)), fill_value=matrix.backgroundColor)
    elif outShape == 'inShape':
        m = np.full(matrix.shape, fill_value=matrix.backgroundColor)
    else:
        m = np.full(outShape, fill_value=matrix.backgroundColor)
    if lay == 'd':
        for d in range(min(len(shc), m.shape[0], m.shape[1])):
            if outColor == None:
                m[d,d] = shc[d].color
            else:
                m[d,d] = outColor
    elif lay == 'h':
        shc = [sh.color for sh in shc]
        if skip:
            shc = [[c,matrix.backgroundColor] for c in shc]
            shc = [c for p in shc for c in p]
        i = 0
        while i < m.shape[0]:
            j = 0
            while j < m.shape[1]:
                if i*m.shape[1]+j >= len(shc):
                    break
                if outColor == None:
                    m[i,j] = shc[i*m.shape[1]+j]
                elif shc[i*m.shape[1] + j] != matrix.backgroundColor:
                    m[i,j] = outColor
                j += 1           
            i += 1 
    return m 

def getBestSymmetrizeAllShapes(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    for cc in set.intersection(*t.inColors).union(set([-1])):
        bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeAllShapes, targetColor = cc), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeAllShapes, targetColor = cc, byColor=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeAllShapes, targetColor = cc, context=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(symmetrizeAllShapes, targetColor = cc, context=True, byColor=True), bestScore, bestFunction)

    return bestFunction

def symmetrizeAllShapes(matrix, diagonal=True, multicolor=True, targetColor=-1, axis=None,\
                        context=False, lr = True, ud = True, byColor=False):
    m = matrix.m.copy()
    bC = matrix.backgroundColor
    if byColor:
        shList = [sh for sh in matrix.shapesByColor if (sh.shape[0]<m.shape[0] and sh.shape[1]<m.shape[1])]
    else:    
        if not multicolor: 
            if diagonal:   
                shList = [sh for sh in matrix.dShapes]
            else:   
                shList = [sh for sh in matrix.shapes]
        else:
            if diagonal: 
                shList = [sh for sh in matrix.multicolorDShapes]
            else:
                shList = [sh for sh in matrix.multicolorShapes]
    if targetColor > -1:
        shList = [sh for sh in shList if hasattr(sh, 'color') and sh.color == targetColor]
    for sh in shList:
        if context:
            shM = m[sh.position[0]:sh.position[0]+sh.shape[0], sh.position[1]:sh.position[1]+sh.shape[1]]
        else:
            shM = sh.m.copy()
        if lr:
            shMlr = np.fliplr(shM)
            for i,j in np.ndindex(sh.shape):
                if shM[i,j] == bC or shM[i,j] == 255:
                    shM[i,j] = shMlr[i,j]
        if ud:
            shMud = np.flipud(shM)
            for i,j in np.ndindex(sh.shape):
                if shM[i,j] == bC or shM[i,j] == 255:
                    shM[i,j] = shMud[i,j]
        if context:
            m[sh.position[0]:sh.position[0]+sh.shape[0], sh.position[1]:sh.position[1]+sh.shape[1]] = shM
        else:
            newInsert = copy.deepcopy(sh)
            newInsert.m = shM
            m = insertShape(m, newInsert)
    return m
    
#paint grids
def paintGridLikeBackground(matrix):
    """
    Ignores the grid by paiting it in the background color (most repeated color or second if first coincides with grid color).
    In some cases cells have more than one color but it is still worth ignoring the grid. 
    """
    m = matrix.m.copy()
    bC = max(matrix.colorCount,key=matrix.colorCount.get)
    if matrix.isGrid:
        m[m==matrix.grid.color] = bC
    elif matrix.isAsymmetricGrid:
        m[m==matrix.asymmetricGrid.color] = bC
    return m

#HOW DO I PASS THE ARGUMENTS????
def paintGridLikeOriginal(matrix, grid):
    """
    Repaint a grid previously painted in the background color.
    """
    m = matrix.m.copy()
    m = insertShape(m, grid)
    return m

def downsizeMode(matrix, newShape, falseColor=None):
    """
    Given a matrix and a shape, this function returns a new matrix with the
    given shape. The elements of the return matrix are given by the colors of 
    each of the submatrices. If a submatrix has more than one color the mode is
    chosen.
    """
    if (matrix.shape[0]%newShape[0])!=0 or (matrix.shape[1]%newShape[1])!=0:
        return matrix.m.copy()
    xBlock = int(matrix.shape[0]/newShape[0])
    yBlock = int(matrix.shape[1]/newShape[1])
    m = np.full(newShape, matrix.backgroundColor, dtype=np.uint8)
    for i,j in np.ndindex(newShape[0], newShape[1]):
        colorCount = Counter(matrix.m[i*xBlock: (i+1)*xBlock, j*yBlock: (j+1)*yBlock].flatten())
        m[i,j] = max(colorCount, key=colorCount.get)
    return m

def getBestColorByPixels(t):
    delPix = False
    if isDeleteTask(t) or t.outSmallerThanIn:
        delPix = True
    bestScore = 1000
    bestFunction = partial(identityM)
    bestFunction, bestScore = updateBestFunction(t, partial(colorByPixels, deletePixels=delPix), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(colorByPixels, colorMap=True, deletePixels=delPix), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(colorByPixels, oneColor=True, deletePixels=delPix), bestScore, bestFunction)
    return bestFunction
    
def colorByPixels(matrix, colorMap=False, oneColor=False, deletePixels=False):
    """
    Attempts to find color changes dictated by pixels. Be it pixels determine the color of the closest shape,\
    be it adjacent pixels determine a color map. 
    """
    m = matrix.m.copy()
    shList = [sh for sh in matrix.shapes if (sh.color != matrix.backgroundColor) and len(sh.pixels)>1]
    pixList = [sh for sh in matrix.dShapes if (sh.color != matrix.backgroundColor) and len(sh.pixels)==1]
    ogPixList = [p for p in pixList]
    if len(shList)==0 or len(pixList)==0 or len(pixList)>15:
        return m
    if colorMap:
        cMap = dict()
        seenP = []
        for p1 in pixList:
            for p2 in pixList:
                if abs(p1.position[1]-p2.position[1])==1 and p1.position[0]==p2.position[0] and p1.color not in cMap.keys():
                    cMap[p1.color] = p2.color
                    seenP.append(p1.position)                   
        if deletePixels:
            for pix in pixList:
                m[pix.position[0], pix.position[1]] = matrix.backgroundColor
        for i,j in np.ndindex(m.shape):
            if m[i,j] in cMap.keys() and (i,j) not in seenP:
                m[i,j] = cMap[m[i,j]]     
    else:
        if oneColor:
            cc = Counter([sh.color for sh in pixList])
            newC = max(cc, key=cc.get)
            if deletePixels:
                m[m==newC]=matrix.backgroundColor
            m[m!=matrix.backgroundColor]=newC
        else:
            if len(pixList) < len(shList):
                return m
            c = 0
            nSh = len(shList)
            while c < nSh:
                minD, newD = 1000, 1000
                bestSh, bestPix = None, None
                for pix in pixList:
                    for i in range(len(pixList)):
                        for sh in shList:
                            newD = min(np.linalg.norm(np.subtract(pix.position,np.add(p, sh.position))) for p in sh.pixels) 
                            if newD < minD:
                                minD = newD
                                bestSh = sh
                                bestPix = pix
                if bestSh != None:
                    for i,j in np.ndindex(bestSh.shape):
                        if bestSh.m[i,j] != 255:
                            m[bestSh.position[0]+i, bestSh.position[1]+j]=bestPix.color
                    c += 1
                    shList.remove(bestSh)
                    pixList.remove(bestPix)
    if deletePixels:
        for pix in ogPixList:
            m[pix.position] = matrix.backgroundColor
    return m
    
def isDeleteTask(t):    
    if hasattr(t, 'colorChanges') and t.backgroundColor in [c[1] for c in t.colorChanges]:
        return True
    return False

#delete shape with x properties
def getBestDeleteShapes(t, multicolor=False, diagonal=True):
    attrs = set(['LaSh','SmSh','MoCl','MoCo','PiXl'])
    bestScore = 1000
    bestFunction = partial(identityM)
    for attr in attrs:
        bestFunction, bestScore = updateBestFunction(t, partial(deleteShapes, diagonal=diagonal, multicolor=multicolor,attributes=set([attr])), bestScore, bestFunction)
    return bestFunction

def getDeleteAttributes(t, diagonal=True):
    bC = max(0, t.backgroundColor)
    if diagonal:
        if t.nCommonInOutDShapes == 0:
            return set()
        attrs = set.union(*[s.inMatrix.getShapeAttributes(backgroundColor=bC,\
                    singleColor=True, diagonals=True)[s.inMatrix.dShapes.index(sh[0])]\
                    for s in t.trainSamples for sh in s.commonDShapes])
        nonAttrs = set()
        c = 0
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=True, diagonals=True)
            for shi in range(len(s.inMatrix.shapes)):
                if any(s.inMatrix.dShapes[shi] == sh2[0] for sh2 in s.commonDShapes) or s.inMatrix.dShapes[shi].color == bC:
                    continue
                else:
                    if c == 0:
                        nonAttrs = shAttrs[shi]
                        c += 1
                    else:
                        nonAttrs = nonAttrs.intersection(shAttrs[shi])
                        c += 1
    else:
        if t.nCommonInOutShapes == 0:
            return set()
        attrs = set.union(*[s.inMatrix.getShapeAttributes(backgroundColor=bC,\
                    singleColor=True, diagonals=False)[s.inMatrix.shapes.index(sh[0])]\
                    for s in t.trainSamples for sh in s.commonShapes])
        nonAttrs = set()
        c = 0
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=True, diagonals=False)
            for shi in range(len(s.inMatrix.shapes)):
                if any(s.inMatrix.shapes[shi] == sh2[0] for sh2 in s.commonShapes) or s.inMatrix.shapes[shi].color == bC:
                    continue
                else:
                    if c == 0:
                        nonAttrs = shAttrs[shi]
                        c += 1
                    else:
                        nonAttrs = nonAttrs.intersection(shAttrs[shi])
                        c += 1
    return set(nonAttrs - attrs)

def deleteShapes(matrix, attributes, diagonal, multicolor):
    m = matrix.m.copy()
    if not multicolor: 
        if diagonal:   
            shList = [sh for sh in matrix.dShapes]
        else:   
            shList = [sh for sh in matrix.shapes]
    else:
        if diagonal: 
            shList = [sh for sh in matrix.multicolorDShapes]
        else:
            shList = [sh for sh in matrix.multicolorShapes]
    attrList = matrix.getShapeAttributes(matrix.backgroundColor, not multicolor, diagonal)
    for shi in range(len(shList)):
        if len(attrList[shi].intersection(attributes)) > 0:
            m = deleteShape(m, shList[shi], matrix.backgroundColor)
    return m

def getBestArrangeShapes(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    if hasattr(t, 'outShape'):
        bestFunction, bestScore = updateBestFunction(t, partial(arrangeShapes, outShape=t.outShape,\
                                                            diagonal=True, multicolor=False), bestScore, bestFunction)
    elif t.outIsInMulticolorShapeSize:
        bestFunction, bestScore = updateBestFunction(t, partial(arrangeShapes, diagonal=True,\
                                                                multicolor=True,outShape='LaSh'), bestScore, bestFunction)
    else:
        bestFunction, bestScore = updateBestFunction(t, partial(arrangeShapes,shByColor=True,outShape='LaSh'), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(arrangeShapes,shByColor=True, fullFrames=True,outShape='LaSh'), bestScore, bestFunction)
    return bestFunction

def arrangeShapes (matrix, outShape = None, multicolor=True, diagonal=True, shByColor=False,\
                   fullFrames=False, outDummyMatrix=None, outDummyColor=0):
    def completeFrames(shape,rotate=False,fill=False):
        'version of symmetrize shape intended for frame-like shapes' 
        m = shape.m.copy()
        if m.shape[0]>m.shape[1]:   
            newm = np.full((m.shape[0], m.shape[0]), fill_value=255)
            sideL = m.shape[0]
        else:
            newm = np.full((m.shape[1], m.shape[1]), fill_value=255)
            sideL = m.shape[1]
        for i,j in np.ndindex(m.shape):
            if m[i,j] != 255:
                if newm[i,j] == 255:
                    newm[i,j] = m[i,j]
                if newm[sideL - j - 1,i] == 255:
                    newm[sideL - j - 1,i] = m[i,j]
                if newm[sideL - i - 1,sideL - j - 1] == 255:
                    newm[sideL - i - 1,sideL - j - 1] = m[i,j]
                if newm[j,sideL - i - 1] == 255 and m[i,j]:
                    newm[j,sideL - i - 1] = m[i,j]
        newSh = copy.deepcopy(shape)
        newSh.m = newm
        newSh.shape = newm.shape
        return newSh
    
    def tessellateShapes (mat, shL, n, bC, rotation=False):
        m = mat.copy()
        arrFound = False
        rot = 1
        """
        Attempts to tessellate matrix mat with background color bC shapes is list sh.
        """
        if rotation:
            rot = 4
            if len(shL[n].pixels)==1:
                rot = 1
            if len(shL[n].pixels)==2:
                rot = 2
        for x in range(rot):
            sh = copy.deepcopy(shL[n])
            sh.m = np.rot90(sh.m,x).copy()
            sh.shape = sh.m.shape
            if mat.shape[0] < sh.shape[0] or mat.shape[1] < sh.shape[1]:
                continue
            for i, j in np.ndindex(tuple(mat.shape[k] - sh.shape[k] + 1 for k in (0,1))):
                if np.all(np.logical_or(m[i: i+sh.shape[0], j: j+sh.shape[1]] == bC, sh.m == 255)):
                    for k, l in np.ndindex(sh.shape):
                        if sh.m[k,l] != 255:
                            m[i+k,j+l] = sh.m[k,l]
                    if n == len(shL) - 1:
                        return m, True
                    m, arrFound = tessellateShapes(m, shL, n+1, bC, rotation)
                    if arrFound:
                        return m, True
                    if not arrFound:
                        for k, l in np.ndindex(sh.shape):
                            if sh.m[k,l] != 255:
                                m[i+k,j+l] = bC
        return m, False
    if shByColor:
        shList = [sh for sh in matrix.shapesByColor]
        if fullFrames:
            shList = [completeFrames(sh) for sh in matrix.shapesByColor]
    else:
        if not multicolor: 
            if diagonal:   
                shList = [sh for sh in matrix.dShapes if sh.color != matrix.backgroundColor]
            else:   
                shList = [sh for sh in matrix.shapes if sh.color != matrix.backgroundColor]
        else:
            if diagonal: 
                shList = [sh for sh in matrix.multicolorDShapes]
            else:
                shList = [sh for sh in matrix.multicolorShapes]
    if len(shList) < 2 or len(shList)>7:
        return matrix.m.copy()
    if outDummyMatrix is None:
        shList.sort(key=lambda x: x.shape[0]*x.shape[1], reverse=True)
        if outShape == 'LaSh':
            outShape = shList[0].shape    
        if outShape == None:
            outShape = matrix.shape
        if all((sh.shape[0]<=outShape[0] and sh.shape[1]<=outShape[1]) for sh in shList) and\
                                sum(len(sh.pixels) for sh in shList) <= outShape[0]*outShape[1]:
            m, tessellate = tessellateShapes(np.full(outShape, fill_value=matrix.backgroundColor),shList,\
                                             0,matrix.backgroundColor)
            if tessellate:
                return m
            m, tessellate = tessellateShapes(np.full(outShape, fill_value=matrix.backgroundColor),shList,\
                                             0,matrix.backgroundColor,rotation=True)
            if tessellate:
                return m
    else:
        m = np.full(outDummyMatrix.shape,fill_value=outDummyColor)
        shC = Counter([sh.shape for sh in shList])
        pSh = shC.most_common(1)[0][0]
        if pSh[0]>m.shape[0] or pSh[1]>m.shape[1] or m.shape[0]%pSh[0] != 0 or m.shape[1]%pSh[1] != 0:
            return matrix.m.copy()
        for i, j in np.ndindex(m.shape[0]//pSh[0], m.shape[1]//pSh[1]):
            for k, l in np.ndindex(matrix.shape[0]-pSh[0]+1, matrix.shape[1]-pSh[1]+1):
                if np.all((matrix.m[k:k+pSh[0],l:l+pSh[1]] == outDummyColor)==(outDummyMatrix[i*pSh[0]:(i+1)*pSh[0],j*pSh[1]:(j+1)*pSh[1]]==0)):
                    m[i*pSh[0]:(i+1)*pSh[0],j*pSh[1]:(j+1)*pSh[1]] = matrix.m[k:k+pSh[0],l:l+pSh[1]]
                    break
        return m
    return matrix.m.copy()

def getBestLayShapes(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    outShape = None
    if all([len(s.inMatrix.shapes) > 15 for s in t.trainSamples]):
        return bestFunction
    if t.sameIOShapes:
        outShape = 'inShape'
    elif hasattr(t, 'outShape'):
        outShape = t.outShape
    if outShape != None:
        for sortBy in ['grid','lr','ud','smallToLarge',set.intersection(*t.inColors)]:
            for reverse in [False, True]:
                for overlap in [(0,0), (1,1), (-1,-1)]:
                    for multicolor in [True, False]:
                        for direction in [(1,0), (0,1), (1,1)]:
                            bestFunction, bestScore = updateBestFunction(t, partial(layShapes, firstPos=(0,0), diagonal=True, multicolor=multicolor,\
                                                            outShape=outShape, overlap=overlap, direction=direction, sortBy=sortBy, reverse=reverse), bestScore, bestFunction)
                        bestFunction, bestScore = updateBestFunction(t, partial(layShapes, firstPos=(1,0), direction=(-1,0), diagonal=True, multicolor=multicolor,\
                                                            outShape=outShape, overlap=overlap, sortBy=sortBy, reverse=reverse), bestScore, bestFunction)
                        bestFunction, bestScore = updateBestFunction(t, partial(layShapes, firstPos=(0,1), direction=(0,-1), diagonal=True, multicolor=multicolor,\
                                                            outShape=outShape, overlap=overlap, sortBy=sortBy, reverse=reverse), bestScore, bestFunction)
                        bestFunction, bestScore = updateBestFunction(t, partial(layShapes, firstPos=(1,1), direction=(-1,-1), diagonal=True, multicolor=multicolor,\
                                                            outShape=outShape, overlap=overlap, sortBy=sortBy, reverse=reverse), bestScore, bestFunction)
    elif t.outSmallerThanIn:
         bestFunction, bestScore = updateBestFunction(t, partial(layShapes,completeRect=True,diagonal=True,\
                                         direction=(0,0),sortBy='smallToLarge', reverse=True, outShape='LaSh',multicolor=False), bestScore, bestFunction)
    return bestFunction

def layShapes(matrix, firstPos=(0,0), direction=(0,1), overlap=(0,0), outShape='inShape', multicolor=True,\
              diagonal=True, sortBy='lrud', completeRect=False, reverse=True):
    def completeRectangles(shape):
        'version of complete rectangles shape intended for frame-like shapes' 
        newSh = copy.deepcopy(shape)
        newSh.m = np.full(shape.shape, fill_value=shape.color)
        newSh.shape = newSh.m.shape
        return newSh
    m = matrix.m.copy()
    """Moves all shapes and lays them in an appropriate way."""
    if not multicolor: 
        if diagonal:   
            shList = [sh for sh in matrix.dShapes if sh.color != matrix.backgroundColor]
        else:   
            shList = [sh for sh in matrix.shapes if sh.color != matrix.backgroundColor]
    else:
        if diagonal: 
            shList = [sh for sh in matrix.multicolorDShapes]
        else:
            shList = [sh for sh in matrix.multicolorShapes]
    if completeRect and (not multicolor):
        shList = [completeRectangles(sh) for sh in shList]
    if sortBy == 'smallToLarge':
        shList.sort(key=lambda x: len(x.pixels), reverse=reverse)
    elif sortBy == 'lr':
        shList.sort(key=lambda x: x.position[1], reverse=reverse)
    elif sortBy == 'ud':
        shList.sort(key=lambda x: x.position[0], reverse=reverse)
    elif sortBy == 'grid':
        shList.sort(key=lambda x: x.position[0])
        newList = []
        shList.sort(key=lambda x: x.position[0])
        gridD = int(len(shList)**(1/2))
        for i in range(gridD):
            newList += sorted(shList[i*gridD: (i + 1)*gridD], key=lambda x: x.position[1])
        shList = [sh for sh in newList]
    elif type(sortBy) == int:
        shList.sort(key=lambda x: x.colorCount[sortBy])
    if len(shList) == 0:
        return m
    #if outShape can't be determined, then
    if outShape == 'inShape':
        m = np.full(matrix.shape, fill_value=matrix.backgroundColor)
    elif outShape == 'LaSh' and sortBy == 'smallToLarge' and reverse:
        m = np.full(shList[0].m.shape, fill_value=matrix.backgroundColor)
    else:
        m = np.full(outShape, fill_value=matrix.backgroundColor)
    shList = [sh for sh in shList if (sh.shape[0] <= m.shape[0] and sh.shape[1] <= m.shape[1])]
    startPos = (firstPos[0]*(m.shape[0]), firstPos[1]*(m.shape[1]))
    (currentX, currentY) = startPos
    for sh in shList:
        if currentX + sh.shape[0]*direction[0] > m.shape[0] or currentX + sh.shape[0]*direction[0] < 0:
            (currentX, currentY) = (startPos[0], currentY + sh.shape[1] - overlap[1])
            if currentY > m.shape[1] or currentY < 0:
                return matrix.m.copy()
        if currentY + sh.shape[1]*direction[1] > m.shape[1] or currentY + sh.shape[1]*direction[1] < 0:
            (currentX, currentY) = (currentX + sh.shape[0] - overlap[0], startPos[1])
            if currentX > m.shape[0] or currentX < 0:
                return matrix.m.copy()
        newInsert = copy.deepcopy(sh)
        newInsert.position = (currentX, currentY)
        if direction[0] < 0:
            newInsert.position = (newInsert.position[0] - sh.shape[0], newInsert.position[1])
        if direction[1] < 0:
            newInsert.position = (newInsert.position[0], newInsert.position[1] - sh.shape[1])
        m = insertShape(m, newInsert)
        currentX, currentY = (currentX + (sh.shape[0]- overlap[0])*direction[0] , currentY + (sh.shape[1]- overlap[1])*direction[1])
    return m

def getBestAlignShapes(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    if t.sameIOShapes:
        for cc in set.intersection(*t.inColors):
            bestFunction, bestScore = updateBestFunction(t, partial(alignShapes, refColor=cc), bestScore, bestFunction)
    elif t.outSmallerThanIn:
        bestFunction, bestScore = updateBestFunction(t, partial(alignShapes, compress=True, crop=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(alignShapes, compress=False, crop=True), bestScore, bestFunction)
    return bestFunction

def alignShapes(matrix, compress=True, diagonal=True, multicolor=False, refColor=None, crop=False):
    """
    Attempts to align the shapes in matrix, maybe with a reference unmoved shape, maybe cropping.
    """
    m = matrix.m.copy()
    if not multicolor: 
        if diagonal:   
            shList = [sh for sh in matrix.dShapes if sh.color != matrix.backgroundColor]
        else:   
            shList = [sh for sh in matrix.shapes if sh.color != matrix.backgroundColor]
    else:
        if diagonal: 
            shList = [sh for sh in matrix.multicolorDShapes]
        else:
            shList = [sh for sh in matrix.multicolorShapes]
    if len(shList) < 2:
        return m
    shList.sort(key=lambda x: x.position[1])
    arrFound = False
    if all(shList[i].position[1]+shList[i].shape[1] <= shList[i+1].position[1] for i in range(len(shList)-1)):
        arrFound = (0,1)
    if arrFound == False:
        shList.sort(key=lambda x: x.position[0])
        if all(shList[i].position[0]+shList[i].shape[0] <= shList[i+1].position[0] for i in range(len(shList)-1)):
           arrFound = (1,0)
    if arrFound == False:
        return m  
    for sh in shList:
        m = deleteShape(m, sh, matrix.backgroundColor)
    (currentX, currentY) = (0, 0)
    if refColor != None and not crop:
        refPos = -1
        for sh in shList:
            if sh.color == refColor:
                refPos = sh.position
                break
        if refPos == -1:
            return matrix.m.copy()
        for sh in shList:
            newInsert = copy.deepcopy(sh)
            if arrFound == (0,1):
                newInsert.position = (refPos[0], sh.position[1])
            elif arrFound == (1,0):
                newInsert.position = (sh.position[0], refPos[1])
            m = insertShape(m, newInsert)
    else:
        for sh in shList:
            newInsert = copy.deepcopy(sh)
            if compress:
                newInsert.position = (currentX, currentY)
                (currentX, currentY) = (currentX + sh.shape[0]*arrFound[0], currentY + sh.shape[1]*arrFound[1])
            else:
                newInsert.position = (0,0)
            m = insertShape(m, newInsert)
        
    if crop:
        bC = matrix.backgroundColor
        if np.all(m == bC):
            return m
        x1, x2, y1, y2 = 0, m.shape[0]-1, 0, m.shape[1]-1
        while x1 <= x2 and np.all(m[x1,:] == bC):
            x1 += 1
        while x2 >= x1 and np.all(m[x2,:] == bC):
            x2 -= 1
        while y1 <= y2 and np.all(m[:,y1] == bC):
            y1 += 1
        while y2 >= y1 and np.all(m[:,y2] == bC):
            y2 -= 1
        return(m[x1:x2+1,y1:y2+1])
    else:
        return m
    
#replicate shape
def isReplicateTask(t):
    #First look at shapes that replicate
    if all(any(sh[2] > 1 for sh in s.commonMulticolorDShapes) for s in t.trainSamples):
        return [True, True, True]
    elif all(any(sh[2] > 1 for sh in s.commonMulticolorShapes) for s in t.trainSamples):
        return [True, True, False]
    elif all(any(sh[2] > 1 for sh in s.commonShapes) for s in t.trainSamples):
        return [True, False, False]
    elif all(any(sh[2] > 1 for sh in s.commonDShapes) for s in t.trainSamples):
        return [True, False, True]
    return [False]

def getBestReplicateShapes(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    deleteOriginal = False
    multicolor = True
    diagonal = True
    if isReplicateTask(t)[0]:
        multicolor = isReplicateTask(t)[1]
        diagonal = isReplicateTask(t)[2]
    if isDeleteTask(t):
        deleteOriginal = True
    
    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, diagonal=diagonal, multicolor=multicolor,deleteOriginal=deleteOriginal,\
                                                            anchorType='subframe'), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, diagonal=diagonal, multicolor=multicolor,deleteOriginal=deleteOriginal,\
                                                            anchorType='subframe', allCombs=False,attributes=set(['MoCl'])), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, diagonal=diagonal, multicolor=multicolor,deleteOriginal=deleteOriginal,\
                                                            anchorType='subframe', allCombs=False,scale=True,attributes=set(['MoCl'])), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, diagonal=diagonal, multicolor=multicolor,deleteOriginal=deleteOriginal,\
                                                            anchorType='subframe', allCombs=True,attributes=set(['MoCl'])), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, diagonal=diagonal, multicolor=multicolor,deleteOriginal=deleteOriginal,\
                                                            anchorType='subframe', allCombs=True,scale=True,attributes=set(['MoCl'])), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes,diagonal=diagonal, multicolor=False, anchorType='subframe', allCombs=False,\
                                                                adoptAnchorColor=True), bestScore, bestFunction)
    if bestScore == 0:
        return bestFunction
    
    if isReplicateTask(t)[0]:
        if bestScore == 0:
            return bestFunction
        for attributes in [set(['MoCl'])]:
            cC = Counter([cc[0] for cc in t.colorChanges])
            cc = max(cC, key=cC.get)
            bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=diagonal, multicolor=multicolor, anchorType='all', anchorColor=cc,\
                                    mirror=None, rotate=0, allCombs=True, scale=False, deleteOriginal=deleteOriginal), bestScore, bestFunction)
            bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=diagonal, multicolor=multicolor, anchorType='all', anchorColor=cc,\
                                    mirror=None, rotate=0, allCombs=True, scale=False, deleteOriginal=deleteOriginal), bestScore, bestFunction)
            if bestScore == 0:
                return bestFunction
            for mirror in [None, 'lr', 'ud']:
                for rotate in range(0, 4):
                    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=diagonal, multicolor=multicolor, anchorType='all', anchorColor=cc,\
                                    mirror=mirror, rotate=rotate, allCombs=False, scale=False, deleteOriginal=deleteOriginal), bestScore, bestFunction)
                    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=diagonal, multicolor=multicolor, anchorType='all', anchorColor=cc,\
                                    mirror=mirror, rotate=rotate, allCombs=False, scale=True, deleteOriginal=deleteOriginal), bestScore, bestFunction)
                    if bestScore == 0:      
                        return bestFunction
    cC = Counter([cc[0] for cc in t.colorChanges])
    cc = max(cC, key=cC.get)
    bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, diagonal=False, multicolor=multicolor, anchorType='all', anchorColor=cc,\
                        allCombs=False, scale=False, deleteOriginal=deleteOriginal,perfectFit=True), bestScore, bestFunction)
    for attributes in [set(['UnCo'])]:    
        bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=True, multicolor=False, anchorType='all', anchorColor=cc,\
                        allCombs=True, scale=False, deleteOriginal=deleteOriginal), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=True, multicolor=False, anchorType='all', anchorColor=cc,\
                        allCombs=False, scale=False, deleteOriginal=deleteOriginal, perfectFit=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=True, multicolor=False, anchorType='all', anchorColor=cc,\
                        allCombs=False, scale=True, deleteOriginal=deleteOriginal, perfectFit=False), bestScore, bestFunction)
   
    if t.hasPartialFrame:
        for attributes in [set(['IsRef'])]:    
            bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=True, multicolor=False, anchorType='all', anchorColor=cc,\
                            allCombs=True, scale=False, deleteOriginal=deleteOriginal), bestScore, bestFunction)
            bestFunction, bestScore = updateBestFunction(t, partial(replicateShapes, attributes=attributes, diagonal=True, multicolor=False, anchorType='all', anchorColor=cc,\
                            allCombs=False, scale=False, deleteOriginal=deleteOriginal, perfectFit=True), bestScore, bestFunction)
    return bestFunction

def replicateShapes(matrix, attributes=None, diagonal=False, multicolor=True, anchorType=None, anchorColor=0,\
                    mirror=None, rotate=0, allCombs=False, scale=False, deleteOriginal=False, perfectFit=False,
                    adoptAnchorColor=False, deleteAnchor=False):
    m = matrix.m.copy()
    #first find the shape or shapes to replicate
    if diagonal:
        if multicolor:
            shList = matrix.multicolorDShapes
        else:
            shList = matrix.dShapes
    else:
        if multicolor:
            shList = matrix.multicolorShapes
        else:
            shList = matrix.shapes
    if attributes != None:
        repList = []
        attrList = matrix.getShapeAttributes(backgroundColor=matrix.backgroundColor,\
                                             singleColor=not multicolor, diagonals=diagonal)
        for shi in range(len(shList)):
            if len(attrList[shi].intersection(attributes)) == 1:
                repList.append(shList[shi])
        if len(repList) == 0:
            return m
    else:
        if multicolor:
            repList = [sh for sh in shList if (len(sh.pixels)>1 and not sh.isSquare)]
        else:
            repList = [sh for sh in shList if (sh.color != matrix.backgroundColor and not sh.isSquare)]
    delList = [sh for sh in repList]
    if len(repList) > 10:
        return m
    #apply transformations to replicating shapes
    if allCombs:
        newList = []
        for repShape in repList:
            for r in range(0,4):
                mr, mrM = np.rot90(repShape.m.copy(), r), np.rot90(repShape.m[::-1,::].copy(), r)
                newRep, newRepM = copy.deepcopy(repShape), copy.deepcopy(repShape)
                newRep.m, newRepM.m = mr, mrM
                newRep.shape, newRepM.shape = mr.shape, mrM.shape
                newList.append(newRep)
                newList.append(newRepM)
        repList = [sh for sh in newList]           
    elif mirror == 'lr' and len(repList) == 1:
        newRep = copy.deepcopy(repList[0])
        newRep.m = repList[0].m[::,::-1]
        repList = [newRep]
    elif mirror == 'ud' and len(repList) == 1:
        newRep = copy.deepcopy(repList[0])
        newRep.m = repList[0].m[::-1,::]
        repList = [newRep]
    elif rotate > 0 and len(repList) == 1:
        newRep = copy.deepcopy(repList[0])
        newRep.m = np.rot90(repList[0].m,rotate)
        newRep.shape = newRep.m.shape
        repList = [newRep]
    if scale == True:
        newRepList=[]
        for repShape in repList:
            for sc in range(4,0,-1):
                newRep = copy.deepcopy(repShape)
                newRep.m = np.repeat(np.repeat(repShape.m, sc, axis=1), sc, axis=0)
                newRep.shape = newRep.m.shape
                newRep.pixels = set([(i,j) for i,j in np.ndindex(newRep.m.shape) if newRep.m[i,j]!=255])
                newRepList.append(newRep)
        repList = [sh for sh in newRepList]
    repList.sort(key=lambda x: len(x.pixels), reverse=True)
    if anchorType == 'subframe' and scale:
        repList.sort(key=lambda x: len(x.pixels))
    #then find places to replicate
    if anchorType == 'all':
        seenM = np.zeros(m.shape, dtype=int)
        for repSh in repList:
            if np.all(np.logical_or(repSh.m==255,repSh.m==anchorColor)):
                continue
            for j in range(matrix.shape[1] - repSh.shape[1]+1):
                for i in range(matrix.shape[0] - repSh.shape[0]+1):
                    if np.all(np.logical_or(m[i:i+repSh.shape[0],j:j+repSh.shape[1]]==anchorColor,repSh.m==255))\
                                                    and np.all(seenM[i:i+repSh.shape[0],j:j+repSh.shape[1]]==0):
                        if perfectFit:
                            surrPixList = set([(i+p[0]+1, j+p[1]) for p in repSh.pixels]+[(i+p[0], j+p[1]+1) for p in repSh.pixels]\
                                               +[(i+p[0]-1, j+p[1]) for p in repSh.pixels]+[(i+p[0], j+p[1]-1) for p in repSh.pixels])
                            surrPixList = surrPixList - set([(i+p[0], j+p[1]) for p in repSh.pixels])
                            surrPixList = set([p for p in surrPixList if (p[0]>=0 and p[1]>=0 and p[0]<m.shape[0] and p[1]<m.shape[1])])
                            if len(set([m[p[0],p[1]] for p in surrPixList]))==1 and m[list(surrPixList)[0]]!=anchorColor: 
                                newInsert = copy.deepcopy(repSh)
                                newInsert.position = (i, j)
                                m = insertShape(m, newInsert)
                                for l, k in np.ndindex(repSh.shape):
                                    seenM[i+l,j+k]=1
                        else:
                            newInsert = copy.deepcopy(repSh)
                            newInsert.position = (i, j)
                            m = insertShape(m, newInsert)
                            for l, k in np.ndindex(repSh.shape):
                                seenM[i+l,j+k]=1
    elif anchorType == 'subframe':
        seenM = np.zeros(matrix.shape)
        if attributes != None:
            for repSh in delList:
                for i,j in np.ndindex(repSh.shape):
                    seenM[i+repSh.position[0],j+repSh.position[1]]=1
        for sh2 in shList:
            score, bestScore= 0, 0
            bestSh = None
            for repSh in repList:
                if adoptAnchorColor:
                    if sh2.isSubshape(repSh,sameColor=False,rotation=False,mirror=False) and len(sh2.pixels)<len(repSh.pixels):
                        for x in range((repSh.shape[0]-sh2.shape[0])+1):
                            for y in range((repSh.shape[1]-sh2.shape[1])+1):
                                mAux = m[max(sh2.position[0]-x, 0):min(sh2.position[0]-x+repSh.shape[0], m.shape[0]), max(sh2.position[1]-y, 0):min(sh2.position[1]-y+repSh.shape[1], m.shape[1])]
                                shAux = repSh.m[max(0, x-sh2.position[0]):min(repSh.shape[0],m.shape[0]+x-sh2.position[0]),max(0, y-sh2.position[1]):min(repSh.shape[1],m.shape[1]+y-sh2.position[1])]
                                seenAux = seenM[max(sh2.position[0]-x, 0):min(sh2.position[0]-x+repSh.shape[0], m.shape[0]), max(sh2.position[1]-y, 0):min(sh2.position[1]-y+repSh.shape[1], m.shape[1])]
                                if np.all(np.logical_or(shAux!=255, mAux == matrix.backgroundColor)) and np.all(seenAux==0):
                                    score = np.count_nonzero(mAux!=matrix.backgroundColor)
                                    if score > bestScore:
                                        bestScore = score
                                        bestX, bestY = sh2.position[0]-x, sh2.position[1]-y
                                        bestSh = copy.deepcopy(repSh)
                else:
                    if sh2.isSubshape(repSh,sameColor=True,rotation=False,mirror=False) and len(sh2.pixels)<len(repSh.pixels):
                        for x in range((repSh.shape[0]-sh2.shape[0])+1):
                            for y in range((repSh.shape[1]-sh2.shape[1])+1):
                                mAux = m[max(sh2.position[0]-x, 0):min(sh2.position[0]-x+repSh.shape[0], m.shape[0]), max(sh2.position[1]-y, 0):min(sh2.position[1]-y+repSh.shape[1], m.shape[1])]
                                shAux = repSh.m[max(0, x-sh2.position[0]):min(repSh.shape[0],m.shape[0]+x-sh2.position[0]),max(0, y-sh2.position[1]):min(repSh.shape[1],m.shape[1]+y-sh2.position[1])]
                                seenAux = seenM[max(sh2.position[0]-x, 0):min(sh2.position[0]-x+repSh.shape[0], m.shape[0]), max(sh2.position[1]-y, 0):min(sh2.position[1]-y+repSh.shape[1], m.shape[1])]
                                if np.all(np.logical_or(mAux==shAux, mAux == matrix.backgroundColor)) and np.all(seenAux==0):
                                    score = np.count_nonzero(mAux==shAux)
                                    if score > bestScore:
                                        bestScore = score
                                        bestX, bestY = sh2.position[0]-x, sh2.position[1]-y
                                        bestSh = copy.deepcopy(repSh)
            if bestSh != None:
                for i,j in np.ndindex(bestSh.shape):
                    if i+bestX>=0 and i+bestX<seenM.shape[0] and j+bestY>=0 and j+bestY<seenM.shape[1]:
                        seenM[i+bestX,j+bestY]=1
                newInsert = copy.deepcopy(bestSh)
                if adoptAnchorColor:
                    newInsert.m[newInsert.m!=255]=sh2.color
                newInsert.position = (bestX, bestY)
                newInsert.shape = newInsert.m.shape
                m = insertShape(m, newInsert)        
    if deleteOriginal:
        for sh in delList:
            m = deleteShape(m, sh, matrix.backgroundColor)
    return(m)

def getBestReplicateOneShape(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    if t.outSmallerThanIn or (not t.sameIOShapes):
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='pixelwise', reshape=True,\
                                                                multicolor=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='horizontal', reshape=True,\
                                                                multicolor=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='vertical', reshape=True,\
                                                                multicolor=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='pixelwise', reshape=True,\
                                                                multicolor=False, paintLikePix=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='horizontal', reshape=True,\
                                                                multicolor=False, paintLikePix=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='vertical', reshape=True,\
                                                                multicolor=False, paintLikePix=True), bestScore, bestFunction)
    deleteO = [False]
    deleteA = [False]
    if isDeleteTask(t):
        deleteO += [True]
        deleteA += [True]
    if t.sameIOShapes:
        bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, lay='pixelwise', multicolor=False,\
                                                                paintLikePix=True), bestScore, bestFunction)
        for deleteOriginal in deleteO:
            for deleteAnchor in deleteA:
                bestFunction, bestScore = updateBestFunction(t, partial(replicateOneShape, deleteOriginal=deleteOriginal,\
                                                                deleteAnchor=deleteAnchor), bestScore, bestFunction)
    return bestFunction

def replicateOneShape(matrix, diagonal=True, multicolor=True, deleteOriginal=False,\
                      deleteAnchor=False, lay=False, reshape=False,overlap=0, paintLikePix=False):
    m = matrix.m.copy()
    #first find the shape or shapes to replicate
    if diagonal:
        if multicolor:
            shList = [sh for sh in matrix.multicolorDShapes if len(sh.pixels)>1]
        else:
            shList = [sh for sh in matrix.dShapes if (len(sh.pixels)>1 and sh.color != matrix.backgroundColor)]
    else:
        if multicolor:
            shList = [sh for sh in matrix.multicolorShapes if len(sh.pixels)>1]
        else:
            shList = [sh for sh in matrix.shapes if (len(sh.pixels)>1 and sh.color != matrix.backgroundColor)]
    pixList = matrix.isolatedPixels#[pix for pix in matrix.dShapes if len(pix.pixels)==1]
    if len(shList) != 1 or len(pixList) == 0:
        return m
    repSh = shList[0]
    if lay != False:
        if lay == 'pixelwise':
            if len(pixList) < 2:
                return m
            if len(set([p.position[0] for p in pixList])) == 1:
                pixList.sort(key=lambda x: x.position[1])
                steps = [(0, pixList[i].position[1] - pixList[i-1].position[1] - 1) for i in range(1,len(pixList))]
                direction = (0, (pixList[1].position[1] - pixList[0].position[1] - 1)//abs(pixList[1].position[1] - pixList[0].position[1] - 1))
            elif len(set([p.position[1] for p in pixList])) == 1:
                pixList.sort(key=lambda x: x.position[0])
                steps = [(pixList[i].position[0] - pixList[i-1].position[0] - 1, 0) for i in range(1,len(pixList))]
                direction = ((pixList[1].position[0] - pixList[0].position[0] - 1)//abs(pixList[1].position[0] - pixList[0].position[0] - 1), 0)
            else:
                return m
            if paintLikePix and repSh.color == pixList[-1].color:
                    steps = steps[::-1]
                    steps = [(-st[0], -st[1]) for st in steps]
                    direction = (-direction[0], -direction[1])
                    pixList = pixList[::-1]
            if reshape:
                m = np.full((repSh.shape[0]*(1 + (len(pixList)-1)*abs(direction[0])), repSh.shape[1]*(1 + (len(pixList)-1)*abs(direction[1]))),\
                            fill_value = matrix.backgroundColor)
                for i in range(len(pixList)):
                    newInsert = copy.deepcopy(repSh)
                    newInsert.position = (i*repSh.shape[0]*direction[0], i*repSh.shape[1]*direction[1])
                    if paintLikePix:
                        newInsert.m[repSh.m == repSh.color] = pixList[i].color
                    m = insertShape(m, newInsert)
                deleteOriginal = False
            else:
                pos = repSh.position
                for (p,i) in zip(steps, [j for j in range(1,len(pixList))]):
                    pos = (pos[0] + direction[0]*repSh.shape[0] + p[0], pos[1] + direction[1]*repSh.shape[1] + p[1])
                    newInsert = copy.deepcopy(repSh)
                    newInsert.position = pos
                    if paintLikePix:
                        newInsert.m[repSh.m == repSh.color] = pixList[i].color
                    m = insertShape(m, newInsert)
        elif lay == 'horizontal': 
            m = np.full((repSh.shape[0], len(pixList)*repSh.shape[1]), fill_value = matrix.backgroundColor)
            deleteOriginal = False
            for i in range(len(pixList)):
                newInsert = copy.deepcopy(repSh)
                newInsert.position = (0, i*repSh.shape[1])
                m = insertShape(m, newInsert)
        elif lay == 'vertical': 
            m = np.full((len(pixList)*repSh.shape[0], repSh.shape[1]), fill_value = matrix.backgroundColor)
            deleteOriginal = False
            for i in range(len(pixList)):
                newInsert = copy.deepcopy(repSh)
                newInsert.position = (i*repSh.shape[0], 0)
                m = insertShape(m, newInsert)
    else:
        for pix in pixList:
            if (pix.position[0] >= repSh.position[0]) and (pix.position[1] >= repSh.position[1]) \
                and (pix.position[0] < repSh.position[0]+repSh.shape[0]) and (pix.position[1] < repSh.position[1]+repSh.shape[1]):
                continue
            newInsert = copy.deepcopy(repSh)
            if pix.m[0,0] in repSh.m:
                newInsert = copy.deepcopy(repSh)
                for i, j in np.ndindex(repSh.shape):
                    if repSh.m[i,j] == pix.m[0,0]:
                        newInsert.position = (pix.position[0]-i, pix.position[1]-j)
                        break
            else:
                newInsert.position = (pix.position[0] - (repSh.shape[0]-1)//2, pix.position[1] - (repSh.shape[1]-1)//2)
            m = insertShape(m, newInsert)
            if deleteAnchor:
                m = deleteShape(m, pix, matrix.backgroundColor)
    if deleteOriginal:
        m = deleteShape(m, repSh, matrix.backgroundColor)
    return m

def getBestMoveToPanel(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    for fit in [True, False]:
        for igPan in [True, False]:
            for uniq in [True, False]:
                bestFunction, bestScore = updateBestFunction(t, partial(moveToPanel,fit=fit,\
                                                ignorePanel=igPan), bestScore, bestFunction)
    return bestFunction

def moveToPanel(matrix, diagonal=True,fit=False, ignorePanel=False, cropPanel=True, uniq=True):
    m = matrix.m.copy()
    shList = [sh for sh in matrix.multicolorDShapes if len(sh.pixels)>1]
    if len(shList) < 2 or len(shList) > 8:
        return m
    shList.sort(key=lambda x: x.shape[0]*x.shape[1],reverse=True)
    panel = shList[0]
    shList = shList[1:]
    if fit and hasattr(panel, 'color'):
        pC = panel.color
        for sh in shList:
            found = False
            for x in range(4):
                rotSh = np.rot90(sh.m, x).copy()
                if panel.shape[0]-rotSh.shape[0]+1<0 or panel.shape[1]-rotSh.shape[1]+1<0:
                    continue
                for i, j in np.ndindex(panel.shape[0]-rotSh.shape[0]+1,panel.shape[1]-rotSh.shape[1]+1):
                    if np.all((rotSh==pC)==(panel.m[i:i+rotSh.shape[0],j:j+rotSh.shape[1]]==255)):
                        newInsert = copy.deepcopy(sh)
                        newInsert.m = rotSh
                        newInsert.shape = rotSh.shape
                        newInsert.position = (panel.position[0]+i, panel.position[1]+j)
                        m = insertShape(m, newInsert)
                        found = True
                        break
                if found:
                    break
    else:
        pixList = [pix for pix in matrix.dShapes if len(pix.pixels)==1]
        pixList = [pix for pix in pixList if all(pix.position[i]>=panel.position[i]\
                                        and pix.position[i]<panel.position[i]+panel.shape[i] for i in [0,1])]
        if len(pixList)==0:
            return m
        newInsList = []
        for pix in pixList:
            for sh in shList:
                if pix.m[0,0] in sh.m:
                    newInsert = copy.deepcopy(sh)
                    if sh.nColors == 1:
                        newInsert.position = (pix.position[0] - (sh.shape[0]-1)//2, pix.position[1] - (sh.shape[1]-1)//2)
                    else:
                        for i, j in np.ndindex(sh.shape):
                            if sh.m[i,j] == pix.m[0,0]:
                                newInsert.position = (pix.position[0]-i, pix.position[1]-j) 
                    newInsList.append(newInsert)
                    if uniq:
                        break
            if ignorePanel:
                m = deleteShape(m, panel, matrix.backgroundColor)
            for sh in newInsList:
                m = insertShape(m, sh)
    if cropPanel:
        return m[panel.position[0]:panel.position[0]+panel.shape[0],\
                 panel.position[1]:panel.position[1]+panel.shape[1]]
    return m

#overlapSubmatrices 
def printShapes(matrices, base=0, backgroundColor=0):
    """
    This function returns the result of printing one matrices on the other.
    The matrices are shape matrices and may contain 255.
    """
    if base == 1:
        matrices = matrices[::-1]
    m = np.zeros(matrices[0].shape, dtype=int)
    for i,j in np.ndindex(m.shape):
        if matrices[0][i,j] != 255 and matrices[0][i,j] != backgroundColor:
            if matrices[1][i,j] != 255:
                m[i,j] = matrices[1][i,j]
            else:
                m[i,j] = matrices[0][i,j]
        else:
            m[i,j] = backgroundColor
    return m

def multiplyMatrices(matrices, outShape=None, background=0, base=0, color=0):
    """
    Copy m1 matrix into every pixel of m2 if it is not background. Output has shape
    m1.shape*m2.shape. base and color are arguments to swap m1 and m2 and to choose
    color appropriately
    """
    if base == 1:
        matrices = matrices[::-1]
    m1 = matrices[0].copy()
    m2 = matrices[1].copy()
    s1 = m1.shape
    s2 = m2.shape
    m = np.full((m1.shape[0]*m2.shape[0],m1.shape[1]*m2.shape[1]), fill_value=background)
    if color == 0:
        for i,j in np.ndindex(s1):
            if m1[i,j] != background:
                m[i*s2[0]:(i+1)*s2[0], j*s2[1]:(j+1)*s2[1]] = m2
    else:
       for i,j in np.ndindex(s1):
            if m1[i,j] != background:
                for k,l in np.ndindex(m2.shape):
                    if m2[k,l] != background:
                        m[i*s2[0]+k, j*s2[1]+l] = m1[i,j] 
    return m

def overlapSubmatrices(matrix, colorHierarchy, shapeFactor=None):
    """
    This function returns the result of overlapping all submatrices of a given
    shape factor pixelswise with a given color hierarchy. Includes option to overlap
    all grid cells.     
    """
    if shapeFactor == None:
       submat = [t[0].m for t in matrix.grid.cellList]

    else:
        matrix = matrix.m
        sF = tuple(sin // sfact for sin, sfact in zip(matrix.shape, shapeFactor))
        submat = [matrix[sF[0]*i:sF[0]*(i+1),sF[1]*j:sF[1]*(j+1)] for i,j in np.ndindex(shapeFactor)]
    return overlapMatrices(submat, colorHierarchy)

def overlapMatrices(matrices, colorHierarchy):
    """
    Overlaps matrices of a given shape according to the color hierarchy
    """
    m = np.zeros(matrices[0].shape, dtype=np.uint8)
    for i,j in np.ndindex(m.shape):
        m[i,j] = colorHierarchy[max([colorHierarchy.index(x[i,j]) for x in matrices])]
    return m

def overlapShapes(matrix, diagonal=True, multicolor=True, byColor=False, hierarchy=[0,1,2,3,4,5,6,7,8,9]):
    if not multicolor: 
        if diagonal:   
            shList = [sh for sh in matrix.dShapes]
        else:   
            shList = [sh for sh in matrix.shapes]
    else:
        if diagonal: 
            shList = [sh for sh in matrix.multicolorDShapes]
        else:
            shList = [sh for sh in matrix.multicolorShapes]
    if byColor:
        shList = [sh for sh in matrix.shapesByColor]
    shList = [sh for sh in shList if sh.isRectangle]
    if len(set([sh.shape for sh in shList])) != 1:
        return matrix.m.copy()
    return overlapMatrices([sh.m for sh in shList],hierarchy)
#Cropshape
def getCropAttributes(t, diagonal, multicolor, sameColor=True):
    bC = max(0, t.backgroundColor)
    if diagonal and not multicolor:
        if t.nCommonInOutDShapes == 0:
            return set()
        attrs = set.intersection(*[s.inMatrix.getShapeAttributes(backgroundColor=bC,\
                    singleColor=True, diagonals=True)[s.inMatrix.dShapes.index(s.commonDShapes[0][0])] for s in t.trainSamples])
        nonAttrs = set()
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=True, diagonals=True)
            for shi in range(len(s.inMatrix.dShapes)):
                if s.inMatrix.dShapes[shi] == s.commonDShapes[0][0]:
                    continue
                else:
                    nonAttrs = nonAttrs.union(shAttrs[shi])
                    
    if not diagonal and not multicolor:
        if t.nCommonInOutShapes == 0:
            return set()
        attrs = set.intersection(*[s.inMatrix.getShapeAttributes(backgroundColor=bC,\
                    singleColor=True, diagonals=False)[s.inMatrix.shapes.index(s.commonShapes[0][0])] for s in t.trainSamples])
        nonAttrs = set()
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=True, diagonals=False)
            for shi in range(len(s.inMatrix.shapes)):
                if s.inMatrix.shapes[shi] == s.commonShapes[0][0]:
                    continue
                else:
                    nonAttrs = nonAttrs.union(shAttrs[shi]) 
                        
    if not diagonal and multicolor:
        if not t.outIsInMulticolorShapeSize:
            return set()                               
        attrs = set()
        nonAttrs = set()
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=False, diagonals=False)
            crop = False
            for shi in range(len(s.inMatrix.multicolorShapes)):
                if s.inMatrix.multicolorShapes[shi].shape == s.outMatrix.shape and\
                np.all(np.logical_or(s.inMatrix.multicolorShapes[shi].m == s.outMatrix.m, s.inMatrix.multicolorShapes[shi].m==255)):
                    crop = True
                    if len(attrs) == 0:
                        attrs = shAttrs[shi]
                    attrs = attrs.intersection(shAttrs[shi])
                else:
                    nonAttrs = nonAttrs.union(shAttrs[shi])    
        if not crop:
                return set()
    if diagonal and multicolor:
        if not t.outIsInMulticolorDShapeSize:
            return set()                               
        attrs = set()
        nonAttrs = set()
        for s in t.trainSamples:
            shAttrs = s.inMatrix.getShapeAttributes(backgroundColor=bC, singleColor=False, diagonals=True)
            crop = False
            for shi in range(len(s.inMatrix.multicolorDShapes)):
                if s.inMatrix.multicolorDShapes[shi].shape == s.outMatrix.shape and\
                np.all(np.logical_or(s.inMatrix.multicolorDShapes[shi].m == s.outMatrix.m, s.inMatrix.multicolorDShapes[shi].m==255)):
                    crop = True
                    if len(attrs) == 0:
                        attrs = shAttrs[shi]
                    attrs = attrs.intersection(shAttrs[shi])
                else:
                    nonAttrs = nonAttrs.union(shAttrs[shi])    
        if not crop:
                return set()
    return(attrs - nonAttrs)

def getBestCropShape(t):
    bestScore = 1000
    bestFunction = partial(identityM)
    bC = max(0, t.backgroundColor)
    bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=getCropAttributes(t,True, False),\
                                                           backgroundColor=bC, singleColor=True, diagonals=True), bestScore, bestFunction)
    if bestScore==0:
        return bestFunction
    bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=getCropAttributes(t,False, False),\
                                                           backgroundColor=bC, singleColor=True, diagonals=False), bestScore, bestFunction)
    if bestScore==0:
        return bestFunction
    bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=getCropAttributes(t,True, True),\
                                                           backgroundColor=bC, singleColor=False, diagonals=True), bestScore, bestFunction)
    if bestScore==0:
        return bestFunction
    bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=getCropAttributes(t,False, True),\
                                                           backgroundColor=bC, singleColor=False, diagonals=False), bestScore, bestFunction)
    if bestScore==0:
        return bestFunction
    for attr in ['LaSh', 'MoCo', 'MoCl', 'UnSh', 'UnSi']:
        bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=set([attr]),\
                                                           backgroundColor=bC, singleColor=True, diagonals=True), bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=set([attr]),\
                                                           backgroundColor=bC, singleColor=True, diagonals=False), bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=set([attr]),\
                                                           backgroundColor=bC, singleColor=False, diagonals=True), bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        bestFunction, bestScore = updateBestFunction(t, partial(cropShape, attributes=set([attr]),\
                                                           backgroundColor=bC, singleColor=True, diagonals=False), bestScore, bestFunction)
        if bestScore==0:
            return bestFunction
        
    return bestFunction
    
def cropShape(matrix, attributes, backgroundColor=0, singleColor=True, diagonals=True, context=False):
    """
    This function crops the shape out of a matrix with the maximum score according to attributes
    """
    if singleColor: 
        if diagonals:   
            shapeList = [sh for sh in matrix.dShapes]
        else:   
            shapeList = [sh for sh in matrix.shapes]
    else:
        if diagonals: 
            shapeList = [sh for sh in matrix.multicolorDShapes]
        else:
            shapeList = [sh for sh in matrix.multicolorShapes]
    bestShapes = []
    score = 0
    attrList = matrix.getShapeAttributes(backgroundColor, singleColor, diagonals)
    for i in range(len(shapeList)):
        shscore = len(attributes.intersection(attrList[i]))
        if shscore > score:
            score = shscore
            bestShapes = [i]
        elif shscore == score:
            bestShapes += [i]
    if len(bestShapes) == 0:
        return matrix
    if context:
        bestShape = shapeList[bestShapes[0]]
        m = matrix.m[bestShape.position[0]:bestShape.position[0]+bestShape.shape[0], bestShape.position[1]:bestShape.position[1]+bestShape.shape[1]]
        return m.copy() 
    else:
        bestShape = shapeList[bestShapes[0]].m.copy()
        bestShape[bestShape==255]=backgroundColor
    return bestShape

def getBestCropReference(t):
    bestFunction = partial(identityM)
    bestScore = 1000
    for sh in t.commonInShapes:
        bestFunction, bestScore = updateBestFunction(t, partial(cropShapeReference, refShape=sh,\
                                refType='subshape', multicolor=True, diagonal=False), bestScore, bestFunction)
    if len(t.commonInShapes) == 1:
        bestFunction, bestScore = updateBestFunction(t, partial(cropShapeReference, refShape=t.commonInShapes[0],\
                                refType='mark', multicolor=False, diagonal=True), bestScore, bestFunction)
        bestFunction, bestScore = updateBestFunction(t, partial(cropShapeReference, refShape=t.commonInShapes[0],\
                                refType='mark', multicolor=False, diagonal=False), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(cropShapeReference,\
                                refType='pixels', maxOrMin='min',multicolor=True, diagonal=False), bestScore, bestFunction)
    bestFunction, bestScore = updateBestFunction(t, partial(cropShapeReference,\
                                refType='pixels', maxOrMin='max',multicolor=True, diagonal=False), bestScore, bestFunction)
    #add referenced by frames
    return bestFunction

#Crop a shape using a reference shape or set of shapes
def cropShapeReference(matrix, refShape=None, refType='subshape', maxOrMin='max', sameColor=True, multicolor=True, diagonal=False):
    if multicolor:
        if diagonal:
            shList = matrix.multicolorDShapes
        else:
            shList = matrix.multicolorShapes
    else:
        if diagonal:
            shList = matrix.dShapes
        else:
            shList = matrix.shapes
    if refType == 'subshape':
        bestSh, bestScore = None, 0
        for sh in shList:
            if hasattr(sh, 'subshapes') and refShape in sh.subshapes:
                score = np.count_nonzero([refShape == sh2 for sh2 in sh.subshapes])
                if score > bestScore:
                    bestSh = sh
                    bestScore = score
        if bestSh == None:
            return matrix.m.copy()
        return bestSh.m
    elif refType == 'pixels':
        if maxOrMin == 'max':
            bestSh, bestScore = None, 0
        else:
            bestSh, bestScore = None, 1000
        for sh in shList:
            if hasattr(sh, 'subshapes'):
                score = len([p for p in sh.subshapes if len(p.pixels) == 1])
                if maxOrMin == 'max' and score > bestScore:
                    bestSh = sh
                    bestScore = score
                if maxOrMin == 'min' and score < bestScore:
                    bestSh = sh
                    bestScore = score
        if bestSh == None:
            return matrix.m.copy()
        return bestSh.m              
    elif refType == 'frame':
        return matrix.m.copy()
    elif refType == 'mark':
        foundRef = False
        for sh in shList:
            if sh == refShape:
                refShape = sh
                foundRef = True
                break
        if not foundRef:
            return matrix.m
        #otherwise return closest to reference
        bestShape = None
        dist = 1000
        refPos = refShape.position
        refPixels = [(p[0]+refPos[0], p[1]+refPos[1]) for p in refShape.pixels]
        for sh in shList:
            if sh == refShape or sh.color == matrix.backgroundColor:
                continue
            for p2 in [(p[0]+sh.position[0], p[1]+sh.position[1]) for p in sh.pixels]:
                if min(abs((p[0]-p2[0]))+abs((p[1]-p2[1])) for p in refPixels) < dist:
                    bestShape = sh
                    dist = min(abs((p[0]-p2[0])+(p[1]-p2[1])) for p in refPixels)
        if bestShape == None:
            return matrix.m.copy()
        bestShape=bestShape.m
        bestShape[bestShape==255]=matrix.backgroundColor
        return bestShape         

def cropAllBackground(matrix):
    m = matrix.m.copy()
    bC = matrix.backgroundColor
    if np.all(m == bC):
        return m
    x1, x2, y1, y2 = 0, m.shape[0]-1, 0, m.shape[1]-1
    while x1 <= x2 and np.all(m[x1,:] == bC):
        x1 += 1
    while x2 >= x1 and np.all(m[x2,:] == bC):
        x2 -= 1
    while y1 <= y2 and np.all(m[:,y1] == bC):
        y1 += 1
    while y2 >= y1 and np.all(m[:,y2] == bC):
        y2 -= 1
    return(m[x1:x2+1,y1:y2+1])
    
def cropOnlyMulticolorShape(matrix, diagonals=False):
    """
    This function is supposed to be called if there is one and only one 
    multicolor shape in all the input samples. This function just returns it.
    """
    if diagonals:
        m = matrix.multicolorDShapes[0].m.copy()
        m[m==255] = matrix.multicolorDShapes[0].background
    else:
        m = matrix.multicolorShapes[0].m.copy()
        m[m==255] = matrix.multicolorShapes[0].background
    return m

def cropFullFrame(matrix, includeBorder=True, bigOrSmall = None):
    m = matrix.m.copy()
    if bigOrSmall == None and len(matrix.fullFrames) != 1:
        return m
    if bigOrSmall == "small":
        frame = matrix.fullFrames[-1]
    else:
        frame = matrix.fullFrames[0]
    if includeBorder:
        return m[frame.position[0]:frame.position[0]+frame.shape[0], \
                 frame.position[1]:frame.position[1]+frame.shape[1]]
    else:
        return m[frame.position[0]+1:frame.position[0]+frame.shape[0]-1, \
                 frame.position[1]+1:frame.position[1]+frame.shape[1]-1]
        
def cropPartialFrame(matrix, includeBorder=True, bigOrSmall = None):
    m = matrix.m.copy()
    if len(matrix.partialFrames) == 0:
        return m
    if bigOrSmall == "small":
        frame = matrix.partialFrames[-1]
    else:
        frame = matrix.partialFrames[0]
    if includeBorder:
        return m[frame.position[0]:frame.position[0]+frame.shape[0], \
                 frame.position[1]:frame.position[1]+frame.shape[1]]
    else:
        return m[frame.position[0]+1:frame.position[0]+frame.shape[0]-1, \
                 frame.position[1]+1:frame.position[1]+frame.shape[1]-1]

def getBestTwoShapeFunction(t):
    """
    Inputs a task with two input shapes and tries a series of operations that make sense.
    """
    bestScore = 1000
    bestFunction = partial(identityM)
    multicolor = t.twoShapeTask[1]
    diagonal = t.twoShapeTask[2]
    typ = t.twoShapeTask[-1]
    cropAfter = [0] 
    if t.outSmallerThanIn:
        cropAfter += [1,2]
    #try possible operations
    for crop in cropAfter:
        for flip in [True,False]:
            #this confirms that the shapes have the same size. 
            if t.twoShapeTask[3]:
                #pixelwise and/or
                for c in permutations(t.totalOutColors,2):
                    bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(pixelwiseAnd, falseColor=c[0],\
                                targetColor=None,trueColor=c[1]), diagonal=diagonal,typ=typ, multicolor=multicolor, crop=crop, flip=flip), bestScore, bestFunction)
                    bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(pixelwiseOr, falseColor=c[0],\
                                targetColor=None,trueColor=c[1]), diagonal=diagonal,typ=typ, multicolor=multicolor, crop=crop, flip=flip), bestScore, bestFunction)
                for base in [0,1]:
                    for bC in t.commonInColors:
                        bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(printShapes, base=base,\
                                backgroundColor=bC),diagonal=diagonal,typ=typ, multicolor=multicolor, crop=crop, flip=flip), bestScore, bestFunction)
                if typ > 1:
                    bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(overlapMatrices,\
                                colorHierarchy=[0,1,2,3,4,5,6,7,8,9]),diagonal=diagonal,typ=typ, multicolor=multicolor, crop=crop, flip=flip), bestScore, bestFunction)
            else:
                for c in permutations(t.totalOutColors,2):
                    for target in [None]:
                        bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(pixelwiseAnd, falseColor=c[0],\
                                    targetColor=target,trueColor=c[1]), diagonal=diagonal, multicolor=multicolor,typ=typ, crop=crop, flip=flip, downsizeToSmallest=True), bestScore, bestFunction)
                        bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(pixelwiseOr, falseColor=c[0],\
                                    targetColor=target,trueColor=c[1]), diagonal=diagonal, multicolor=multicolor,typ=typ, crop=crop, flip=flip, downsizeToSmallest=True), bestScore, bestFunction)
                        bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(pixelwiseAnd, falseColor=c[0],\
                                    targetColor=target,trueColor=c[1]), diagonal=diagonal, multicolor=multicolor,typ=typ, crop=crop, flip=flip, scaleToLargest=True), bestScore, bestFunction)
                        bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(pixelwiseOr, falseColor=c[0],\
                                    targetColor=target,trueColor=c[1]), diagonal=diagonal, multicolor=multicolor,typ=typ, crop=crop, flip=flip, scaleToLargest=True), bestScore, bestFunction)  
                for base in [0,1]:
                    for bC in t.commonInColors:
                        bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(printShapes, base=base,\
                                backgroundColor=bC),diagonal=diagonal, multicolor=multicolor, crop=crop,typ=typ, flip=flip, downsizeToSmallest=True), bestScore, bestFunction)
                        bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(printShapes, base=base,\
                                backgroundColor=bC),diagonal=diagonal, multicolor=multicolor, crop=crop,typ=typ, flip=flip, scaleToLargest=True), bestScore, bestFunction)
            #multiply matrices
            for c in [0,1]:
                for b in [0,1]:
                    bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(multiplyMatrices, base=b,\
                                background=0, color=c), typ=typ, crop=crop, flip=flip), bestScore, bestFunction)
                    bestFunction, bestScore = updateBestFunction(t, partial(twoShapeFun, f=partial(multiplyMatrices, base=b,\
                                background=0, color=c), typ=typ, crop=crop, flip=flip, downsizeToSmallest=True), bestScore, bestFunction)
               
    return bestFunction

def twoShapeFun(matrix, f=partial(identityM), typ=1, diagonal=True, multicolor=True,\
                flip=False, rotate=False, downsizeToSmallest=False, scaleToLargest=False, crop=False):
    """
    Apply function f to the shapes of matrix. By default, this function should only be called when matrix has two shapes. 
    """
    def cropAllBackgroundM(m, background):
        """
        Same as above but for a np matrix and specified background color
        """
        if np.all(m == background):
            return m
        x1, x2, y1, y2 = 0, m.shape[0]-1, 0, m.shape[1]-1
        while x1 <= x2 and np.all(m[x1,:] == background):
            x1 += 1
        while x2 >= x1 and np.all(m[x2,:] == background):
            x2 -= 1
        while y1 <= y2 and np.all(m[:,y1] == background):
            y1 += 1
        while y2 >= y1 and np.all(m[:,y2] == background):
            y2 -= 1
        return(m[x1:x2+1,y1:y2+1])
    def minimizeM(m):
        x = 1
        for i in range(1, m.shape[0]):
            if np.array_equal(m[x,:],m[x-1,:]):
                m = np.delete(m, (x), axis=0)
            else:
                x+=1
        x = 1
        for i in range(1, m.shape[1]):
            if np.array_equal(m[:,x],m[:,x-1]):
                m = np.delete(m, (x), axis=1)
            else:
                x+=1
        return m
    def scaleM(m, s):
        sc = min(s[0]//m.shape[0], s[1]//m.shape[1])
        sm = np.repeat(np.repeat(m, sc, axis=1), sc, axis=0)
        return sm

    m = matrix.m.copy()
    if typ == 1:
        if multicolor:
            if diagonal: 
                shList = [sh for sh in matrix.multicolorDShapes]
            else:
                shList = [sh for sh in matrix.multicolorShapes]
        else: 
            if diagonal:   
                shList = [sh for sh in matrix.dShapes]
            else:   
                shList = [sh for sh in matrix.shapes]
        posList = [sh.position for sh in shList]
        mList = [sh.m.copy() for sh in shList]
    elif typ == 2:
        if not hasattr(matrix,'grid'):
            return m
        mList = [c[0].m for c in matrix.grid.cellList]
        posList = [c[1] for c in matrix.grid.cellList]
    elif typ == 3:
        if matrix.shape[0] == 2*matrix.shape[1]:
            mList = [matrix.m[:matrix.shape[0]//2,:].copy(), matrix.m[matrix.shape[0]//2:,:].copy()]
            posList = [(0,0),(matrix.shape[1],0)]
        elif matrix.shape[1] == 2*matrix.shape[0]:
            mList = [matrix.m[:,:matrix.shape[1]//2].copy(), matrix.m[:,matrix.shape[1]//2:].copy()]
            posList = [(0,0),(0,matrix.shape[0])]
        else:
            return m
        mList.sort(key=lambda x: len(np.unique(x))) 
    if len(mList)  != 2:
        return m    
    if flip:
        mList[1] = np.fliplr(mList[1])
    #sort list largest first
    if mList[0].shape[0]*mList[0].shape[1]<mList[1].shape[0]*mList[1].shape[1]:
        mList = mList[::-1]
        posList = posList[::-1]
    if downsizeToSmallest:
        mList[0] = minimizeM(mList[0])
    if scaleToLargest:
        mList[1] = scaleM(mList[1], mList[0].shape)
    if mList[0].shape != mList[1].shape and f.func.__name__ != 'multiplyMatrices':
        return m
    if  f.func.__name__ == 'multiplyMatrices':
        return f(mList)
    elif crop == 1:
        return(f(mList))
    elif crop == 2:
        return cropAllBackgroundM(f(mList),matrix.backgroundColor)
    else:
        if scaleToLargest:
            pos = posList[0]
            sh = mList[0]
            m[pos[0]:pos[0] + sh.shape[0], pos[1]:pos[1] + sh.shape[1]] = f(mList)                
        elif downsizeToSmallest:
            pos = posList[1]
            sh = mList[1]
            m[pos[0]:pos[0] + sh.shape[0], pos[1]:pos[1] + sh.shape[1]] = f(mList)   
        return m

# %% Main function: getPossibleOperations
def getPossibleOperations(t, c):
    """
    Given a Task.Task t and a Candidate c, this function returns a list of all
    the possible operations that make sense applying to the input matrices of
    c.
    The elements of the list to be returned are partial functions, whose input
    is a Task.Matrix and whose output is a numpy.ndarray (2-dim matrix).
    """ 
    candTask = c.t
    # print("dir: ", dir(candTask))
    x = [] # List to be returned
    ###########################################################################
    # Fill the blanks
    if t.fillTheBlank:
        params = fillTheBlankParameters(t)
        x.append(partial(fillTheBlank, params=params))
        
    # downsize
    if candTask.sameOutShape:
        outShape = candTask.outShape
        x.append(partial(downsize, newShape=outShape))
        x.append(partial(downsizeMode,newShape=outShape))
        if t.backgroundColor!=-1:
            x.append(partial(downsize, newShape=outShape, falseColor=t.backgroundColor))
        if candTask.sameOutDummyMatrix and candTask.backgroundColor != -1:
            x.append(partial(arrangeShapes,outDummyMatrix=candTask.trainSamples[0].outMatrix.dummyMatrix,\
                             outDummyColor=candTask.trainSamples[0].outMatrix.backgroundColor))
    x.append(getBestCountColors(candTask))  
    x.append(getBestCountShapes(candTask))  
    ###########################################################################
    # sameIOShapes
    if candTask.sameIOShapes:
    
        #######################################################################
        
        # ColorMap
        ncc = len(candTask.colorChanges)
        if len(set([cc[0] for cc in candTask.colorChanges])) == ncc and ncc != 0:
            x.append(partial(colorMap, cMap=dict(candTask.colorChanges)))
            
        # Symmetrize
        if all([len(x)==1 for x in candTask.changedInColors]):
            color = next(iter(candTask.changedInColors[0]))
            axis = []
            if candTask.lrSymmetric:
                axis.append("lr")
            if candTask.udSymmetric:
                axis.append("ud")
            if all([sample.inMatrix.shape[0]==sample.inMatrix.shape[1] for sample in candTask.testSamples]):              
                if candTask.d1Symmetric: 
                    axis.append("d1")
                if candTask.d2Symmetric:
                    axis.append("d2")
            x.append(partial(symmetrize, axis=axis, color=color))
            if candTask.totalOutColors==1:
                for fc in candTask.fixedColors:
                    x.append(partial(symmetrize, axis=axis, refColor=fc,\
                                     outColor=next(iter(candTask.totalOutColors))))
    
        # Color symmetric pixels
        x.append(getBestColorSymmetricPixels(candTask))
        
        # Complete rectangles
        if candTask.backgroundColor!=-1 and len(candTask.fixedColors)==1 and \
        len(candTask.colorChanges)==1:
            sc = next(iter(candTask.fixedColors))
            nc = next(iter(candTask.colorChanges))[1]
            x.append(partial(completeRectangles, sourceColor=sc, newColor=nc))
        
        x.append(partial(deletePixels, diagonals=True))
        x.append(partial(deletePixels, diagonals=False))
        
        #######################################################################
        # For LinearShapeModel we need to have the same shapes in the input
        # and in the output, and in the exact same positions.
        # This model predicts the color of the shape in the output.
        if candTask.onlyShapeColorChanges:
            ccwf = getColorChangesWithFeatures(candTask)
            fsf = candTask.fixedShapeFeatures
            x.append(partial(changeShapesWithFeatures, ccwf=ccwf, fixedColors=candTask.fixedColors,\
                             fixedShapeFeatures=fsf))
                
            if all(["getBestLSTM" not in str(op.func) for op in c.ops]):        
                # x.append(getBestLSTM(candTask))
                pass
            
            # Other deterministic functions that change the color of shapes.
            for cc in candTask.commonColorChanges:
                for border, bs in product([True, False, None], ["big", "small", None]):
                    x.append(partial(changeShapes, inColor=cc[0], outColor=cc[1],\
                                     bigOrSmall=bs, isBorder=border))
            x.append(getBestReplicateShapes(candTask))
            x.append(getBestColorByPixels(candTask))
            return x

        #######################################################################
        # Complete row/col patterns
        colStep=None
        rowStep=None
        if candTask.followsRowPattern:
            if candTask.allEqual(candTask.rowPatterns):
                rowStep=candTask.rowPatterns[0]
        if candTask.followsColPattern:
            if candTask.allEqual(candTask.colPatterns):
                colStep=candTask.colPatterns[0]
        if candTask.allEqual(candTask.changedInColors) and len(candTask.changedInColors[0])==1:
            c2c = next(iter(candTask.changedInColors[0]))
        else:
            c2c=None
                
        if candTask.followsRowPattern and candTask.followsColPattern:
            x.append(partial(followPattern, rc="both", colorToChange=c2c,\
                             rowStep=rowStep, colStep=colStep))
        elif candTask.followsRowPattern:
            x.append(partial(followPattern, rc="row", colorToChange=c2c,\
                             rowStep=rowStep, colStep=colStep))
        elif candTask.followsColPattern:
            x.append(partial(followPattern, rc="col", colorToChange=c2c,\
                             rowStep=rowStep, colStep=colStep))

        #######################################################################
        # CNNs
        
        # if candTask.sameNSampleColors and all(["predictCNN" not in str(op.func) for op in c.ops]):
        #     x.append(getBestSameNSampleColorsCNN(candTask))

        """
        if t.backgroundColor != -1:
            model = trainCNNDummyColor(candTask, 5, -1)
            x.append(partial(predictCNNDummyColor, model=model))
            model = trainCNNDummyColor(candTask, 3, 0)
            x.append(partial(predictCNNDummyColor, model=model))
            #model = trainOneConvModelDummyColor(candTask, 7, -1)
            #x.append(partial(predictConvModelDummyColor, model=model))
        """
            
        #cc = list(t.commonSampleColors)
        #model = trainCNNDummyCommonColors(t, cc, 3, -1)
        #x.append(partial(predictCNNDummyCommonColors, model=model,\
        #                commonColors=cc))
        
        #######################################################################
        # Transformations if the color count is always the same:
        # Rotations, Mirroring, Move Shapes, Mirror Shapes, ...
        if candTask.sameColorCount:
            for axis in ["lr", "ud"]:
                x.append(partial(mirror, axis = axis))
            # You can only mirror d1/d2 or rotate if the matrix is squared.
            if candTask.inMatricesSquared:
                for axis in ["d1", "d2"]:
                    x.append(partial(mirror, axis = axis))
                for angle in [90, 180, 270]:
                    x.append(partial(rotate, angle = angle))
                
            # Mirror shapes
            x.append(getBestFlipAllShapes(candTask))
            
        #######################################################################
        # Other sameIOShapes functions
        # Move shapes
        x.append(getBestMoveShapes(candTask, c))
        
        pr = pixelRecolor(candTask)
        if len(pr)!=1:
            x.append(partial(executePixelRecolor, Best_Dict=pr[0], Best_v=pr[1], Best_Q1=pr[2], Best_Q2=pr[3]))
        
        fun = getPixelChangeCriteria(candTask)
        if fun != 0:
            x.append(fun)
            
        # extendColor
        x.append(getBestExtendColor(candTask))
        
        # surround shapes
        x.append(getBestSurroundShapes(candTask))
        
        # Paint shapes in half
        x.append(getBestPaintShapesInHalf(candTask))
        
        # Paint shapes from border color
        x.append(getBestPaintShapeFromBorderColor(candTask))
            
        # fillRectangleInside
        for cic in candTask.commonChangedInColors:
            for coc in candTask.commonChangedOutColors:
                x.append(partial(fillRectangleInside, rectangleColor=cic, fillColor=coc))
        
        # Color longest lines
        if len(candTask.colorChanges)==1:
            change = next(iter(candTask.colorChanges))
            x.append(partial(colorLongestLines, cic=change[0], coc=change[1], direction='h'))
            x.append(partial(colorLongestLines, cic=change[0], coc=change[1], direction='v'))
            x.append(partial(colorLongestLines, cic=change[0], coc=change[1], direction='hv'))
            x.append(partial(colorLongestLines, cic=change[0], coc=change[1], direction='d'))

        # Connect Pixels
        x.append(partial(connectAnyPixels))
        x.append(partial(connectAnyPixels, diagonal=True))
        if all([len(x)==1 for x in candTask.changedInColors]):
            x.append(partial(connectAnyPixels, connColor=next(iter(candTask.changedOutColors[0]))))
            x.append(partial(connectAnyPixels, connColor=next(iter(candTask.changedOutColors[0])), \
                             diagonal=True))

        fc = candTask.fixedColors
        #if hasattr(t, "fixedColors"):
        #    tfc = candTask.fixedColors
        #else:
        #    tfc = set()
        for pc in candTask.colors - candTask.commonChangedInColors:
            for cc in candTask.commonChangedOutColors:
                x.append(partial(connectAnyPixels, pixelColor=pc, \
                                 connColor=cc, fixedColors=fc))
                x.append(partial(connectAnyPixels, pixelColor=pc, \
                                 connColor=cc, fixedColors=fc, diagonal=True))
        for pc in candTask.colors - candTask.commonChangedInColors:
            x.append(partial(connectAnyPixels, pixelColor=pc, allowedChanges=dict(candTask.colorChanges)))
            x.append(partial(connectAnyPixels, pixelColor=pc, allowedChanges=dict(candTask.colorChanges), \
                             diagonal=True))
            x.append(partial(connectAnyPixels, pixelColor=pc, allowedChanges=dict(candTask.colorChanges),\
                             lineExclusive=True))
            x.append(partial(connectAnyPixels, pixelColor=pc, allowedChanges=dict(candTask.colorChanges),\
                             lineExclusive=True, diagonal=True))
                
        for cc in candTask.commonColorChanges:
            for cc in candTask.commonColorChanges:
                for border, bs in product([True, False, None], ["big", "small", None]):
                    x.append(partial(changeShapes, inColor=cc[0], outColor=cc[1],\
                                     bigOrSmall=bs, isBorder=border))
        
        #Test for tasks 436,261
        #if len(set.union(*[set(sh.shape for sh in s.outMatrix.shapes) for s in candTask.trainSamples])) == 1:
        if all(len(set([sh.shape for sh in s.outMatrix.shapes]))==1 for s in candTask.trainSamples):
            x.append(partial(subMatToLayer,shapeAndDict=getLayerDict(candTask)))
        #replicate/symmterize/other shape related tasks
        
        x.append(getBestAlignShapes(candTask))
        x.append(getBestSymmetrizeSubmatrix(candTask))
        x.append(getBestReplicateShapes(candTask))
        x.append(getBestSymmetrizeAllShapes(candTask))
        x.append(getBestColorByPixels(candTask))
        #delete shapes
        if isDeleteTask(candTask) and all(t.backgroundColor == c[1] for c in candTask.colorChanges):
            x.append(getBestDeleteShapes(candTask, True, True))
        
        # TODO
        """
        if all([len(s.inMatrix.multicolorShapes)==1 for s in candTask.trainSamples+candTask.testSamples]) and\
        all([len(s.outMatrix.multicolorShapes)==1 for s in candTask.testSamples]):
            if all([s.outMatrix.multicolorShapes[0].isRotationInvariant() for s in candTask.trainSamples]):
                for color in candTask.commonChangedOutColors:
                    x.append(makeShapeRotationInvariant, color=color)
        """
                    
    ###########################################################################
    # Cases in which the input has always the same shape, and the output too
    if candTask.sameInShape and candTask.sameOutShape and \
    all(candTask.trainSamples[0].inMatrix.shape == s.inMatrix.shape for s in candTask.testSamples):
        """
        if candTask.backgroundColor != -1:
            model = trainLinearDummyModel(candTask)
            x.append(partial(predictLinearDummyModel, model=model, \
                             outShape=candTask.outShape,\
                             backgroundColor=candTask.backgroundColor))
        
        if candTask.sameNSampleColors:
            cc = list(candTask.commonSampleColors)
            nc = candTask.trainSamples[0].nColors
            model = trainLinearModel(candTask, cc, nc)
            x.append(partial(predictLinearModel, model=model, commonColors=cc,\
                             nChannels=nc, outShape=candTask.outShape))
        """
            
        if candTask.sameNSampleColors:
            #Cases where output colors are a subset of input colors
            if candTask.sameNInColors and all(s.inHasOutColors for s in candTask.trainSamples):
                if hasattr(candTask, 'outShapeFactor') or (hasattr(candTask,\
                              'gridCellIsOutputShape') and candTask.gridCellIsOutputShape):
                    ch = dict(sum([Counter(s.outMatrix.colorCount) for s in candTask.trainSamples],Counter()))
                    ch = sorted(ch, key=ch.get)
                    if candTask.backgroundColor in ch:
                        ch.remove(candTask.backgroundColor)
                    ch = list(set([0,1,2,3,4,5,6,7,8,9]).difference(set(ch))) + ch
                    if hasattr(candTask, 'outShapeFactor'):
                        x.append(partial(overlapSubmatrices, colorHierarchy=ch, shapeFactor=candTask.outShapeFactor))
                    else:
                        x.append(partial(overlapSubmatrices, colorHierarchy=ch))
        #pixelMap = Models.pixelCorrespondence(candTask)
        #if len(pixelMap) != 0:
        #    x.append(partial(mapPixels, pixelMap=pixelMap, outShape=candTask.outShape))
    
    ###########################################################################
    # Evolve
    #if candTask.sameIOShapes and all([len(x)==1 for x in candTask.changedInColors]) and\
    #len(candTask.commonChangedInColors)==1 and candTask.sameNSampleColors:
        #cfn = evolve(candTask)
        #x.append(getBestEvolve(candTask, cfn))
        #x.append(partial(applyEvolve, cfn=cfn, nColors=candTask.trainSamples[0].nColors,\
        #                 kernel=5, nIterations=1, fixedColors=candTask.fixedColors,\
        #                 changedInColors=candTask.commonChangedInColors,\
        #                 changedOutColors=candTask.commonChangedOutColors, \
        #                 referenceIsFixed=True))
    
    #if candTask.sameIOShapes and all([len(x)==1 for x in candTask.changedInColors]) and\
    #len(candTask.commonChangedInColors)==1:

    if candTask.sameIOShapes and len(candTask.commonChangedInColors)==1:   
        x.append(getBestEvolvingLines(candTask))

    ###########################################################################
    # Other cases
    
    if candTask.sameIOShapes and len(candTask.fixedColors)!=0:
        for color in candTask.fixedColors:
            for outColor in candTask.commonOutColors:
                x.append(partial(paintCrossedCoordinates, refColor=color,\
                                 outColor=outColor, fixedColors=candTask.fixedColors))
    
    if candTask.inSmallerThanOut and t.inSmallerThanOut:
        x.append(getBestExtendMatrix(candTask))

    if hasattr(candTask, 'inShapeFactor'):
        x.append(partial(multiplyPixels, factor=candTask.inShapeFactor))
        x.append(partial(multiplyMatrix, factor=candTask.inShapeFactor))
        
        for c in candTask.commonSampleColors:
            x.append(partial(multiplyPixelsAndAnd, factor=candTask.inShapeFactor,\
                             falseColor=c))
            x.append(partial(multiplyPixelsAndOr, factor=candTask.inShapeFactor,\
                             falseColor=c))
            x.append(partial(multiplyPixelsAndXor, factor=candTask.inShapeFactor,\
                             falseColor=c))
            
        if type(candTask.inShapeFactor)==tuple:
            ops = getBestMosaic(candTask)
            x.append(partial(generateMosaic, ops=ops, factor=candTask.inShapeFactor))
            
        if all([s.inMatrix.shape[0]**2 == s.outMatrix.shape[0] and \
                s.inMatrix.shape[1]**2 == s.outMatrix.shape[1] for s in candTask.trainSamples]):
            totalColorCount = Counter()
            for sample in t.trainSamples:
                for color in sample.outMatrix.colorCount.keys():
                    totalColorCount[color] += sample.outMatrix.colorCount[color]
            falseColor = max(totalColorCount.items(), key=operator.itemgetter(1))[0]
            opCond = getBestMultiplyMatrix(candTask, falseColor)
            x.append(partial(doBestMultiplyMatrix, opCond=opCond, falseColor=falseColor))
            
            
    if hasattr(candTask, 'outShapeFactor'):
        if outputIsSubmatrix(candTask):
            for color in range(10):
                x.append(partial(selectSubmatrixWithMaxColor, color=color, outShapeFactor=candTask.outShapeFactor))
                x.append(partial(selectSubmatrixWithMinColor, color=color, outShapeFactor=candTask.outShapeFactor))
            x.append(partial(selectSubmatrixWithMostColors, outShapeFactor=candTask.outShapeFactor))
            x.append(partial(selectSubmatrixWithLeastColors, outShapeFactor=candTask.outShapeFactor))
            position = getBestSubmatrixPosition(candTask, outShapeFactor=candTask.outShapeFactor)
            x.append(partial(selectSubmatrixInPosition, position=position, outShapeFactor=candTask.outShapeFactor))
        
        # Pixelwise And
        for c in candTask.commonOutColors:
            x.append(partial(pixelwiseAndInSubmatrices, factor=candTask.outShapeFactor,\
                             falseColor=c))
        if len(candTask.totalOutColors) == 2:
            for target in candTask.totalInColors:
                for c in permutations(candTask.totalOutColors, 2):
                    x.append(partial(pixelwiseAndInSubmatrices, \
                                     factor=candTask.outShapeFactor, falseColor=c[0],\
                                     targetColor=target, trueColor=c[1]))
        
        # Pixelwise Or
        for c in candTask.commonOutColors:
            x.append(partial(pixelwiseOrInSubmatrices, factor=candTask.outShapeFactor,\
                             falseColor=c))
        if len(candTask.totalOutColors) == 2:
            for target in candTask.totalInColors:
                for c in permutations(candTask.totalOutColors, 2):
                    x.append(partial(pixelwiseOrInSubmatrices, \
                                     factor=candTask.outShapeFactor, falseColor=c[0],\
                                     targetColor=target, trueColor=c[1]))
        if candTask.backgroundColor!=-1:
            colors = candTask.commonOutColors - candTask.commonInColors
            if candTask.outShapeFactor[0]*candTask.outShapeFactor[1]==len(colors):
                for c in permutations(colors, len(colors)):
                    x.append(partial(pixelwiseOrInSubmatrices, factor=candTask.outShapeFactor,\
                                     falseColor=candTask.backgroundColor,\
                                     trueValues = c))
        
        # Pixelwise Xor
        if candTask.outShapeFactor in [(2,1), (1,2)]:
            for c in candTask.commonOutColors:
                x.append(partial(pixelwiseXorInSubmatrices, factor=candTask.outShapeFactor,\
                                 falseColor=c))
            if len(candTask.commonOutColors - candTask.commonInColors)==2 and\
            candTask.backgroundColor!=-1:
                colors = candTask.commonOutColors - candTask.commonInColors
                for c in permutations(colors, 2):
                    x.append(partial(pixelwiseXorInSubmatrices, falseColor=candTask.backgroundColor,\
                                     firstTrue=c[0], secondTrue=c[1]))
            if len(candTask.totalOutColors) == 2:
                for target in candTask.totalInColors:
                    for c in permutations(candTask.totalOutColors, 2):
                        x.append(partial(pixelwiseXorInSubmatrices, \
                                         factor=candTask.outShapeFactor, falseColor=c[0],\
                                         targetColor=target, trueColor=c[1]))
    
    if hasattr(candTask, 'gridCellIsOutputShape') and candTask.gridCellIsOutputShape:
        if outputIsSubmatrix(candTask, isGrid=True):
            for color in range(10):
                x.append(partial(selectSubmatrixWithMaxColor, color=color, isGrid=True))
                x.append(partial(selectSubmatrixWithMinColor, color=color, isGrid=True))
            x.append(partial(selectSubmatrixWithMostColors, isGrid=True))
            x.append(partial(selectSubmatrixWithLeastColors, isGrid=True))
            position = getBestSubmatrixPosition(candTask, isGrid=True)
            x.append(partial(selectSubmatrixInPosition, position=position, isGrid=True))
        
        # Pixelwise And
        for c in candTask.commonOutColors:
            x.append(partial(pixelwiseAndInGridSubmatrices, falseColor=c))
        if len(candTask.totalOutColors) == 2:
            for target in candTask.totalInColors:
                for c in permutations(candTask.totalOutColors, 2):
                    x.append(partial(pixelwiseAndInGridSubmatrices, falseColor=c[0],\
                                     targetColor=target, trueColor=c[1]))
                        
        # Pixelwise Or
        for c in candTask.commonOutColors:
            x.append(partial(pixelwiseOrInGridSubmatrices, falseColor=c))
        if len(candTask.totalOutColors) == 2:
            for target in candTask.totalInColors:
                for c in permutations(candTask.totalOutColors, 2):
                    x.append(partial(pixelwiseOrInGridSubmatrices, falseColor=c[0],\
                                     targetColor=target, trueColor=c[1]))
        if candTask.backgroundColor!=-1:
            colors = candTask.commonOutColors - candTask.commonInColors
            if candTask.trainSamples[0].inMatrix.grid.nCells==len(colors):
                for c in permutations(colors, len(colors)):
                    x.append(partial(pixelwiseOrInGridSubmatrices,\
                                     falseColor=candTask.backgroundColor,\
                                     trueValues = c))
        
        # Pixelwise Xor
        if all([s.inMatrix.grid.nCells == 2 for s in candTask.trainSamples]) \
        and all([s.inMatrix.grid.nCells == 2 for s in candTask.testSamples]):
            for c in candTask.commonOutColors:
                x.append(partial(pixelwiseXorInGridSubmatrices, falseColor=c))
            if len(candTask.commonOutColors - candTask.commonInColors)==2 and\
            candTask.backgroundColor!=-1:
                colors = candTask.commonOutColors - candTask.commonInColors
                for c in permutations(colors, 2):
                    x.append(partial(pixelwiseXorInGridSubmatrices, falseColor=candTask.backgroundColor,\
                                     firstTrue=c[0], secondTrue=c[1]))
            if len(candTask.totalOutColors) == 2:
                for target in candTask.totalInColors:
                    for c in permutations(candTask.totalOutColors, 2):
                        x.append(partial(pixelwiseXorInGridSubmatrices, falseColor=c[0],\
                                         targetColor=target, trueColor=c[1]))
    
    # 733
    # if candTask.hasUnchangedGrid and candTask.sameColorCount:
    #    x.append(getBestOrderGridCells)
    
    # 757
    #if candTask.hasUnchangedAsymmetricGrid and candTask.outAsymmetricGridCellsHaveOneColor:
    # paintGridCellWithMostColor    
    
    if candTask.inputIsGrid:
        if all([s.inMatrix.grid.shape==s.outMatrix.shape for s in candTask.trainSamples]):
            for times in range(1, 6):
                x.append(partial(colorAppearingXTimes, times=times))
            x.append(partial(maxColorFromCell))
    
    x.append(getBestLayShapes(candTask))
    x.append(getBestReplicateOneShape(candTask))
    #tasks with two shapes
    if candTask.twoShapeTask[0]:
        x.append(getBestTwoShapeFunction(t))
    # Cropshape    
    if candTask.outSmallerThanIn:
        x.append(getBestAlignShapes(candTask))
        x.append(getBestArrangeShapes(candTask))    
        x.append(partial(replicateShapes, allCombs=True, scale=False,attributes=set(['MoCl']),anchorType='subframe',deleteOriginal=True))
        x.append(partial(replicateShapes, allCombs=False, scale=True,attributes=set(['MoCl']),anchorType='subframe',deleteOriginal=True))
        x.append(partial(colorByPixels, deletePixels=True))
        x.append(partial(colorByPixels, colorMap=True, deletePixels=True))
        x.append(partial(deleteShapes, attributes = getDeleteAttributes(candTask, diagonal = False), diagonal = False, multicolor=False))
        x.append(partial(overlapShapes))
        if candTask.backgroundColor!=-1:              
            x.append(partial(cropAllShapes, background=candTask.backgroundColor, diagonal=True))
            x.append(partial(cropAllShapes, background=candTask.backgroundColor, diagonal=False))
           
        bestCrop = getBestCropShape(candTask)
        if 'attributes' in bestCrop.keywords.keys():
            for attr in bestCrop.keywords['attributes']:
                if isinstance(attr, str) and attr[:2] == 'mo' and len(bestCrop.keywords['attributes']) > 1:
                    continue
                newCrop = copy.deepcopy(bestCrop)
                newCrop.keywords['attributes'] = set([attr])
                x.append(newCrop)
        x.append(getBestCropReference(candTask))           
        
        for attrs in [set(['LaSh']),set(['UnCo'])]:
            x.append(partial(cropShape, attributes=attrs, backgroundColor=max(0,candTask.backgroundColor),\
                                 singleColor=True, diagonals=True)) 
            x.append(partial(cropShape, attributes=attrs, backgroundColor=max(0,candTask.backgroundColor),\
                                 singleColor=True, diagonals=True, context=True)) 
        if candTask.outIsInMulticolorShapeSize:
            x.append(getBestMoveToPanel(candTask))
    if all([len(s.inMatrix.multicolorShapes)==1 for s in candTask.trainSamples+candTask.testSamples]):
        x.append(partial(cropOnlyMulticolorShape, diagonals=False))
    if all([len(s.inMatrix.multicolorDShapes)==1 for s in candTask.trainSamples+candTask.testSamples]):
        x.append(partial(cropOnlyMulticolorShape, diagonals=True))
    if all([len(sample.inMatrix.fullFrames)==1 for sample in candTask.trainSamples+candTask.testSamples]):
        x.append(partial(cropFullFrame))
        x.append(partial(cropFullFrame, includeBorder=False))
    if all([len(sample.inMatrix.fullFrames)>1 for sample in candTask.trainSamples+candTask.testSamples]):
        x.append(partial(cropFullFrame, bigOrSmall="big"))
        x.append(partial(cropFullFrame, bigOrSmall="small"))
        x.append(partial(cropFullFrame, bigOrSmall="big", includeBorder=False))
        x.append(partial(cropFullFrame, bigOrSmall="small", includeBorder=False))
    #more frames
    if candTask.hasPartialFrame:
        x.append(getBestFitToFrame(candTask))
        if candTask.outSmallerThanIn:
            x.append(partial(cropPartialFrame, includeBorder=False))
            x.append(partial(cropPartialFrame, includeBorder=True))
    # startOps
    x.append(partial(paintGridLikeBackground))
    x.append(partial(cropAllBackground)) 
    
    # switchColors
    if all([n==2 for n in candTask.nInColors]):
        x.append(partial(switchColors))
    
    # minimize
    if not candTask.sameIOShapes:
        x.append(partial(minimize))
    
    return x
