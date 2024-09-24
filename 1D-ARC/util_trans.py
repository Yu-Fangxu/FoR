
import numpy as np
import copy
from typing import NamedTuple, Callable, Any
import json
import numpy as np
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors


import Utils
import Task
def plot_pictures(pictures, labels):
    fig, axs = plt.subplots(1, len(pictures), figsize=(2*len(pictures),32))
    for i, (pict, label) in enumerate(zip(pictures, labels)):
        axs[i].imshow(np.array(pict), cmap=cmap, norm=norm)
        axs[i].set_title(label)
    plt.show()

def plot_sample(sample, predict=None):
    """
    This function plots a sample. sample is an object of the class Task.Sample.
    predict is any matrix (numpy ndarray).
    """
    if predict is None:
        plot_pictures([sample.inMatrix.m, sample.outMatrix.m], ['Input', 'Output'])
    else:
        plot_pictures([sample.inMatrix.m, sample.outMatrix.m, predict], ['Input', 'Output', 'Predict'])

def plot_task(task):
    """
    Given a task (in its original format), this function plots all of its
    matrices.
    """
    cmap = colors.ListedColormap(
            ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
            '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    if type(task)==int:
        task = allTasks[index[task]]
    len_train = len(task['train'])
    len_test  = len(task['test'])
    len_max   = max(len_train, len_test)
    length    = {'train': len_train, 'test': len_test}
    fig, axs  = plt.subplots(len_max, 4, figsize=(15, 15*len_max//4))
    for col, mode in enumerate(['train', 'test']):
        for idx in range(length[mode]):
            axs[idx][2*col+0].axis('off')
            axs[idx][2*col+0].imshow(task[mode][idx]['input'], cmap=cmap, norm=norm)
            axs[idx][2*col+0].set_title(f"Input {mode}, {np.array(task[mode][idx]['input']).shape}")
            try:
                axs[idx][2*col+1].axis('off')
                axs[idx][2*col+1].imshow(task[mode][idx]['output'], cmap=cmap, norm=norm)
                axs[idx][2*col+1].set_title(f"Output {mode}, {np.array(task[mode][idx]['output']).shape}")
            except:
                pass
        for idx in range(length[mode], len_max):
            axs[idx][2*col+0].axis('off')
            axs[idx][2*col+1].axis('off')
    plt.tight_layout()
    plt.axis('off')
    plt.savefig("intermediate")

# For formatting the output
def flattener(pred):
    """
    This function formats the output. Only to be used when submitting to Kaggle
    """
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred

# %% Current approach
class Candidate():
    """
    Objects of the class Candidate store the information about a possible
    candidate for the solution.

    ...
    Attributes
    ----------
    ops: list
        A list containing the operations to be performed to the input matrix
        in order to get to the solution. The elements of the list are partial
        functions (from functools.partial).
    score: int
        The score of the candidate. The score is defined as the sum of the
        number incorrect pixels when applying ops to the input matrices of the
        train samples of the task.
    tasks: list
        A list containing the tasks (in its original format) after performing
        each of the operations in ops, starting from the original inputs.
    t: Task.Task
        The Task.Task object corresponding to the current status of the task.
        This is, the status after applying all the operations of ops to the
        input matrices of the task.
    """
    def __init__(self, ops, tasks, score=1000, predictions=np.zeros((2,2))):
        self.ops = ops
        self.score = score
        self.tasks = tasks
        self.t = None
        self.predictions = predictions

    def __lt__(self, other):
        """
        A candidate is better than another one if its score is lower.
        """
        if self.score == other.score:
            return len(self.ops) < len(other.ops)
        return self.score < other.score

    def generateTask(self):
        """
        Assign to the attribute t the Task.Task object corresponding to the
        current task status.
        """
        self.t = Task.Task(self.tasks[-1], 'dummyIndex', submission=False)

class Best1Candidates():
    """
    An object of this class stores the three best candidates of a task.

    ...
    Attributes
    ----------
    candidates: list
        A list of three elements, each one of them being an object of the class
        Candidate.
    """
    def __init__(self, Candidate1):
        self.candidates = [Candidate1]

    def maxCandidate(self):
        """
        Returns the index of the candidate with highest score.
        """
        x = 0
        return x

    def addCandidate(self, c):
        """
        Given a candidate c, this function substitutes c with the worst
        candidate in self.candidates only if it's a better candidate (its score
        is lower).
        """
        if all([self.candidates[i].score < c.score for i in range(1)]):
            return
        
        for i in range(1):
            if all([np.array_equal(self.candidates[i].predictions[x], c.predictions[x]) \
                    for x in range(len(c.predictions))]):
                return
        iMaxCand = self.maxCandidate()
        for i in range(1):
            if c < self.candidates[iMaxCand]:
                c.generateTask()
                self.candidates[iMaxCand] = c
                break

    def allPerfect(self):
        return all([c.score==0 for c in self.candidates])

    def getOrderedIndices(self):
        """
        Returns a list of 3 indices (from 0 to 2) with the candidates ordered
        from best to worst.
        """
        orderedList = [0]
        return orderedList


# Separate task by shapes
class TaskSeparatedByShapes():
    def __init__(self, task, background, diagonal=False):
        self.originalTask = task
        self.separatedTask = None
        self.nShapes = {'train': [], 'test': []}
        self.background = background

    def getRange(self, trainOrTest, index):
        i, position = 0, 0
        while i < index:
            position += self.nShapes[trainOrTest][i]
            i += 1
        return (position, position+self.nShapes[trainOrTest][index])


def needsSeparationByShapes(t):
    def getOverlap(inShape, inPos, outShape, outPos):
        x1a, y1a, x1b, y1b = inPos[0], inPos[1], outPos[0], outPos[1]
        x2a, y2a = inPos[0]+inShape[0]-1, inPos[1]+inShape[1]-1
        x2b, y2b = outPos[0]+outShape[0]-1, outPos[1]+outShape[1]-1
        if x1a<=x1b:
            if x2a<=x1b:
                return 0
            x = x2a-x1b+1
        elif x1b<=x1a:
            if x2b<=x1a:
                return 0
            x = x2b-x1a+1
        if y1a<=y1b:
            if y2a<=y1b:
                return 0
            y = y2a-y1b+1
        elif y1b<=y1a:
            if y2b<=y1a:
                return 0
            y = y2b-y1a+1

        return x*y

    def generateNewTask(inShapes, outShapes, testShapes):
        # Assign every input shape to the output shape with maximum overlap
        separatedTask = TaskSeparatedByShapes(t.task.copy(), t.backgroundColor)
        task = {'train': [], 'test': []}
        for s in range(t.nTrain):
            seenIndices = set()
            for inShape in inShapes[s]:
                shapeIndex = 0
                maxOverlap = 0
                bestIndex = -1
                for outShape in outShapes[s]:
                    overlap = getOverlap(inShape.shape, inShape.position, outShape.shape, outShape.position)
                    if overlap > maxOverlap:
                        maxOverlap = overlap
                        bestIndex = shapeIndex
                    shapeIndex += 1
                if bestIndex!=-1 and bestIndex not in seenIndices:
                    seenIndices.add(bestIndex)
                    # Generate the new input and output matrices
                    inM = np.full(t.trainSamples[s].inMatrix.shape, t.backgroundColor ,dtype=np.uint8)
                    outM = inM.copy()
                    inM = Utils.insertShape(inM, inShape)
                    outM = Utils.insertShape(outM, outShapes[s][bestIndex])
                    task['train'].append({'input': inM.tolist(), 'output': outM.tolist()})
            # If we haven't dealt with all the shapes successfully, then return
            if len(seenIndices) != len(inShapes[s]):
                return False
            # Record the number of new samples generated by sample s
            separatedTask.nShapes['train'].append(len(inShapes[s]))
        for s in range(t.nTest):
            for testShape in testShapes[s]:
                inM = np.full(t.testSamples[s].inMatrix.shape, t.backgroundColor ,dtype=np.uint8)
                inM = Utils.insertShape(inM, testShape)
                if t.submission:
                    task['test'].append({'input': inM.tolist()})
                else:
                    task['test'].append({'input': inM.tolist(), 'output': t.testSamples[s].outMatrix.m.tolist()})
            # Record the number of new samples generated by sample s
            separatedTask.nShapes['test'].append(len(testShapes[s]))


        # Complete and return the TaskSeparatedByShapes object
        separatedTask.separatedTask = task.copy()
        return separatedTask


    # I need to have a background color to generate the new task object
    if t.backgroundColor==-1 or not t.sameIOShapes:
        return False
    # Only consider tasks without small matrices
    if any([s.inMatrix.shape[0]*s.inMatrix.shape[1]<43 for s in t.trainSamples+t.testSamples]):
        return False

    # First, consider normal shapes (not background, not diagonal, not multicolor) (Task 84 as example)
    inShapes = [[shape for shape in s.inMatrix.shapes if shape.color!=t.backgroundColor] for s in t.trainSamples]
    outShapes = [[shape for shape in s.outMatrix.shapes if shape.color!=t.backgroundColor] for s in t.trainSamples]
    testShapes = [[shape for shape in s.inMatrix.shapes if shape.color!=t.backgroundColor] for s in t.testSamples]
    if all([len(inShapes[s])<=7 and len(inShapes[s])==len(outShapes[s]) for s in range(t.nTrain)]):
        newTask = generateNewTask(inShapes, outShapes, testShapes)
        if newTask != False:
            return newTask

    # Now, consider diagonal shapes (Task 681 as example)
    inShapes = [[shape for shape in s.inMatrix.dShapes if shape.color!=t.backgroundColor] for s in t.trainSamples]
    outShapes = [[shape for shape in s.outMatrix.dShapes if shape.color!=t.backgroundColor] for s in t.trainSamples]
    testShapes = [[shape for shape in s.inMatrix.dShapes if shape.color!=t.backgroundColor] for s in t.testSamples]
    if all([len(inShapes[s])<=5 and len(inShapes[s])==len(outShapes[s]) for s in range(t.nTrain)]):
        newTask = generateNewTask(inShapes, outShapes, testShapes)
        if newTask != False:
            return newTask

    # Now, multicolor non-diagonal shapes (Task 611 as example)
    inShapes = [[shape for shape in s.inMatrix.multicolorShapes] for s in t.trainSamples]
    outShapes = [[shape for shape in s.outMatrix.multicolorShapes] for s in t.trainSamples]
    testShapes = [[shape for shape in s.inMatrix.multicolorShapes] for s in t.testSamples]
    if all([len(inShapes[s])<=7 and len(inShapes[s])==len(outShapes[s]) for s in range(t.nTrain)]):
        newTask = generateNewTask(inShapes, outShapes, testShapes)
        if newTask != False:
            return newTask

    # Finally, multicolor diagonal (Task 610 as example)
    inShapes = [[shape for shape in s.inMatrix.multicolorDShapes] for s in t.trainSamples]
    outShapes = [[shape for shape in s.outMatrix.multicolorDShapes] for s in t.trainSamples]
    testShapes = [[shape for shape in s.inMatrix.multicolorDShapes] for s in t.testSamples]
    if all([len(inShapes[s])<=5 and len(inShapes[s])==len(outShapes[s]) for s in range(t.nTrain)]):
        newTask = generateNewTask(inShapes, outShapes, testShapes)
        if newTask != False:
            return newTask

    return False

# Separate task by colors
class TaskSeparatedByColors():
    def __init__(self, task):
        self.originalTask = task
        self.separatedTask = None
        self.commonColors = None
        self.extraColors = {'train': [], 'test': []}

    def getRange(self, trainOrTest, index):
        i, position = 0, 0
        while i < index:
            position += len(self.extraColors[trainOrTest][i])
            i += 1
        return (position, position+len(self.extraColors[trainOrTest][index]))


def needsSeparationByColors(t):
    def generateMatrix(matrix, colorsToKeep, backgroundColor):
        m = matrix.copy()
        for i,j in np.ndindex(matrix.shape):
            if m[i,j] not in colorsToKeep:
                m[i,j] = backgroundColor

        return m

    def generateNewTask(commonColors, backgroundColor):
        # Assign every input shape to the output shape with maximum overlap
        separatedTask = TaskSeparatedByColors(t.task.copy())
        task = {'train': [], 'test': []}
        for s in range(t.nTrain):
            separatedTask.extraColors['train'].append([])
            colorsToConsider = (t.trainSamples[s].inMatrix.colors | t.trainSamples[s].outMatrix.colors)\
                                - commonColors
            if len(colorsToConsider)==0:
                return False
            for color in colorsToConsider:
                separatedTask.extraColors['train'][s].append(color)
                inM = generateMatrix(t.trainSamples[s].inMatrix.m, commonColors|set([color]), backgroundColor)
                outM = generateMatrix(t.trainSamples[s].outMatrix.m, commonColors|set([color]), backgroundColor)
                task['train'].append({'input': inM.tolist(), 'output': outM.tolist()})

        for s in range(t.nTest):
            separatedTask.extraColors['test'].append([])
            if t.submission:
                colorsToConsider = t.testSamples[s].inMatrix.colors - commonColors
                if len(colorsToConsider)==0:
                    return False
                for color in colorsToConsider:
                    separatedTask.extraColors['test'][s].append(color)
                    inM = generateMatrix(t.testSamples[s].inMatrix.m, commonColors|set([color]), backgroundColor)
                    task['test'].append({'input': inM.tolist()})
            else:
                colorsToConsider = (t.testSamples[s].inMatrix.colors | t.testSamples[s].outMatrix.colors)\
                                    - commonColors
                if len(colorsToConsider)==0:
                    return False
                for color in colorsToConsider:
                    separatedTask.extraColors['test'][s].append(color)
                    inM = generateMatrix(t.testSamples[s].inMatrix.m, commonColors|set([color]), backgroundColor)
                    outM = generateMatrix(t.testSamples[s].outMatrix.m, commonColors|set([color]), backgroundColor)
                    task['test'].append({'input': inM.tolist(), 'output': t.testSamples[s].outMatrix.m.tolist()})

        # Complete and return the TaskSeparatedByShapes object
        separatedTask.separatedTask = task.copy()
        return separatedTask


    # I need to have a background color to generate the new task object
    if t.backgroundColor==-1 or not t.sameIOShapes:
        return False
    # Only consider tasks without small matrices
    if any([s.inMatrix.shape[0]*s.inMatrix.shape[1]<50 for s in t.trainSamples+t.testSamples]):
        return False

    commonColors = t.commonInColors | t.commonOutColors

    if all([sample.nColors == len(commonColors) for sample in t.trainSamples]):
        return False
    if any([sample.nColors < len(commonColors) for sample in t.trainSamples]):
        return False

    newTask = generateNewTask(commonColors, t.backgroundColor)

    return newTask

# Crop task if necessary

def getCroppingPosition(matrix):
    bC = matrix.backgroundColor
    x, xMax, y, yMax = 0, matrix.m.shape[0]-1, 0, matrix.m.shape[1]-1
    while x <= xMax and np.all(matrix.m[x,:] == bC):
        x += 1
    while y <= yMax and np.all(matrix.m[:,y] == bC):
        y += 1
    return [x,y]

def needsCropping(t):
    # Only to be used if t.sameIOShapes
    for sample in t.trainSamples:
        if sample.inMatrix.backgroundColor != sample.outMatrix.backgroundColor:
            return False
        if getCroppingPosition(sample.inMatrix) != getCroppingPosition(sample.outMatrix):
            return False
        inMatrix = Utils.cropAllBackground(sample.inMatrix)
        outMatrix = Utils.cropAllBackground(sample.outMatrix)
        if inMatrix.shape!=outMatrix.shape or sample.inMatrix.shape==inMatrix.shape:
            return False
    return True

def cropTask(t, task):
    positions = {"train": [], "test": []}
    backgrounds = {"train": [], "test": []}
    for s in range(t.nTrain):
        task["train"][s]["input"] = Utils.cropAllBackground(t.trainSamples[s].inMatrix).tolist()
        task["train"][s]["output"] = Utils.cropAllBackground(t.trainSamples[s].outMatrix).tolist()
        backgrounds["train"].append(t.trainSamples[s].inMatrix.backgroundColor)
        positions["train"].append(getCroppingPosition(t.trainSamples[s].inMatrix))
    for s in range(t.nTest):
        task["test"][s]["input"] = Utils.cropAllBackground(t.testSamples[s].inMatrix).tolist()
        backgrounds["test"].append(t.testSamples[s].inMatrix.backgroundColor)
        positions["test"].append(getCroppingPosition(t.testSamples[s].inMatrix))
        if not t.submission:
            task["test"][s]["output"] = Utils.cropAllBackground(t.testSamples[s].outMatrix).tolist()
    return positions, backgrounds

def recoverCroppedMatrix(matrix, outShape, position, backgroundColor):
    m = np.full(outShape, backgroundColor, dtype=np.uint8)
    m[position[0]:position[0]+matrix.shape[0], position[1]:position[1]+matrix.shape[1]] = matrix.copy()
    return m

def needsRecoloring(t):
    """
    This method determines whether the task t needs recoloring or not.
    It needs recoloring if every color in an output matrix appears either
    in the input or in every output matrix.
    Otherwise a recoloring doesn't make sense.
    If this function returns True, then orderTaskColors should be executed
    as the first part of the preprocessing of t.
    """
    for sample in t.trainSamples:
        for color in sample.outMatrix.colors:
            if (color not in sample.inMatrix.colors) and (color not in t.commonOutColors):
                return False
    return True

def orderTaskColors(t):
    """
    Given a task t, this function generates a new task (as a dictionary) by
    recoloring all the matrices in a specific way.
    The goal of this function is to impose that if two different colors
    represent the exact same thing in two different samples, then they have the
    same color in both of the samples.
    Right now, the criterium to order colors is:
        1. Common colors ordered according to Task.Task.orderColors
        2. Colors that appear both in the input and the output
        3. Colors that only appear in the input
        4. Colors that only appear in the output
    In steps 2-4, if there is more that one color satisfying that condition,
    the ordering will happen according to the colorCount.
    """
    def orderColors(trainOrTest):
        if trainOrTest=="train":
            samples = t.trainSamples
        else:
            samples = t.testSamples
        for sample in samples:
            sampleColors = t.orderedColors.copy()
            sortedColors = [k for k, v in sorted(sample.inMatrix.colorCount.items(), key=lambda item: item[1])]
            for c in sortedColors:
                if c not in sampleColors:
                    sampleColors.append(c)
            if trainOrTest=="train" or t.submission==False:
                sortedColors = [k for k, v in sorted(sample.outMatrix.colorCount.items(), key=lambda item: item[1])]
                for c in sortedColors:
                    if c not in sampleColors:
                        sampleColors.append(c)

            rel, invRel = Utils.relDicts(sampleColors)
            if trainOrTest=="train":
                trainRels.append(rel)
                trainInvRels.append(invRel)
            else:
                testRels.append(rel)
                testInvRels.append(invRel)

            inMatrix = np.zeros(sample.inMatrix.shape, dtype=np.uint8)
            for c in sample.inMatrix.colors:
                inMatrix[sample.inMatrix.m==c] = invRel[c]
            if trainOrTest=='train' or t.submission==False:
                outMatrix = np.zeros(sample.outMatrix.shape, dtype=np.uint8)
                for c in sample.outMatrix.colors:
                    outMatrix[sample.outMatrix.m==c] = invRel[c]
                if trainOrTest=='train':
                    task['train'].append({'input': inMatrix.tolist(), 'output': outMatrix.tolist()})
                else:
                    task['test'].append({'input': inMatrix.tolist(), 'output': outMatrix.tolist()})
            else:
                task['test'].append({'input': inMatrix.tolist()})

    task = {'train': [], 'test': []}
    trainRels = []
    trainInvRels = []
    testRels = []
    testInvRels = []

    orderColors("train")
    orderColors("test")

    return task, trainRels, trainInvRels, testRels, testInvRels

def recoverOriginalColors(matrix, rel):
    """
    Given a matrix, this function is intended to recover the original colors
    before being modified in the orderTaskColors function.
    rel is supposed to be either one of the trainRels or testRels outputs of
    that function.
    """
    m = matrix.copy()
    for i,j in np.ndindex(matrix.shape):
        if matrix[i,j] in rel.keys(): # TODO Task 162 fails. Delete this when fixed
            m[i,j] = rel[matrix[i,j]][0]
    return m

def hasRepeatedOutputs(t):
    nonRepeated = []
    for i in range(t.nTrain):
        seen = False
        for j in range(i+1, t.nTrain):
            if np.array_equal(t.trainSamples[i].outMatrix.m, t.trainSamples[j].outMatrix.m):
                seen = True
        if not seen:
            nonRepeated.append(t.trainSamples[i].outMatrix.m.copy())
    if len(nonRepeated)==t.nTrain:
        return False, []
    else:
        return True, nonRepeated

def ignoreGrid(t, task, inMatrix=True, outMatrix=True):
    for s in range(t.nTrain):
        if inMatrix:
            m = np.zeros(t.trainSamples[s].inMatrix.grid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.trainSamples[s].inMatrix.grid.cells[i][j][0].colors))
            task["train"][s]["input"] = m.tolist()
        if outMatrix:
            m = np.zeros(t.trainSamples[s].outMatrix.grid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.trainSamples[s].outMatrix.grid.cells[i][j][0].colors))
            task["train"][s]["output"] = m.tolist()
    for s in range(t.nTest):
        if inMatrix:
            m = np.zeros(t.testSamples[s].inMatrix.grid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.testSamples[s].inMatrix.grid.cells[i][j][0].colors))
            task["test"][s]["input"] = m.tolist()
        if outMatrix and not t.submission:
            m = np.zeros(t.testSamples[s].outMatrix.grid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.testSamples[s].outMatrix.grid.cells[i][j][0].colors))
            task["test"][s]["output"] = m.tolist()

def recoverGrid(t, x, s):
    realX = t.testSamples[s].inMatrix.m.copy()
    cells = t.testSamples[s].inMatrix.grid.cells
    for cellI in range(len(cells)):
        for cellJ in range(len(cells[0])):
            cellShape = cells[cellI][cellJ][0].shape
            position = cells[cellI][cellJ][1]
            for k,l in np.ndindex(cellShape):
                realX[position[0]+k, position[1]+l] = x[cellI,cellJ]
    return realX

def ignoreAsymmetricGrid(t, task):
    for s in range(t.nTrain):
        m = np.zeros(t.trainSamples[s].inMatrix.asymmetricGrid.shape, dtype=np.uint8)
        for i,j in np.ndindex(m.shape):
            m[i,j] = next(iter(t.trainSamples[s].inMatrix.asymmetricGrid.cells[i][j][0].colors))
        task["train"][s]["input"] = m.tolist()
        m = np.zeros(t.trainSamples[s].outMatrix.asymmetricGrid.shape, dtype=np.uint8)
        for i,j in np.ndindex(m.shape):
            m[i,j] = next(iter(t.trainSamples[s].outMatrix.asymmetricGrid.cells[i][j][0].colors))
        task["train"][s]["output"] = m.tolist()
    for s in range(t.nTest):
        m = np.zeros(t.testSamples[s].inMatrix.asymmetricGrid.shape, dtype=np.uint8)
        for i,j in np.ndindex(m.shape):
            m[i,j] = next(iter(t.testSamples[s].inMatrix.asymmetricGrid.cells[i][j][0].colors))
        task["test"][s]["input"] = m.tolist()
        if not t.submission:
            m = np.zeros(t.testSamples[s].outMatrix.asymmetricGrid.shape, dtype=np.uint8)
            for i,j in np.ndindex(m.shape):
                m[i,j] = next(iter(t.testSamples[s].outMatrix.asymmetricGrid.cells[i][j][0].colors))
            task["test"][s]["output"] = m.tolist()

def recoverAsymmetricGrid(t, x, s):
    realX = t.testSamples[s].inMatrix.m.copy()
    cells = t.testSamples[s].inMatrix.asymmetricGrid.cells
    for cellI in range(len(cells)):
        for cellJ in range(len(cells[0])):
            cellShape = cells[cellI][cellJ][0].shape
            position = cells[cellI][cellJ][1]
            for k,l in np.ndindex(cellShape):
                realX[position[0]+k, position[1]+l] = x[cellI,cellJ]
    return realX

def ignoreGeneralGrid(t, task, inMatrix=True, outMatrix=True):
    for s in range(t.nTrain):
        if inMatrix:
            cellShape = t.trainSamples[s].inMatrix.grid.cellShape
            m = np.zeros((t.trainSamples[s].inMatrix.grid.shape[0]*cellShape[0],\
                          t.trainSamples[s].inMatrix.grid.shape[1]*cellShape[1]),dtype=np.uint8)
            for i,j in np.ndindex(t.trainSamples[s].inMatrix.grid.shape):
                m[i*cellShape[0]:(i+1)*cellShape[0],j*cellShape[1]:(j+1)*cellShape[1]] = t.trainSamples[s].inMatrix.grid.cells[i][j][0].m
            task["train"][s]["input"] = m.tolist()
        if outMatrix:
            cellShape = t.trainSamples[s].outMatrix.grid.cellShape
            m = np.zeros((t.trainSamples[s].outMatrix.grid.shape[0]*cellShape[0],\
                          t.trainSamples[s].outMatrix.grid.shape[1]*cellShape[1]),dtype=np.uint8)
            for i,j in np.ndindex(t.trainSamples[s].outMatrix.grid.shape):
                m[i*cellShape[0]:(i+1)*cellShape[0],j*cellShape[1]:(j+1)*cellShape[1]] = t.trainSamples[s].outMatrix.grid.cells[i][j][0].m
            task["train"][s]["output"] = m.tolist()
    for s in range(t.nTest):
        if inMatrix:
            cellShape = t.trainSamples[s].inMatrix.grid.cellShape
            m = np.zeros((t.trainSamples[s].inMatrix.grid.shape[0]*cellShape[0],\
                          t.trainSamples[s].inMatrix.grid.shape[1]*cellShape[1]),dtype=np.uint8)
            for i,j in np.ndindex(t.trainSamples[s].inMatrix.grid.shape):
                m[i*cellShape[0]:(i+1)*cellShape[0],j*cellShape[1]:(j+1)*cellShape[1]] = t.trainSamples[s].inMatrix.grid.cells[i][j][0].m
            task["test"][s]["input"] = m.tolist()
        if outMatrix:
            cellShape = t.trainSamples[s].outMatrix.grid.cellShape
            m = np.zeros((t.trainSamples[s].outMatrix.grid.shape[0]*cellShape[0],\
                          t.trainSamples[s].outMatrix.grid.shape[1]*cellShape[1]),dtype=np.uint8)
            for i,j in np.ndindex(t.trainSamples[s].outMatrix.grid.shape):
                m[i*cellShape[0]:(i+1)*cellShape[0],j*cellShape[1]:(j+1)*cellShape[1]] = t.trainSamples[s].outMatrix.grid.cells[i][j][0].m
            task["test"][s]["output"] = m.tolist()

def ignoreGeneralAsymmetricGrid(t, task):
    for s in range(t.nTrain):
        cellList = [cell for cell in t.trainSamples[s].inMatrix.asymmetricGrid.cells]
        gridShape = t.trainSamples[0].inMatrix.asymmetricGrid.shape
        newShape = (sum(cellList[i][0][0].shape[0] for i in range(gridShape[0])),sum(cellList[0][i][0].shape[1] for i in range(gridShape[1])))
        m = np.zeros(newShape, dtype=np.uint8)
        currX = 0
        for i in range(gridShape[0]):
            currY = 0
            for j in range(gridShape[1]):
                m[currX:currX + cellList[i][j][0].shape[0],currY:currY + cellList[i][j][0].shape[1]] = cellList[i][j][0].m
                currY += cellList[i][j][0].shape[1]
            currX += cellList[i][j][0].shape[0]
        task["train"][s]["input"] = m.tolist()

        cellList = [cell for cell in t.trainSamples[s].outMatrix.asymmetricGrid.cells]
        gridShape = t.trainSamples[0].outMatrix.asymmetricGrid.shape
        newShape = (sum(cellList[i][0][0].shape[0] for i in range(gridShape[0])),sum(cellList[0][i][0].shape[1] for i in range(gridShape[1])))
        m = np.zeros(newShape, dtype=np.uint8)
        currX = 0
        for i in range(gridShape[0]):
            currY = 0
            for j in range(gridShape[1]):
                m[currX:currX + cellList[i][j][0].shape[0],currY:currY + cellList[i][j][0].shape[1]] = cellList[i][j][0].m
                currY += cellList[i][j][0].shape[1]
            currX += cellList[i][j][0].shape[0]
        task["train"][s]["output"] = m.tolist()

    for s in range(t.nTest):
        cellList = [cell for cell in t.testSamples[s].inMatrix.asymmetricGrid.cells]
        gridShape = t.testSamples[0].inMatrix.asymmetricGrid.shape
        newShape = (sum(cellList[i][0][0].shape[0] for i in range(gridShape[0])),sum(cellList[0][i][0].shape[1] for i in range(gridShape[1])))
        m = np.zeros(newShape, dtype=np.uint8)
        currX = 0
        for i in range(gridShape[0]):
            currY = 0
            for j in range(gridShape[1]):
                m[currX:currX + cellList[i][j][0].shape[0],currY:currY + cellList[i][j][0].shape[1]] = cellList[i][j][0].m
                currY += cellList[i][j][0].shape[1]
            currX += cellList[i][j][0].shape[0]
        task["test"][s]["input"] = m.tolist()

        cellList = [cell for cell in t.testSamples[s].outMatrix.asymmetricGrid.cells]
        gridShape = t.testSamples[0].outMatrix.asymmetricGrid.shape
        newShape = (sum(cellList[i][0][0].shape[0] for i in range(gridShape[0])),sum(cellList[0][i][0].shape[1] for i in range(gridShape[1])))
        m = np.zeros(newShape, dtype=np.uint8)
        currX = 0
        for i in range(gridShape[0]):
            currY = 0
            for j in range(gridShape[1]):
                m[currX:currX + cellList[i][j][0].shape[0],currY:currY + cellList[i][j][0].shape[1]] = cellList[i][j][0].m
                currY += cellList[i][j][0].shape[1]
            currX += cellList[i][j][0].shape[0]
        task["test"][s]["output"] = m.tolist()

def recoverGeneralGrid(t, x, s):
    realX = t.testSamples[s].inMatrix.m.copy()
    cells = t.testSamples[s].inMatrix.grid.cells
    currX = 0
    for cellI in range(len(cells)):
        currY = 0
        for cellJ in range(len(cells[0])):
            cellShape = cells[cellI][cellJ][0].shape
            position = cells[cellI][cellJ][1]
            realX[position[0]: position[0]+cellShape[0], position[1]:position[1]+cellShape[1]] =\
                x[currX: currX+cellShape[0],currY: currY+cellShape[1]]
            currY += cellShape[1]
        currX += cellShape[0]
    return realX

def recoverGeneralAsymmetricGrid(t, x, s):
    realX = t.testSamples[s].inMatrix.m.copy()
    cells = t.testSamples[s].inMatrix.asymmetricGrid.cells
    currX = 0
    for cellI in range(len(cells)):
        currY = 0
        for cellJ in range(len(cells[0])):
            cellShape = cells[cellI][cellJ][0].shape
            position = cells[cellI][cellJ][1]
            realX[position[0]: position[0]+cellShape[0], position[1]:position[1]+cellShape[1]] =\
                x[currX: currX+cellShape[0],currY: currY+cellShape[1]]
            currY += cellShape[1]
        currX += cellShape[0]
    return realX

def rotateTaskWithOneBorder(t, task):
    rotTask = copy.deepcopy(task)
    rotations = {'train': [], 'test': []}
    for s in range(t.nTrain):
        border = t.trainSamples[s].commonFullBorders[0]
        if border.direction=='h' and border.position==0:
            rotations['train'].append(1)
            rotTask['train'][s]['input'] = np.rot90(t.trainSamples[s].inMatrix.m, 1).tolist()
            rotTask['train'][s]['output'] = np.rot90(t.trainSamples[s].outMatrix.m, 1).tolist()
        elif border.direction=='v' and border.position==t.trainSamples[s].inMatrix.shape[1]-1:
            rotations['train'].append(2)
            rotTask['train'][s]['input'] = np.rot90(t.trainSamples[s].inMatrix.m, 2).tolist()
            rotTask['train'][s]['output'] = np.rot90(t.trainSamples[s].outMatrix.m, 2).tolist()
        elif border.direction=='h' and border.position==t.trainSamples[s].inMatrix.shape[0]-1:
            rotations['train'].append(3)
            rotTask['train'][s]['input'] = np.rot90(t.trainSamples[s].inMatrix.m, 3).tolist()
            rotTask['train'][s]['output'] = np.rot90(t.trainSamples[s].outMatrix.m, 3).tolist()
        else:
            rotations['train'].append(0)

    for s in range(t.nTest):
        if t.submission:
            hasBorder=False
            for border in t.testSamples[s].inMatrix.fullBorders:
                if border.color!=t.testSamples[s].inMatrix.backgroundColor:
                    if border.direction=='h' and border.position==0:
                        rotations['test'].append(1)
                        rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 1).tolist()
                    elif border.direction=='v' and border.position==t.testSamples[s].inMatrix.shape[1]-1:
                        rotations['test'].append(2)
                        rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 2).tolist()
                    elif border.direction=='h' and border.position==t.testSamples[s].inMatrix.shape[0]-1:
                        rotations['test'].append(3)
                        rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 3).tolist()
                    else:
                        rotations['test'].append(0)
                    hasBorder=True
                    break
            if not hasBorder:
                return False, False
        else:
            border = t.testSamples[s].commonFullBorders[0]
            if border.direction=='h' and border.position==0:
                rotations['test'].append(1)
                rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 1).tolist()
                rotTask['test'][s]['output'] = np.rot90(t.testSamples[s].outMatrix.m, 1).tolist()
            elif border.direction=='v' and border.position==t.testSamples[s].inMatrix.shape[1]-1:
                rotations['test'].append(2)
                rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 2).tolist()
                rotTask['test'][s]['output'] = np.rot90(t.testSamples[s].outMatrix.m, 2).tolist()
            elif border.direction=='h' and border.position==t.testSamples[s].inMatrix.shape[0]-1:
                rotations['test'].append(3)
                rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 3).tolist()
                rotTask['test'][s]['output'] = np.rot90(t.testSamples[s].outMatrix.m, 3).tolist()
            else:
                rotations['test'].append(0)

    return rotTask, rotations

def rotateHVTask(t, task):
    rotTask = copy.deepcopy(task)
    rotations = {'train': [], 'test': []}

    for s in range(t.nTrain):
        if t.trainSamples[s].isVertical:
            rotations['train'].append(1)
            rotTask['train'][s]['input'] = np.rot90(t.trainSamples[s].inMatrix.m, 1).tolist()
            rotTask['train'][s]['output'] = np.rot90(t.trainSamples[s].outMatrix.m, 1).tolist()
        else:
            rotations['train'].append(0)

    for s in range(t.nTest):
        if t.submission:
            if t.testSamples[s].inMatrix.isHorizontal:
                rotations['test'].append(0)
            elif t.testSamples[s].inMatrix.isVertical:
                rotations['test'].append(1)
                rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 1).tolist()
            else:
                return False, False
        else:
            if not (hasattr(t.testSamples[s], 'isHorizontal') and hasattr(t.testSamples[s], 'isVertical')):
                return False, False
            if t.testSamples[s].isHorizontal:
                rotations['test'].append(0)
            elif t.testSamples[s].isVertical:
                rotations['test'].append(1)
                rotTask['test'][s]['input'] = np.rot90(t.testSamples[s].inMatrix.m, 1).tolist()
                rotTask['test'][s]['output'] = np.rot90(t.testSamples[s].outMatrix.m, 1).tolist()
            else:
                return False, False

    return rotTask, rotations

def recoverRotations(matrix, trainOrTest, s, rotations):
    if rotations[trainOrTest][s] == 1:
        m = np.rot90(matrix, 3)
    elif rotations[trainOrTest][s] == 2:
        m = np.rot90(matrix, 2)
    elif rotations[trainOrTest][s] == 3:
        m = np.rot90(matrix, 1)
    else:
        m = matrix.copy()
    return m


def tryOperations(t, c, cTask, b3c, firstIt=False):
    """
    Given a Task.Task t and a Candidate c, this function applies all the
    operations that make sense to the input matrices of c. After a certain
    operation is performed to all the input matrices, a new candidate is
    generated from the resulting output matrices. If the score of the candidate
    improves the score of any of the 3 best candidates, it will be saved in the
    variable b3c, which is an object of the class Best3Candidates.
    """
    if c.score==0 or b3c.allPerfect():
        return
    startOps = ("switchColors", "cropShape", "cropAllBackground", "minimize", \
                "maxColorFromCell", "deleteShapes", "replicateShapes","colorByPixels", \
                "paintGridLikeBackground") # applyEvolve?
    repeatIfPerfect = ("extendColor", "moveAllShapes")
    possibleOps = Utils.getPossibleOperations(t, c)
    for op in possibleOps:
        for s in range(t.nTrain):
            cTask["train"][s]["input"] = op(c.t.trainSamples[s].inMatrix).tolist()
            if c.t.sameIOShapes and len(c.t.fixedColors) != 0:
                cTask["train"][s]["input"] = Utils.correctFixedColors(\
                     c.t.trainSamples[s].inMatrix.m,\
                     np.array(cTask["train"][s]["input"]),\
                     c.t.fixedColors, c.t.commonOnlyChangedInColors).tolist()
        newPredictions = []
        for s in range(t.nTest):
            newOutput = op(c.t.testSamples[s].inMatrix)
            newPredictions.append(newOutput)
            cTask["test"][s]["input"] = newOutput.tolist()
            if c.t.sameIOShapes and len(c.t.fixedColors) != 0:
                cTask["test"][s]["input"] = Utils.correctFixedColors(\
                     c.t.testSamples[s].inMatrix.m,\
                     np.array(cTask["test"][s]["input"]),\
                     c.t.fixedColors, c.t.commonOnlyChangedInColors).tolist()
        cScore = sum([Utils.incorrectPixels(np.array(cTask["train"][s]["input"]), \
                                            t.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
        changedPixels = sum([Utils.incorrectPixels(c.t.trainSamples[s].inMatrix.m, \
                                                  np.array(cTask["train"][s]["input"])) for s in range(t.nTrain)])
        #print(op, cScore)
        #plot_task(cTask)
        newCandidate = Candidate(c.ops+[op], c.tasks+[copy.deepcopy(cTask)], cScore,\
                                 predictions=newPredictions)
        b3c.addCandidate(newCandidate)
        if firstIt and str(op)[28:60].startswith(startOps):
            if all([np.array_equal(np.array(cTask["train"][s]["input"]), \
                                   t.trainSamples[s].inMatrix.m) for s in range(t.nTrain)]):
                continue
            newCandidate.generateTask()
            tryOperations(t, newCandidate, cTask, b3c)
        elif str(op)[28:60].startswith(repeatIfPerfect) and c.score - changedPixels == cScore and changedPixels != 0:
            newCandidate.generateTask()
            tryOperations(t, newCandidate, cTask, b3c)

class Solution():
    def __init__(self, index, taskId, ops):
        self.index = index
        self.taskId = taskId
        self.ops = ops

def getPredictionsFromTask(originalT, task):
    taskNeedsRecoloring = needsRecoloring(originalT)

    if taskNeedsRecoloring:
        task, trainRels, trainInvRels, testRels, testInvRels = orderTaskColors(originalT)
        t = Task.Task(task, taskId, submission=False)
    else:
        t = originalT
    cTask = copy.deepcopy(task)

    if t.sameIOShapes:
        taskNeedsCropping = needsCropping(t)
    else:
        taskNeedsCropping = False
    if taskNeedsCropping:
        cropPositions, backgrounds = cropTask(t, cTask)
        t2 = Task.Task(cTask, taskId, submission=False, backgrounds=backgrounds)
    elif t.hasUnchangedGrid:
        if t.gridCellsHaveOneColor:
            ignoreGrid(t, cTask) # This modifies cTask, ignoring the grid
            t2 = Task.Task(cTask, taskId, submission=False)
        elif t.outGridCellsHaveOneColor:
            ignoreGrid(t, cTask, inMatrix=False)
            t2 = Task.Task(cTask, taskId, submission=False)
        else:
            t2 = t
    elif t.hasUnchangedAsymmetricGrid and t.assymmetricGridCellsHaveOneColor:
        ignoreAsymmetricGrid(t, cTask)
        t2 = Task.Task(cTask, taskId, submission=False)
    #if t.hasUnchangedGrid:
    #    ignoreGeneralGrid(t, cTask)
    #    t2 = Task.Task(cTask, taskId, submission=False)
    #if t.hasUnchangedAsymmetricGrid:
    #    ignoreGeneralAsymmetricGrid(t, cTask)
    #    t2 = Task.Task(cTask, taskId, submission=False)
    else:
        t2 = t

    if t2.sameIOShapes:
        hasRotated = False
        if t2.hasOneFullBorder:
            hasRotated, rotateParams = rotateTaskWithOneBorder(t2, cTask)
        elif t2.requiresHVRotation:
            hasRotated, rotateParams = rotateHVTask(t2, cTask)
        if hasRotated!=False:
            cTask = hasRotated.copy()
            t2 = Task.Task(cTask, taskId, submission=False)

    cScore = sum([Utils.incorrectPixels(np.array(cTask["train"][s]["input"]), \
                                         t2.trainSamples[s].outMatrix.m) for s in range(t.nTrain)])
    
    dummyPredictions = [sample.inMatrix.m for sample in t2.testSamples]
    c = Candidate([], [task], score=cScore, predictions=dummyPredictions)
    c.t = t2
    b3c = Best1Candidates(c)

    # Generate the three candidates with best possible score
    prevScore = sum([c.score for c in b3c.candidates])
    firstIt = True
    while True:
        copyB3C = copy.deepcopy(b3c)
        for c in copyB3C.candidates:
            if c.score == 0:
                continue
            tryOperations(t2, c, cTask, b3c, firstIt)
            if firstIt:
                firstIt = False
                break
        score = sum([c.score for c in b3c.candidates])
        if score >= prevScore:
            break
        else:
            prevScore = score

    taskPredictions = []

    # Once the best 3 candidates have been found, make the predictions
    for s in range(t.nTest):
        taskPredictions.append([])
        for c in b3c.candidates:
            #print(c.ops)
            x = t2.testSamples[s].inMatrix.m.copy()
            for opI in range(len(c.ops)):
                newX = c.ops[opI](Task.Matrix(x))
                if t2.sameIOShapes and len(t2.fixedColors) != 0:
                    x = Utils.correctFixedColors(x, newX, t2.fixedColors, t2.commonOnlyChangedInColors)
                else:
                    x = newX.copy()
            if t2.sameIOShapes and hasRotated!=False:
                x = recoverRotations(x, "test", s, rotateParams)
            if taskNeedsCropping:
                x = recoverCroppedMatrix(x, originalT.testSamples[s].inMatrix.shape, \
                                         cropPositions["test"][s], t.testSamples[s].inMatrix.backgroundColor)
            elif t.hasUnchangedGrid and (t.gridCellsHaveOneColor or t.outGridCellsHaveOneColor):
                x = recoverGrid(t, x, s)
            elif t.hasUnchangedAsymmetricGrid and t.assymmetricGridCellsHaveOneColor:
                x = recoverAsymmetricGrid(t, x, s)
            #elif t.hasUnchangedGrid:
            #    x = recoverGeneralGrid(t, x, s)
            #elif t.hasUnchangedAsymmetricGrid:
            #    x = recoverGeneralAsymmetricGrid(t, x, s)
            if taskNeedsRecoloring:
                x = recoverOriginalColors(x, testRels[s])
            taskPredictions[s].append(x)

            #print(c.ops)
            #plot_sample(originalT.testSamples[s], x)
            #if Utils.incorrectPixels(x, originalT.testSamples[s].outMatrix.m) == 0:
                #print(idx)
                #print(idx, c.ops)
                #plot_task(idx)
                #break
                #solved.append(Solution(idx, taskId, c.ops))
                #solvedIds.append(idx)
                #break


    return taskPredictions, b3c

def load_data(mode):
    with open(f"./data/arc-agi_{mode}_challenges.json", 'r', encoding='utf-8') as file:
        training_examples = json.load(file)
    if mode != "test":
        with open(f"./data/arc-agi_{mode}_solutions.json", 'r', encoding='utf-8') as file:
            test_example = json.load(file)
        
        for key in training_examples.keys():
            # print(key)
            test_case = []
            test_outputs = test_example[key]
            test_inputs = training_examples[key]['test']
            for test_input, test_output in zip(test_inputs, test_outputs):
                test_input['output'] = test_output
                test_case.append(test_input)
            training_examples[key]['test'] = test_case

    else:
        test_example = None
    
    

    return training_examples, test_example


if __name__ == "__main__":
    train_tasks, train_solutions = load_data("training")
    valid_tasks, valid_solutions = load_data("evaluation")
    test_tasks, _ = load_data("test")
    cmap = colors.ListedColormap(
            ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
            '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    index = list(test_tasks.keys())

    allTasks = train_tasks.copy()

    for taskId in tqdm(allTasks, position=0, leave=True):
        if taskId != "694f12f3":
            continue
        task = allTasks[taskId]
        originalT = Task.Task(task, taskId, submission=False)
        print("Do all output matrices have the same shape as the input matrix?", originalT.sameIOShapes)
        print("Is the output always smaller?", originalT.outSmallerThanIn)
        print("Are there the same number of colors in every sample?", originalT.sameNSampleColors)
        print("Does the output always have the same colors as the input?", originalT.sameNumColors)
        predictions, b3c = getPredictionsFromTask(originalT, task.copy())

        separationByShapes = needsSeparationByShapes(originalT)
        separationByColors = needsSeparationByColors(originalT)
        print("separationByShapes:", separationByShapes)
        print("separationByColors:", separationByColors)
        if separationByShapes != False:
            separatedT = Task.Task(separationByShapes.separatedTask, taskId, submission=False)
            sepPredictions, sepB3c = getPredictionsFromTask(separatedT, separationByShapes.separatedTask.copy())

            mergedPredictions = []
            for s in range(originalT.nTest):
                mergedPredictions.append([])
                matrixRange = separationByShapes.getRange("test", s)
                matrices = [[sepPredictions[i][cand] for i in range(matrixRange[0], matrixRange[1])] \
                            for cand in range(3)]
                for cand in range(3):
                    pred = Utils.mergeMatrices(matrices[cand], originalT.backgroundColor)
                    mergedPredictions[s].append(pred)
                    #plot_sample(originalT.testSamples[s], pred)

            finalPredictions = []
            for s in range(originalT.nTest):
                finalPredictions.append([[], [], []])
            
            b3cIndices = b3c.getOrderedIndices()
            sepB3cIndices = sepB3c.getOrderedIndices()

            b3cIndex, sepB3cIndex = 0, 0
            i = 0
            if b3c.candidates[b3cIndices[0]].score==0:
                for s in range(originalT.nTest):
                    finalPredictions[s][0] = predictions[s][b3cIndices[0]]
                i += 1
            if sepB3c.candidates[sepB3cIndices[0]].score==0:
                for s in range(originalT.nTest):
                    finalPredictions[s][i] = mergedPredictions[s][sepB3cIndices[0]]
                i += 1
            while i < 3:
                if b3c.candidates[b3cIndices[b3cIndex]] < sepB3c.candidates[sepB3cIndices[sepB3cIndex]]:
                    for s in range(originalT.nTest):
                        finalPredictions[s][i] = predictions[s][b3cIndices[b3cIndex]]
                    b3cIndex += 1
                else:
                    for s in range(originalT.nTest):
                        finalPredictions[s][i] = mergedPredictions[s][sepB3cIndices[sepB3cIndex]]
                    sepB3cIndex += 1
                i += 1

        elif separationByColors != False:
            separatedT = Task.Task(separationByColors.separatedTask, taskId, submission=False)
            sepPredictions, sepB3c = getPredictionsFromTask(separatedT, separationByColors.separatedTask.copy())

            mergedPredictions = []
            for s in range(originalT.nTest):
                mergedPredictions.append([])
                matrixRange = separationByColors.getRange("test", s)
                matrices = [[sepPredictions[i][cand] for i in range(matrixRange[0], matrixRange[1])] \
                            for cand in range(3)]
                for cand in range(3):
                    pred = Utils.mergeMatrices(matrices[cand], originalT.backgroundColor)
                    mergedPredictions[s].append(pred)
                    #plot_sample(originalT.testSamples[s], pred)

            finalPredictions = []
            for s in range(originalT.nTest):
                finalPredictions.append([[], [], []])
            
            b3cIndices = b3c.getOrderedIndices()
            sepB3cIndices = sepB3c.getOrderedIndices()

            b3cIndex, sepB3cIndex = 0, 0
            i = 0
            if b3c.candidates[b3cIndices[0]].score==0:
                for s in range(originalT.nTest):
                    finalPredictions[s][0] = predictions[s][b3cIndices[0]]
                i += 1
            if sepB3c.candidates[sepB3cIndices[0]].score==0:
                for s in range(originalT.nTest):
                    finalPredictions[s][i] = mergedPredictions[s][sepB3cIndices[0]]
                i += 1
            while i < 3:
                if b3c.candidates[b3cIndices[b3cIndex]] < sepB3c.candidates[sepB3cIndices[sepB3cIndex]]:
                    for s in range(originalT.nTest):
                        finalPredictions[s][i] = predictions[s][b3cIndices[b3cIndex]]
                    b3cIndex += 1
                else:
                    for s in range(originalT.nTest):
                        finalPredictions[s][i] = mergedPredictions[s][sepB3cIndices[sepB3cIndex]]
                    sepB3cIndex += 1
                i += 1
        else:
            finalPredictions = predictions

        for s in range(originalT.nTest):
            for i in range(3):
                plot_sample(originalT.testSamples[s], finalPredictions[s][i])

class ARCState(NamedTuple):
    """The state of the Blocksworld.
    
    See the docstring of BlocksWorldModel for more details.
    """
    step_idx : int
    task: Task.Task
    originalT: Task.Task
    candidate: Candidate
    ctask: Task.Task

ARCAction = Callable[[Any], Any]
