import numpy as np
from collections import Counter
import copy


# %% Frontiers
class Frontier:
    """
    A Frontier is defined as a straight line with a single color that crosses
    all of the matrix. For example, if the matrix has shape MxN, then a
    Frontier will have shape Mx1 or 1xN. See the function "detectFrontiers"
    for details in the implementation.

    ...

    Attributes
    ----------
    color: int
        The color of the frontier
    directrion: str
        A character ('h' or 'v') determining whether the frontier is horizontal
        or vertical
    position: tuple
        A 2-tuple of ints determining the position of the upper-left pixel of
        the frontier
    """

    def __init__(self, color, direction, position):
        """
        direction can be 'h' or 'v' (horizontal, vertical)
        color, position and are all integers
        """
        self.color = color
        self.direction = direction
        self.position = position

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False


def detectFrontiers(m):
    """
    m is a numpy 2-dimensional matrix.
    """
    frontiers = []

    # Horizontal lines
    if m.shape[0] > 1:
        for i in range(m.shape[0]):
            color = m[i, 0]
            isFrontier = True
            for j in range(m.shape[1]):
                if color != m[i, j]:
                    isFrontier = False
                    break
            if isFrontier:
                frontiers.append(Frontier(color, "h", i))

    # Vertical lines
    if m.shape[1] > 1:
        for j in range(m.shape[1]):
            color = m[0, j]
            isFrontier = True
            for i in range(m.shape[0]):
                if color != m[i, j]:
                    isFrontier = False
                    break
            if isFrontier:
                frontiers.append(Frontier(color, "v", j))

    return frontiers


# %% Grids
class Grid:
    """
    An object of the class Grid is basically a collection of frontiers that
    have all the same color.
    It is useful to check, for example, whether the cells defined by the grid
    always have the same size or not.

    ...

    Attributes
    ----------
    color: int
        The color of the grid
    m: numpy.ndarray
        The whole matrix
    frontiers: list
        A list of all the frontiers the grid is composed of
    cells: list of list of 2-tuples
        cells can be viewed as a 2-dimensional matrix of 2-tuples (Matrix,
        position). The first element is an object of the class Matrix, and the
        second element is the position of the cell in m.
        Each element represents a cell of the grid.
    shape: tuple
        A 2-tuple of ints representing the number of cells of the grid
    nCells: int
        Number of cells of the grid
    cellList: list
        A list of all the cells
    allCellsSameShape: bool
        Determines whether all the cells of the grid have the same shape (as
        matrices).
    cellShape: tuple
        Only defined if allCellsSameShape is True. Shape of the cells.
    allCellsHaveOneColor: bool
        Determines whether the ALL of the cells of the grid are composed of
        pixels of the same color
    """

    def __init__(self, m, frontiers):
        self.color = frontiers[0].color
        self.m = m
        self.frontiers = frontiers
        hPositions = [f.position for f in frontiers if f.direction == "h"]
        hPositions.append(-1)
        hPositions.append(m.shape[0])
        hPositions.sort()
        vPositions = [f.position for f in frontiers if f.direction == "v"]
        vPositions.append(-1)
        vPositions.append(m.shape[1])
        vPositions.sort()
        # cells is a matrix (list of lists) of 2-tuples (Matrix, position)
        self.cells = []
        hShape = 0
        vShape = 0
        for h in range(len(hPositions) - 1):
            if hPositions[h] + 1 == hPositions[h + 1]:
                continue
            self.cells.append([])
            for v in range(len(vPositions) - 1):
                if vPositions[v] + 1 == vPositions[v + 1]:
                    continue
                if hShape == 0:
                    vShape += 1
                self.cells[hShape].append(
                    (
                        Matrix(
                            m[
                                hPositions[h] + 1 : hPositions[h + 1],
                                vPositions[v] + 1 : vPositions[v + 1],
                            ],
                            detectGrid=False,
                        ),
                        (hPositions[h] + 1, vPositions[v] + 1),
                    )
                )
            hShape += 1

        self.shape = (hShape, vShape)  # N of h cells x N of v cells
        self.cellList = []
        for cellRow in range(len(self.cells)):
            for cellCol in range(len(self.cells[0])):
                self.cellList.append(self.cells[cellRow][cellCol])
        self.allCellsSameShape = len(set([c[0].shape for c in self.cellList])) == 1
        if self.allCellsSameShape:
            self.cellShape = self.cells[0][0][0].shape

        self.nCells = len(self.cellList)

        # Check whether each cell has one and only one color
        self.allCellsHaveOneColor = True
        for c in self.cellList:
            if c[0].nColors != 1:
                self.allCellsHaveOneColor = False
                break

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([f in other.frontiers for f in self.frontiers])
        else:
            return False


# %% Frames
"""
class Frame:
    def __init__(self, matrix):
        self.m
        self.color
        self.position
        self.shape
        self.isFull
        
def detectFrames(matrix):
    frames = []
    m = matrix.m.copy()
    for i,j in np.ndindex(m.shape):
        color = m[i,j]
        iMax = m.shape[0]
        jMax = m.shape[1]
        for k in range(i+1, m.shape[0]):
            for l in range(j+1, m.shape[1]):
                if m[k,l]==color:
                    
        
    return frames
"""


# %% Shapes and subclasses
class Shape:
    def __init__(self, m, xPos, yPos, background, isBorder):
        # pixels is a 2xn numpy array, where n is the number of pixels
        self.m = m
        self.nPixels = m.size - np.count_nonzero(m == 255)
        self.background = background
        self.shape = m.shape
        self.position = (xPos, yPos)
        self.pixels = set([(i, j) for i, j in np.ndindex(m.shape) if m[i, j] != 255])

        # Is the shape in the border?
        self.isBorder = isBorder

        # Which colors does the shape have?
        self.colors = set(np.unique(m)) - set([255])
        self.nColors = len(self.colors)
        if self.nColors == 1:
            self.color = next(iter(self.colors))

        self.colorCount = Counter(self.m.flatten()) + Counter(
            {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        )
        del self.colorCount[255]

        # Symmetries
        self.lrSymmetric = np.array_equal(self.m, np.fliplr(self.m))
        self.udSymmetric = np.array_equal(self.m, np.flipud(self.m))
        if self.m.shape[0] == self.m.shape[1]:
            self.d1Symmetric = np.array_equal(self.m, self.m.T)
            self.d2Symmetric = np.array_equal(np.fliplr(self.m), (np.fliplr(self.m)).T)
        else:
            self.d1Symmetric = False
            self.d2Symmetric = False

        self.isRectangle = 255 not in np.unique(m)
        self.isSquare = self.isRectangle and self.shape[0] == self.shape[1]

        if self.isRectangle and self.nColors > 1:
            self.subshapes = detectShapes(
                self.m,
                background=self.colorCount.most_common(1)[0][0],
                singleColor=True,
                diagonals=False,
            )
        self.nHoles = self.getNHoles()

        if self.nColors == 1:
            self.isFullFrame = self.isFullFrame()
            self.isPartialFrame = self.isPartialFrame()
        else:
            self.isFullFrame = False
            self.isPartialFrame = False

        if self.nColors == 1:
            self.boolFeatures = []
            for c in range(10):
                self.boolFeatures.append(self.color == c)
            self.boolFeatures.append(self.isBorder)
            self.boolFeatures.append(not self.isBorder)
            self.boolFeatures.append(self.lrSymmetric)
            self.boolFeatures.append(self.udSymmetric)
            self.boolFeatures.append(self.d1Symmetric)
            self.boolFeatures.append(self.d2Symmetric)
            self.boolFeatures.append(self.isSquare)
            self.boolFeatures.append(self.isRectangle)
            for nPix in range(1, 30):
                self.boolFeatures.append(self.nPixels == nPix)
            self.boolFeatures.append((self.nPixels % 2) == 0)
            self.boolFeatures.append((self.nPixels % 2) == 1)

    def hasSameShape(
        self,
        other,
        sameColor=False,
        samePosition=False,
        rotation=False,
        mirror=False,
        scaling=False,
    ):
        if samePosition:
            if self.position != other.position:
                return False
        if sameColor:
            m1 = self.m
            m2 = other.m
        else:
            m1 = self.shapeDummyMatrix()
            m2 = other.shapeDummyMatrix()
        if scaling and m1.shape != m2.shape:

            def multiplyPixels(matrix, factor):
                m = np.zeros(
                    tuple(s * f for s, f in zip(matrix.shape, factor)), dtype=np.uint8
                )
                for i, j in np.ndindex(matrix.shape):
                    for k, l in np.ndindex(factor):
                        m[i * factor[0] + k, j * factor[1] + l] = matrix[i, j]
                return m

            if (m1.shape[0] % m2.shape[0]) == 0 and (m1.shape[1] % m2.shape[1]) == 0:
                factor = (
                    int(m1.shape[0] / m2.shape[0]),
                    int(m1.shape[1] / m2.shape[1]),
                )
                m2 = multiplyPixels(m2, factor)
            elif (m2.shape[0] % m1.shape[0]) == 0 and (m2.shape[1] % m1.shape[1]) == 0:
                factor = (
                    int(m2.shape[0] / m1.shape[0]),
                    int(m2.shape[1] / m1.shape[1]),
                )
                m1 = multiplyPixels(m1, factor)
            elif (
                rotation
                and (m1.shape[0] % m2.shape[1]) == 0
                and (m1.shape[1] % m2.shape[0]) == 0
            ):
                factor = (
                    int(m1.shape[0] / m2.shape[1]),
                    int(m1.shape[1] / m2.shape[0]),
                )
                m2 = multiplyPixels(m2, factor)
            elif (
                rotation
                and (m2.shape[0] % m1.shape[1]) == 0
                and (m2.shape[1] % m1.shape[0]) == 0
            ):
                factor = (
                    int(m2.shape[0] / m1.shape[1]),
                    int(m2.shape[1] / m1.shape[0]),
                )
                m1 = multiplyPixels(m1, factor)
            else:
                return False
        if rotation and not mirror:
            if any([np.array_equal(m1, np.rot90(m2, x)) for x in range(1, 4)]):
                return True
        if mirror and not rotation:
            if np.array_equal(m1, np.fliplr(m2)) or np.array_equal(m1, np.flipud(m2)):
                return True
        if mirror and rotation:
            for x in range(1, 4):
                if any(
                    [
                        np.array_equal(m1, np.rot90(m2, x))
                        or np.array_equal(m1, np.fliplr(np.rot90(m2, x)))
                        for x in range(0, 4)
                    ]
                ):
                    return True

        return np.array_equal(m1, m2)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.shape != other.shape:
                return False
            return np.array_equal(self.m, other.m)
        else:
            return False

    """
    def __hash__(self):
        return self.m
    """

    def isSubshape(self, other, sameColor=False, rotation=False, mirror=False):
        """
        The method checks if a shape fits inside another. Can take into account rotations and mirrors.
        Maybe it should be updated to return the positions of subshapes instead of a boolean?
        """
        # return positions
        if rotation:
            m1 = self.m
            for x in range(1, 4):
                if Shape(np.rot90(m1, x), 0, 0, 0, self.isBorder).isSubshape(
                    other, sameColor, False, mirror
                ):
                    return True
        if mirror == "lr":
            if Shape(self.m[::, ::-1], 0, 0, 0, self.isBorder).isSubshape(
                other, sameColor, rotation, False
            ):
                return True
        if mirror == "ud":
            if Shape(self.m[::-1, ::], 0, 0, 0, self.isBorder).isSubshape(
                other, sameColor, rotation, False
            ):
                return True
        if sameColor:
            if (
                hasattr(self, "color")
                and hasattr(other, "color")
                and self.color != other.color
            ):
                return False
        if any(other.shape[i] < self.shape[i] for i in [0, 1]):
            return False

        for yIn in range(other.shape[1] - self.shape[1] + 1):
            for xIn in range(other.shape[0] - self.shape[0] + 1):
                if sameColor:
                    if np.all(
                        np.logical_or(
                            (
                                self.m
                                == other.m[
                                    xIn : xIn + self.shape[0], yIn : yIn + self.shape[1]
                                ]
                            ),
                            self.m == 255,
                        )
                    ):
                        return True
                else:
                    if (
                        set([tuple(np.add(ps, [xIn, yIn])) for ps in self.pixels])
                        <= other.pixels
                    ):
                        return True
        return False

    def shapeDummyMatrix(self):
        """
        Returns the smallest possible matrix containing the shape. The values
        of the matrix are ones and zeros, depending on whether the pixel is a
        shape pixel or not.
        """
        return (self.m != 255).astype(np.uint8)

    def hasFeatures(self, features):
        for i in range(len(features)):
            if features[i] and not self.boolFeatures[i]:
                return False
        return True

    def getNHoles(self):
        nHoles = 0
        m = self.m
        seen = np.zeros((self.shape[0], self.shape[1]), dtype=bool)

        def isInHole(i, j):
            if i < 0 or j < 0 or i > self.shape[0] - 1 or j > self.shape[1] - 1:
                return False
            if seen[i, j] or m[i, j] != 255:
                return True
            seen[i, j] = True
            ret = (
                isInHole(i + 1, j)
                * isInHole(i - 1, j)
                * isInHole(i, j + 1)
                * isInHole(i, j - 1)
            )
            return ret

        for i, j in np.ndindex(m.shape):
            if m[i, j] == 255 and not seen[i, j]:
                if isInHole(i, j):
                    nHoles += 1
        return nHoles

    def isRotationInvariant(self, color=False):
        if color:
            m = np.rot90(self.m, 1)
            return np.array_equal(m, self.m)
        else:
            m2 = self.shapeDummyMatrix()
            m = np.rot90(m2, 1)
            return np.array_equal(m, m2)

    """
    def isFullFrame(self):
        if self.shape[0]<3 or self.shape[1]<3:
            return False
        for i in range(1, self.shape[0]-1):
            for j in range(1, self.shape[1]-1):
                if self.m[i,j] != 255:
                    return False
        if self.nPixels == 2 * (self.shape[0]+self.shape[1]-2):
            return True
        return False
    """

    def isPartialFrame(self):
        if self.shape[0] < 4 or self.shape[1] < 4 or len(self.pixels) < 4:
            return False
        if len(np.unique(self.m[1:-1, 1:-1])) > 1 or self.color in np.unique(
            self.m[1:-1, 1:-1]
        ):
            return False
        return True

    def isFullFrame(self):
        if self.shape[0] < 3 or self.shape[1] < 3:
            return False
        for i in range(self.shape[0]):
            if self.m[i, 0] == 255 or self.m[i, self.shape[1] - 1] == 255:
                return False
        for j in range(self.shape[1]):
            if self.m[0, j] == 255 or self.m[self.shape[0] - 1, j] == 255:
                return False

        # We require fullFrames to have less than 20% of the pixels inside the
        # frame of the same color of the frame

        if self.nPixels - 2 * (self.shape[0] + self.shape[1] - 2) < 0.2 * (
            self.shape[0] - 2
        ) * (self.shape[1] - 2):
            return True

        return False


def detectShapesByColor(x, background):
    shapes = []
    for c in range(10):
        if c == background or c not in x:
            continue
        mc = np.zeros(x.shape, dtype=int)
        mc[x == c] = c
        mc[x != c] = 255
        x1, x2, y1, y2 = 0, mc.shape[0] - 1, 0, mc.shape[1] - 1
        while x1 <= x2 and np.all(mc[x1, :] == 255):
            x1 += 1
        while x2 >= x1 and np.all(mc[x2, :] == 255):
            x2 -= 1
        while y1 <= y2 and np.all(mc[:, y1] == 255):
            y1 += 1
        while y2 >= y1 and np.all(mc[:, y2] == 255):
            y2 -= 1
        m = mc[x1 : x2 + 1, y1 : y2 + 1]
        s = Shape(m.copy(), x1, y1, background, False)
        shapes.append(s)
    return shapes


def detectShapes(x, background, singleColor=False, diagonals=False):
    """
    Given a numpy array x (2D), returns a list of the Shapes present in x
    """

    # Helper function to add pixels to a shape
    def addPixelsAround(i, j):
        def addPixel(i, j):
            if i < 0 or j < 0 or i > iMax or j > jMax or seen[i, j] == True:
                return
            if singleColor:
                if x[i, j] != color:
                    return
                newShape[i, j] = color
            else:
                if x[i, j] == background:
                    return
                newShape[i, j] = x[i, j]
            seen[i, j] = True
            addPixelsAround(i, j)

        addPixel(i - 1, j)
        addPixel(i + 1, j)
        addPixel(i, j - 1)
        addPixel(i, j + 1)

        if diagonals:
            addPixel(i - 1, j - 1)
            addPixel(i - 1, j + 1)
            addPixel(i + 1, j - 1)
            addPixel(i + 1, j + 1)

    def crop(matrix):
        ret = matrix.copy()
        for k in range(x.shape[0]):
            if any(matrix[k, :] != 255):  # -1==255 for dtype=np.uint8
                x0 = k
                break
        for k in reversed(range(x.shape[0])):
            if any(matrix[k, :] != 255):  # -1==255 for dtype=np.uint8
                x1 = k
                break
        for k in range(x.shape[1]):
            if any(matrix[:, k] != 255):  # -1==255 for dtype=np.uint8
                y0 = k
                break
        for k in reversed(range(x.shape[1])):
            if any(matrix[:, k] != 255):  # -1==255 for dtype=np.uint8
                y1 = k
                break
        return ret[x0 : x1 + 1, y0 : y1 + 1], x0, y0

    shapes = []
    seen = np.zeros(x.shape, dtype=bool)
    iMax = x.shape[0] - 1
    jMax = x.shape[1] - 1
    for i, j in np.ndindex(x.shape):
        if seen[i, j] == False:
            seen[i, j] = True
            if not singleColor and x[i, j] == background:
                continue
            newShape = np.full((x.shape), -1, dtype=np.uint8)
            newShape[i, j] = x[i, j]
            if singleColor:
                color = x[i][j]
            addPixelsAround(i, j)
            m, xPos, yPos = crop(newShape)
            isBorder = (
                xPos == 0
                or yPos == 0
                or (xPos + m.shape[0] == x.shape[0])
                or (yPos + m.shape[1] == x.shape[1])
            )
            s = Shape(m.copy(), xPos, yPos, background, isBorder)
            shapes.append(s)
    return shapes
    """
            # Is the shape in the border?
            isBorder = False
            if any([c[0] == 0 or c[1] == 0 or c[0] == iMax or c[1] == jMax for c in newShape]):
                isBorder = True
            
            # Now: What kind of shape is it???
            if len(newShape) == 1:
                s = Pixel(np.array(newShape), color, isBorder)
            else:
                # List containing the number of shape pixels surrounding each pixel
                nSurroundingPixels = []
                for p in newShape:
                    psp = 0
                    if [p[0]-1, p[1]] in newShape:
                        psp += 1
                    if [p[0]+1, p[1]] in newShape:
                        psp += 1
                    if [p[0], p[1]+1] in newShape:
                        psp += 1
                    if [p[0], p[1]-1] in newShape:
                        psp += 1
                    nSurroundingPixels.append(psp)
                # Check for loops and frames
                # If len(newShape) == 4, then it is a 2x2 square, not a loop!
                if all([s==2 for s in nSurroundingPixels]) and len(newShape) != 4:
                    maxI = max([p[0] for p in newShape])
                    minI = min([p[0] for p in newShape])
                    maxJ = max([p[1] for p in newShape])
                    minJ = min([p[1] for p in newShape])
                    isFrame = True
                    for p in newShape:
                        if p[0] not in [maxI, minI] and p[1] not in [maxJ, minJ]:
                            isFrame = False
                            s = Loop(np.array(newShape), color, isBorder)
                            break
                    if isFrame:
                        s = Frame(np.array(newShape), color, isBorder)
                # Check for lines and Frontiers
                spCounter = Counter(nSurroundingPixels)
                if len(spCounter) == 2 and 1 in spCounter.keys() and spCounter[1] == 2:
                    if len(set([p[0] for p in newShape])) == 1 or len(set([p[0] for p in newShape])):
                        if newShape[0][0] == newShape[1][0]:
                            s = Line(np.array(newShape), color, isBorder, 'v')
                        else:
                            s = Line(np.array(newShape), color, isBorder, 'h')
                    else:
                        s = Path(np.array(newShape), color, isBorder)
            if 's' not in locals(): 
                s = GeneralShape(np.array(newShape), color, isBorder)
            shapes.append(s)
            del s
    
    return shapes    

class Pixel(Shape):
    def __init__(self, pixels, color, isBorder):
        super().__init__(pixels, color, isBorder)
        self.nHoles=0
        self.isSquare=True
        self.isRectangle=True
        
class Path(Shape):
    def __init__(self, pixels, color, isBorder):
        super().__init__(pixels, color, isBorder)
        self.isSquare=False
        self.isRectangle=False
        self.nHoles=0

class Line(Path):
    def __init__(self, pixels, color, isBorder, orientation):
        super().__init__(pixels, color, isBorder)
        self.orientation = orientation
        
class Loop(Shape):
    def __init__(self, pixels, color, isBorder):
        super().__init__(pixels, color, isBorder)
        self.nHoles=1
        self.isSquare=False
            self.isRectangle=False
        
class Frame(Loop):
    def __init__(self, pixels, color, isBorder):
        super().__init__(pixels, color, isBorder)
        
class GeneralShape(Shape):
    def __init__(self, pixels, color, isBorder):
        super().__init__(pixels, color, isBorder)
        
        self.isRectangle = self.nPixels == (self.xLen+1) * (self.yLen+1)
        self.isSquare = self.isRectangle and self.xLen == self.yLen
        
        # Number of holes
        self.nHoles = self.getNHoles()
        
    """


def detectIsolatedPixels(matrix, dShapeList):
    pixList = []
    for sh in dShapeList:
        if sh.nPixels > 1 or sh.color == matrix.backgroundColor:
            continue
        else:
            cc = set()
            for i, j in np.ndindex(3, 3):
                if (
                    i - 1 + sh.position[0] < matrix.shape[0]
                    and i - 1 + sh.position[0] >= 0
                    and j - 1 + sh.position[1] < matrix.shape[1]
                    and j - 1 + sh.position[1] >= 0
                ):
                    cc = cc.union(
                        set([matrix.m[i - 1 + sh.position[0], j - 1 + sh.position[1]]])
                    )
            if len(cc) == 2:
                pixList.append(sh)
    return pixList


# %% Class Matrix
class Matrix:
    def __init__(self, m, detectGrid=True, backgroundColor=None):
        if type(m) == Matrix:
            return m

        self.m = np.array(m)

        # interesting properties:

        # Dimensions
        self.shape = self.m.shape
        self.nElements = self.m.size

        # Counter of colors
        self.colorCount = self.getColors()
        self.colors = set(self.colorCount.keys())
        self.nColors = len(self.colorCount)

        # Background color
        if backgroundColor == None:
            self.backgroundColor = max(self.colorCount, key=self.colorCount.get)
        else:
            self.backgroundColor = backgroundColor

        # Shapes
        self.shapes = detectShapes(self.m, self.backgroundColor, singleColor=True)
        self.nShapes = len(self.shapes)
        self.dShapes = detectShapes(
            self.m, self.backgroundColor, singleColor=True, diagonals=True
        )
        self.nDShapes = len(self.dShapes)
        self.fullFrames = [shape for shape in self.shapes if shape.isFullFrame]
        self.fullFrames = sorted(
            self.fullFrames, key=lambda x: x.shape[0] * x.shape[1], reverse=True
        )
        self.shapesByColor = detectShapesByColor(self.m, self.backgroundColor)
        self.partialFrames = [
            shape for shape in self.shapesByColor if shape.isPartialFrame
        ]
        self.isolatedPixels = detectIsolatedPixels(self, self.dShapes)
        self.nIsolatedPixels = len(self.isolatedPixels)
        # self.multicolorShapes = detectShapes(self.m, self.backgroundColor)
        # self.multicolorDShapes = detectShapes(self.m, self.backgroundColor, diagonals=True)
        # R: Since black is the most common background color.
        # self.nonBMulticolorShapes = detectShapes(self.m, 0)
        # self.nonBMulticolorDShapes = detectShapes(self.m, 0, diagonals=True)
        # Non-background shapes
        # self.notBackgroundShapes = [s for s in self.shapes if s.color != self.backgroundColor]
        # self.nNBShapes = len(self.notBackgroundShapes)
        # self.notBackgroundDShapes = [s for s in self.dShapes if s.color != self.backgroundColor]
        # self.nNBDShapes = len(self.notBackgroundDShapes)

        self.shapeColorCounter = Counter([s.color for s in self.shapes])
        self.blanks = []
        for s in self.shapes:
            if s.isRectangle and self.shapeColorCounter[s.color] == 1:
                self.blanks.append(s)

        # Frontiers
        self.frontiers = detectFrontiers(self.m)
        self.frontierColors = [f.color for f in self.frontiers]
        if len(self.frontiers) == 0:
            self.allFrontiersEqualColor = False
        else:
            self.allFrontiersEqualColor = self.frontierColors.count(
                self.frontiers[0]
            ) == len(self.frontiers)
        # Check if it's a grid and the dimensions of the cells
        self.isGrid = False
        self.isAsymmetricGrid = False
        if detectGrid:
            for fc in set(self.frontierColors):
                possibleGrid = [f for f in self.frontiers if f.color == fc]
                possibleGrid = Grid(self.m, possibleGrid)
                if possibleGrid.nCells > 1:
                    if possibleGrid.allCellsSameShape:
                        self.grid = copy.deepcopy(possibleGrid)
                        self.isGrid = True
                        self.asymmetricGrid = copy.deepcopy(possibleGrid)
                        self.isAsymmetricGrid = True
                        break
                    else:
                        self.asymmetricGrid = copy.deepcopy(possibleGrid)
                        self.isAsymmetricGrid = True

        # Shape-based backgroundColor
        if not self.isGrid:
            for shape in self.shapes:
                if shape.shape == self.shape:
                    self.backgroundColor = shape.color
                    break
        # Define multicolor shapes based on the background color
        self.multicolorShapes = detectShapes(self.m, self.backgroundColor)
        self.multicolorDShapes = detectShapes(
            self.m, self.backgroundColor, diagonals=True
        )
        self.dummyMatrix = (self.m != self.backgroundColor).astype(np.uint8)
        # Symmetries
        self.lrSymmetric = np.array_equal(self.m, np.fliplr(self.m))
        # Up-Down
        self.udSymmetric = np.array_equal(self.m, np.flipud(self.m))
        # Diagonals (only if square)
        if self.m.shape[0] == self.m.shape[1]:
            self.d1Symmetric = np.array_equal(self.m, self.m.T)
            self.d2Symmetric = np.array_equal(np.fliplr(self.m), (np.fliplr(self.m)).T)
        else:
            self.d1Symmetric = False
            self.d2Symmetric = False
        self.totalSymmetric = (
            self.lrSymmetric
            and self.udSymmetric
            and self.d1Symmetric
            and self.d2Symmetric
        )

        self.fullBorders = []
        for f in self.frontiers:
            if f.color != self.backgroundColor:
                if f.position == 0:
                    self.fullBorders.append(f)
                elif (f.direction == "h" and f.position == self.shape[0] - 1) or (
                    f.direction == "v" and f.position == self.shape[1] - 1
                ):
                    self.fullBorders.append(f)

        self.isVertical = False
        self.isHorizontal = False
        if len(self.frontiers) != 0:
            self.isVertical = all([f.direction == "v" for f in self.frontiers])
            self.isHorizontal = all([f.direction == "h" for f in self.frontiers])

    def getColors(self):
        unique, counts = np.unique(self.m, return_counts=True)
        return dict(zip(unique, counts))

    def getShapes(self, color=None, bigOrSmall=None, isBorder=None, diag=False):
        """
        Return a list of the shapes meeting the required specifications.
        """
        if diag:
            candidates = self.dShapes
        else:
            candidates = self.shapes
        if color != None:
            candidates = [c for c in candidates if c.color == color]
        if isBorder == True:
            candidates = [c for c in candidates if c.isBorder]
        if isBorder == False:
            candidates = [c for c in candidates if not c.isBorder]
        if len(candidates) == 0:
            return []
        sizes = [c.nPixels for c in candidates]
        if bigOrSmall == "big":
            maxSize = max(sizes)
            return [c for c in candidates if c.nPixels == maxSize]
        elif bigOrSmall == "small":
            minSize = min(sizes)
            return [c for c in candidates if c.nPixels == minSize]
        else:
            return candidates

    def followsColPattern(self):
        """
        This function checks whether the matrix follows a pattern of lines or
        columns being always the same (task 771 for example).
        Meant to be used for the output matrix mainly.
        It returns a number (length of the pattern) and "row" or "col".
        """
        m = self.m.copy()
        col0 = m[:, 0]
        for i in range(1, int(m.shape[1] / 2) + 1):
            if np.all(col0 == m[:, i]):
                isPattern = True
                for j in range(i):
                    k = 0
                    while k * i + j < m.shape[1]:
                        if np.any(m[:, j] != m[:, k * i + j]):
                            isPattern = False
                            break
                        k += 1
                    if not isPattern:
                        break
                if isPattern:
                    return i
        return False

    def followsRowPattern(self):
        m = self.m.copy()
        row0 = m[0, :]
        for i in range(1, int(m.shape[0] / 2) + 1):
            if np.all(row0 == m[i, :]):
                isPattern = True
                for j in range(i):
                    k = 0
                    while k * i + j < m.shape[0]:
                        if np.any(m[j, :] != m[k * i + j, :]):
                            isPattern = False
                            break
                        k += 1
                    if not isPattern:
                        break
                if isPattern:
                    return i
        return False

    """
    def shapeHasFeatures(self, index, features):
        for i in range(len(features)):
            if features[i] and not self.shapeFeatures[index][i]:
                return False
        return True
    """

    def isUniqueShape(self, shape):
        count = 0
        for sh in self.shapes:
            if sh.hasSameShape(shape):
                count += 1
        if count == 1:
            return True
        return False

    def getShapeAttributes(self, backgroundColor=0, singleColor=True, diagonals=True):
        """
        Returns list of shape attributes that matches list of shapes
        Add:
            - is border
            - has neighbors
            - is reference
            - is referenced
        """
        if singleColor:
            if diagonals:
                shapeList = [sh for sh in self.dShapes]
            else:
                shapeList = [sh for sh in self.shapes]
            if len([sh for sh in shapeList if sh.color != backgroundColor]) == 0:
                return [set() for sh in shapeList]
        else:
            if diagonals:
                shapeList = [sh for sh in self.multicolorDShapes]
            else:
                shapeList = [sh for sh in self.multicolorShapes]
            if len(shapeList) == 0:
                return [set()]
        attrList = [[] for i in range(len(shapeList))]
        if singleColor:
            cc = Counter([sh.color for sh in shapeList])
        if singleColor:
            sc = Counter(
                [sh.nPixels for sh in shapeList if sh.color != backgroundColor]
            )
        else:
            sc = Counter([sh.nPixels for sh in shapeList])
        largest, smallest, mcopies, mcolors = -1, 1000, 0, 0
        if singleColor:
            maxH, minH = max(
                [sh.nHoles for sh in shapeList if sh.color != backgroundColor]
            ), min([sh.nHoles for sh in shapeList if sh.color != backgroundColor])
        ila, ism = [], []
        for i in range(len(shapeList)):
            # color count
            if singleColor:
                if shapeList[i].color == backgroundColor:
                    attrList[i].append(-1)
                    continue
                else:
                    attrList[i].append(shapeList[i].color)
            else:
                attrList[i].append(shapeList[i].nColors)
                if shapeList[i].nColors > mcolors:
                    mcolors = shapeList[i].nColors
            # copies
            if singleColor:
                attrList[i] = [
                    np.count_nonzero(
                        [np.all(shapeList[i].pixels == osh.pixels) for osh in shapeList]
                    )
                ] + attrList[i]
                if attrList[i][0] > mcopies:
                    mcopies = attrList[i][0]
            else:
                attrList[i] = [
                    np.count_nonzero([shapeList[i] == osh for osh in shapeList])
                ] + attrList[i]
                if attrList[i][0] > mcopies:
                    mcopies = attrList[i][0]
            # unique color?
            if singleColor:
                if cc[shapeList[i].color] == 1:
                    attrList[i].append("UnCo")
            # more of x color?
            if not singleColor:
                for c in range(10):
                    if shapeList[i].colorCount[c] > 0 and shapeList[i].colorCount[
                        c
                    ] == max([sh.colorCount[c] for sh in shapeList]):
                        attrList[i].append("mo" + str(c))
            # largest?
            if len(shapeList[i].pixels) >= largest:
                ila += [i]
                if len(shapeList[i].pixels) > largest:
                    largest = len(shapeList[i].pixels)
                    ila = [i]
            # smallest?
            if len(shapeList[i].pixels) <= smallest:
                ism += [i]
                if len(shapeList[i].pixels) < smallest:
                    smallest = len(shapeList[i].pixels)
                    ism = [i]
            # unique size
            if sc[shapeList[i].nPixels] == 1 and len(sc) == 2:
                attrList[i].append("UnSi")
            # symmetric?
            if shapeList[i].lrSymmetric:
                attrList[i].append("LrSy")
            else:
                attrList[i].append("NlrSy")
            if shapeList[i].udSymmetric:
                attrList[i].append("UdSy")
            else:
                attrList[i].append("NudSy")
            if shapeList[i].d1Symmetric:
                attrList[i].append("D1Sy")
            else:
                attrList[i].append("ND1Sy")
            if shapeList[i].d2Symmetric:
                attrList[i].append("D2Sy")
            else:
                attrList[i].append("ND2Sy")
            attrList[i].append(shapeList[i].position)
            # pixels
            if len(shapeList[i].pixels) == 1:
                attrList[i].append("PiXl")
            # holes
            if singleColor:
                if maxH > minH:
                    if shapeList[i].nHoles == maxH:
                        attrList[i].append("MoHo")
                    elif shapeList[i].nHoles == minH:
                        attrList[i].append("LeHo")
                # is referenced by a full/partial frame?
                if any(
                    (
                        shapeList[i].position[0] >= fr.position[0]
                        and shapeList[i].position[1] >= fr.position[1]
                        and shapeList[i].position[0] + shapeList[i].shape[0]
                        <= fr.position[0] + fr.shape[0]
                        and shapeList[i].position[1] + shapeList[i].shape[1]
                        <= fr.position[1] + fr.shape[1]
                        and shapeList[i].color != fr.color
                    )
                    for fr in self.partialFrames
                ):
                    attrList[i].append("IsRef")
                if any(
                    (
                        shapeList[i].position[0] >= fr.position[0]
                        and shapeList[i].position[1] >= fr.position[1]
                        and shapeList[i].position[0] + shapeList[i].shape[0]
                        <= fr.position[0] + fr.shape[0]
                        and shapeList[i].position[1] + shapeList[i].shape[1]
                        <= fr.position[1] + fr.shape[1]
                        and shapeList[i].color != fr.color
                    )
                    for fr in self.fullFrames
                ):
                    attrList[i].append("IsFRef")

        if len(ism) == 1:
            attrList[ism[0]].append("SmSh")
        if len(ila) == 1:
            attrList[ila[0]].append("LaSh")
        for i in range(len(shapeList)):
            if len(attrList[i]) > 0 and attrList[i][0] == mcopies:
                attrList[i].append("MoCo")
        if not singleColor:
            for i in range(len(shapeList)):
                if len(attrList[i]) > 0 and attrList[i][1] == mcolors:
                    attrList[i].append("MoCl")
        if [l[0] for l in attrList].count(1) == 1:
            for i in range(len(shapeList)):
                if len(attrList[i]) > 0 and attrList[i][0] == 1:
                    attrList[i].append("UnSh")
                    break
        return [set(l[1:]) for l in attrList]


# %% Class Sample
class Sample:
    def __init__(self, s, trainOrTest, submission=False, backgroundColor=None):
        print(f"sample {s}")
        self.inMatrix = Matrix(s["input"], backgroundColor=backgroundColor)

        if trainOrTest == "train" or submission == False:
            self.outMatrix = Matrix(s["output"], backgroundColor=backgroundColor)

            # We want to compare the input and the output
            # Do they have the same dimensions?
            self.sameHeight = self.inMatrix.shape[0] == self.outMatrix.shape[0]
            self.sameWidth = self.inMatrix.shape[1] == self.outMatrix.shape[1]
            self.sameShape = self.sameHeight and self.sameWidth

            # Is the input shape a factor of the output shape?
            # Or the other way around?
            if not self.sameShape:
                if (self.inMatrix.shape[0] % self.outMatrix.shape[0]) == 0 and (
                    self.inMatrix.shape[1] % self.outMatrix.shape[1]
                ) == 0:
                    self.outShapeFactor = (
                        int(self.inMatrix.shape[0] / self.outMatrix.shape[0]),
                        int(self.inMatrix.shape[1] / self.outMatrix.shape[1]),
                    )
                if (self.outMatrix.shape[0] % self.inMatrix.shape[0]) == 0 and (
                    self.outMatrix.shape[1] % self.inMatrix.shape[1]
                ) == 0:
                    self.inShapeFactor = (
                        int(self.outMatrix.shape[0] / self.inMatrix.shape[0]),
                        int(self.outMatrix.shape[1] / self.inMatrix.shape[1]),
                    )
            """
            if self.sameShape:
                self.diffMatrix = Matrix((self.inMatrix.m - self.outMatrix.m).tolist())
                self.diffPixels = np.count_nonzero(self.diffMatrix.m)
            """
            # Is one a subset of the other? for now always includes diagonals
            self.inSmallerThanOut = (
                all(self.inMatrix.shape[i] <= self.outMatrix.shape[i] for i in [0, 1])
                and not self.sameShape
            )
            self.outSmallerThanIn = (
                all(self.inMatrix.shape[i] >= self.outMatrix.shape[i] for i in [0, 1])
                and not self.sameShape
            )

            # R: Is the output a shape (faster than checking if is a subset?

            if self.outSmallerThanIn:
                # check if output is the size of a multicolored shape
                self.outIsInMulticolorShapeSize = any(
                    (sh.shape == self.outMatrix.shape)
                    for sh in self.inMatrix.multicolorShapes
                )
                self.outIsInMulticolorDShapeSize = any(
                    (sh.shape == self.outMatrix.shape)
                    for sh in self.inMatrix.multicolorDShapes
                )
            (
                self.commonShapes,
                self.commonDShapes,
                self.commonMulticolorShapes,
                self.commonMulticolorDShapes,
            ) = ([], [], [], [])
            if len(self.inMatrix.shapes) < 15 or len(self.outMatrix.shapes) < 10:
                self.commonShapes = self.getCommonShapes(
                    diagonal=False,
                    sameColor=True,
                    multicolor=False,
                    rotation=True,
                    scaling=True,
                    mirror=True,
                )
            if len(self.inMatrix.dShapes) < 15 or len(self.outMatrix.dShapes) < 10:
                self.commonDShapes = self.getCommonShapes(
                    diagonal=True,
                    sameColor=True,
                    multicolor=False,
                    rotation=True,
                    scaling=True,
                    mirror=True,
                )
            if (
                len(self.inMatrix.multicolorShapes) < 15
                or len(self.outMatrix.multicolorShapes) < 10
            ):
                self.commonMulticolorShapes = self.getCommonShapes(
                    diagonal=False,
                    sameColor=True,
                    multicolor=True,
                    rotation=True,
                    scaling=True,
                    mirror=True,
                )
            if (
                len(self.inMatrix.multicolorDShapes) < 15
                or len(self.outMatrix.multicolorDShapes) < 10
            ):
                self.commonMulticolorDShapes = self.getCommonShapes(
                    diagonal=True,
                    sameColor=True,
                    multicolor=True,
                    rotation=True,
                    scaling=True,
                    mirror=True,
                )
            # self.commonShapesNoColor = self.getCommonShapes(diagonal=False, sameColor=False,\
            #                                         multicolor=False, rotation=True, scaling=True, mirror=True)
            # self.commonDShapesNoColor = self.getCommonShapes(diagonal=True, sameColor=False,\
            #                                          multicolor=False, rotation=True, scaling=True, mirror=True)
            # self.commonShapesNoColor = self.getCommonShapes(diagonal=False, sameColor=False,\
            #                                                       multicolor=True, rotation=True, scaling=True, mirror=True)
            # self.commonDShapesNoColor = self.getCommonShapes(diagonal=True, sameColor=False,\
            #                                                        multicolor=True, rotation=True, scaling=True, mirror=True)

            """
            # Is the output a subset of the input?
            self.inSubsetOfOutIndices = set()
            if self.inSmallerThanOut:
                for i, j in np.ndindex((self.outMatrix.shape[0] - self.inMatrix.shape[0] + 1, self.outMatrix.shape[1] - self.inMatrix.shape[1] + 1)):
                    if np.all(self.inMatrix.m == self.outMatrix.m[i:i+self.inMatrix.shape[0], j:j+self.inMatrix.shape[1]]):
                        self.inSubsetOfOutIndices.add((i, j))
            # Is the input a subset of the output?
            self.outSubsetOfInIndices = set()
            if self.outSmallerThanIn:
                for i, j in np.ndindex((self.inMatrix.shape[0] - self.outMatrix.shape[0] + 1, self.inMatrix.shape[1] - self.outMatrix.shape[1] + 1)):
                    if np.all(self.outMatrix.m == self.inMatrix.m[i:i+self.outMatrix.shape[0], j:j+self.outMatrix.shape[1]]):
                        self.outSubsetOfInIndices.add((i, j))
                #Is output a single input shape?
                if len(self.outSubsetOfInIndices) == 1:
                    #modify to compute background correctly
                    for sh in self.outMatrix.shapes:
                        if sh.m.size == self.outMatrix.m.size:
                            osh = sh
                            self.outIsShape = True
                            self.outIsShapeAttributes = []
                            for ish in self.inMatrix.shapes:
                                if ish.m == osh.m:
                                    break
                            self.outIsShapeAttributes = attribute_list(ish, self.inMatrix)
                            break
            """
            # Which colors are there in the sample?
            self.colors = set(self.inMatrix.colors | self.outMatrix.colors)
            self.commonColors = set(self.inMatrix.colors & self.outMatrix.colors)
            self.nColors = len(self.colors)
            # Do they have the same colors?
            self.sameColors = len(self.colors) == len(self.commonColors)
            # Do they have the same number of colors?
            self.sameNumColors = self.inMatrix.nColors == self.outMatrix.nColors
            # Does output contain all input colors or viceversa?
            self.inHasOutColors = self.outMatrix.colors <= self.inMatrix.colors
            self.outHasInColors = self.inMatrix.colors <= self.outMatrix.colors
            if self.sameShape:
                # Which pixels changes happened? How many times?
                self.changedPixels = Counter()
                self.sameColorCount = (
                    self.inMatrix.colorCount == self.outMatrix.colorCount
                )
                for i, j in np.ndindex(self.inMatrix.shape):
                    if self.inMatrix.m[i, j] != self.outMatrix.m[i, j]:
                        self.changedPixels[
                            (self.inMatrix.m[i, j], self.outMatrix.m[i, j])
                        ] += 1
                # Are any of these changes complete? (i.e. all pixels of one color are changed to another one)
                self.completeColorChanges = set(
                    change
                    for change in self.changedPixels.keys()
                    if self.changedPixels[change] == self.inMatrix.colorCount[change[0]]
                    and change[0] not in self.outMatrix.colorCount.keys()
                )
                self.allColorChangesAreComplete = len(self.changedPixels) == len(
                    self.completeColorChanges
                )
                # Does any color never change?
                self.changedInColors = set(
                    change[0] for change in self.changedPixels.keys()
                )
                self.changedOutColors = set(
                    change[1] for change in self.changedPixels.keys()
                )
                self.unchangedColors = set(
                    x
                    for x in self.colors
                    if x not in set.union(self.changedInColors, self.changedOutColors)
                )
                # Colors that stay unchanged
                self.fixedColors = set(
                    x
                    for x in self.colors
                    if x not in set.union(self.changedInColors, self.changedOutColors)
                )

            if self.sameShape and self.sameColorCount:
                self.sameRowCount = True
                for r in range(self.inMatrix.shape[0]):
                    _, inCounts = np.unique(self.inMatrix.m[r, :], return_counts=True)
                    _, outCounts = np.unique(self.outMatrix.m[r, :], return_counts=True)
                    if not np.array_equal(inCounts, outCounts):
                        self.sameRowCount = False
                        break
                self.sameColCount = True
                for c in range(self.inMatrix.shape[1]):
                    _, inCounts = np.unique(self.inMatrix.m[:, c], return_counts=True)
                    _, outCounts = np.unique(self.outMatrix.m[:, c], return_counts=True)
                    if not np.array_equal(inCounts, outCounts):
                        self.sameColCount = False
                        break

            # Shapes in the input that are fixed
            if self.sameShape:
                self.fixedShapes = []
                for sh in self.inMatrix.shapes:
                    if sh.color in self.fixedColors:
                        continue
                    shapeIsFixed = True
                    for i, j in np.ndindex(sh.shape):
                        if sh.m[i, j] != 255:
                            if (
                                self.outMatrix.m[sh.position[0] + i, sh.position[1] + j]
                                != sh.m[i, j]
                            ):
                                shapeIsFixed = False
                                break
                    if shapeIsFixed:
                        self.fixedShapes.append(sh)

            # Frames
            self.commonFullFrames = [
                f for f in self.inMatrix.fullFrames if f in self.outMatrix.fullFrames
            ]
            if len(self.inMatrix.fullFrames) == 1:
                frameM = self.inMatrix.fullFrames[0].m.copy()
                frameM[frameM == 255] = self.inMatrix.fullFrames[0].background
                if frameM.shape == self.outMatrix.shape:
                    self.frameIsOutShape = True
                elif frameM.shape == (
                    self.outMatrix.shape[0] + 1,
                    self.outMatrix.shape[1] + 1,
                ):
                    self.frameInsideIsOutShape = True

            # Grids
            # Is the grid the same in the input and in the output?
            self.gridIsUnchanged = (
                self.inMatrix.isGrid
                and self.outMatrix.isGrid
                and self.inMatrix.grid == self.outMatrix.grid
            )
            # Does the shape of the grid cells determine the output shape?
            if hasattr(self.inMatrix, "grid") and self.inMatrix.grid.allCellsSameShape:
                self.gridCellIsOutputShape = (
                    self.outMatrix.shape == self.inMatrix.grid.cellShape
                )
            # Does the shape of the input determine the shape of the grid cells of the output?
            if (
                hasattr(self.outMatrix, "grid")
                and self.outMatrix.grid.allCellsSameShape
            ):
                self.gridCellIsInputShape = (
                    self.inMatrix.shape == self.outMatrix.grid.cellShape
                )
            # Do all the grid cells have one color?
            if self.gridIsUnchanged:
                self.gridCellsHaveOneColor = (
                    self.inMatrix.grid.allCellsHaveOneColor
                    and self.outMatrix.grid.allCellsHaveOneColor
                )
            # Asymmetric grids
            self.asymmetricGridIsUnchanged = (
                self.inMatrix.isAsymmetricGrid
                and self.outMatrix.isAsymmetricGrid
                and self.inMatrix.asymmetricGrid == self.outMatrix.asymmetricGrid
            )
            if self.asymmetricGridIsUnchanged:
                self.asymmetricGridCellsHaveOneColor = (
                    self.inMatrix.asymmetricGrid.allCellsHaveOneColor
                    and self.outMatrix.asymmetricGrid.allCellsHaveOneColor
                )

            # Is there a blank to fill?
            self.inputHasBlank = len(self.inMatrix.blanks) > 0
            if self.inputHasBlank:
                for s in self.inMatrix.blanks:
                    if s.shape == self.outMatrix.shape:
                        self.blankToFill = s

            # Does the output matrix follow a pattern?
            self.followsRowPattern = self.outMatrix.followsRowPattern()
            self.followsColPattern = self.outMatrix.followsColPattern()

            # Full borders and horizontal/vertical
            if self.sameShape:
                self.commonFullBorders = []
                for inBorder in self.inMatrix.fullBorders:
                    for outBorder in self.outMatrix.fullBorders:
                        if inBorder == outBorder:
                            self.commonFullBorders.append(inBorder)

                self.isHorizontal = (
                    self.inMatrix.isHorizontal and self.outMatrix.isHorizontal
                )
                self.isVertical = self.inMatrix.isVertical and self.outMatrix.isVertical

    def getCommonShapes(
        self,
        diagonal=True,
        multicolor=False,
        sameColor=False,
        samePosition=False,
        rotation=False,
        mirror=False,
        scaling=False,
    ):
        comSh, countSh = [], []
        if diagonal:
            if not multicolor:
                ishs = self.inMatrix.dShapes
                oshs = self.outMatrix.dShapes
            else:
                ishs = self.inMatrix.multicolorDShapes
                oshs = self.outMatrix.multicolorDShapes
        else:
            if not multicolor:
                ishs = self.inMatrix.shapes
                oshs = self.outMatrix.shapes
            else:
                ishs = self.inMatrix.multicolorShapes
                oshs = self.outMatrix.multicolorShapes
        # Arbitrary: shapes have size < 100.
        for ish in ishs:
            outCount = 0
            if len(ish.pixels) == 1 or len(ish.pixels) > 100:
                continue
            for osh in oshs:
                if len(osh.pixels) == 1 or len(osh.pixels) > 100:
                    continue
                if ish.hasSameShape(
                    osh,
                    sameColor=sameColor,
                    samePosition=samePosition,
                    rotation=rotation,
                    mirror=mirror,
                    scaling=scaling,
                ):
                    outCount += 1
            if outCount > 0:
                comSh.append(
                    (
                        ish,
                        np.count_nonzero(
                            [
                                ish.hasSameShape(
                                    ish2,
                                    sameColor=sameColor,
                                    samePosition=samePosition,
                                    rotation=rotation,
                                    mirror=mirror,
                                    scaling=scaling,
                                )
                                for ish2 in ishs
                            ]
                        ),
                        outCount,
                    )
                )
            # countSh.append((ish, np.count_nonzero([ish.hasSameShape(ish2, sameColor=True, samePosition=samePosition,\
            #                        rotation=rotation, mirror=mirror, scaling=False) for ish2 in ishs])))
        if multicolor:
            return comSh
        else:
            return comSh  # , countSh


# %% Class Task
class Task:
    def __init__(self, t, i, submission=False, backgrounds=None):
        self.task = t
        self.index = i
        self.submission = submission

        if backgrounds == None:
            self.trainSamples = [Sample(s, "train", submission) for s in t["train"]]
            self.testSamples = [Sample(s, "test", submission) for s in t["test"]]
        else:
            self.trainSamples = [
                Sample(t["train"][s], "train", submission, backgrounds["train"][s])
                for s in range(len(t["train"]))
            ]
            self.testSamples = [
                Sample(t["test"][s], "test", submission, backgrounds["test"][s])
                for s in range(len(t["test"]))
            ]

        self.nTrain = len(self.trainSamples)
        self.nTest = len(self.testSamples)

        # Common properties I want to know:

        # Dimension:
        # Do all input/output matrices have the same shape?
        inShapes = [s.inMatrix.shape for s in self.trainSamples]
        self.sameInShape = self.allEqual(inShapes)
        if self.sameInShape:
            self.inShape = self.trainSamples[0].inMatrix.shape
        outShapes = [s.outMatrix.shape for s in self.trainSamples]
        self.sameOutShape = self.allEqual(outShapes)
        if self.sameOutShape:
            self.outShape = self.trainSamples[0].outMatrix.shape

        # Do all output matrices have the same shape as the input matrix?
        self.sameIOShapes = all([s.sameShape for s in self.trainSamples])

        # Are the input/output matrices always squared?
        self.inMatricesSquared = all(
            [
                s.inMatrix.shape[0] == s.inMatrix.shape[1]
                for s in self.trainSamples + self.testSamples
            ]
        )
        self.outMatricesSquared = all(
            [s.outMatrix.shape[0] == s.outMatrix.shape[1] for s in self.trainSamples]
        )

        # Are shapes of in (out) matrices always a factor of the shape of the
        # out (in) matrices?
        if all([hasattr(s, "inShapeFactor") for s in self.trainSamples]):
            if self.allEqual([s.inShapeFactor for s in self.trainSamples]):
                self.inShapeFactor = self.trainSamples[0].inShapeFactor
            elif all(
                [
                    s.inMatrix.shape[0] ** 2 == s.outMatrix.shape[0]
                    and s.inMatrix.shape[1] ** 2 == s.outMatrix.shape[1]
                    for s in self.trainSamples
                ]
            ):
                self.inShapeFactor = "squared"
            elif all(
                [
                    s.inMatrix.shape[0] ** 2 == s.outMatrix.shape[0]
                    and s.inMatrix.shape[1] == s.outMatrix.shape[1]
                    for s in self.trainSamples
                ]
            ):
                self.inShapeFactor = "xSquared"
            elif all(
                [
                    s.inMatrix.shape[0] == s.outMatrix.shape[0]
                    and s.inMatrix.shape[1] ** 2 == s.outMatrix.shape[1]
                    for s in self.trainSamples
                ]
            ):
                self.inShapeFactor = "ySquared"
            elif all(
                [
                    s.inMatrix.shape[0] * s.inMatrix.nColors == s.outMatrix.shape[0]
                    and s.inMatrix.shape[1] * s.inMatrix.nColors == s.outMatrix.shape[1]
                    for s in self.trainSamples
                ]
            ):
                self.inShapeFactor = "nColors"
            elif all(
                [
                    s.inMatrix.shape[0] * (s.inMatrix.nColors - 1)
                    == s.outMatrix.shape[0]
                    and s.inMatrix.shape[1] * (s.inMatrix.nColors - 1)
                    == s.outMatrix.shape[1]
                    for s in self.trainSamples
                ]
            ):
                self.inShapeFactor = "nColors-1"
        if all([hasattr(s, "outShapeFactor") for s in self.trainSamples]):
            if self.allEqual([s.outShapeFactor for s in self.trainSamples]):
                self.outShapeFactor = self.trainSamples[0].outShapeFactor

        # Is the output always smaller?
        self.outSmallerThanIn = all(s.outSmallerThanIn for s in self.trainSamples)
        self.inSmallerThanOut = all(s.inSmallerThanOut for s in self.trainSamples)

        # Check for I/O subsets
        """
        self.inSubsetOfOut = self.trainSamples[0].inSubsetOfOutIndices
        for s in self.trainSamples:
            self.inSubsetOfOut = set.intersection(self.inSubsetOfOut, s.inSubsetOfOutIndices)
        self.outSubsetOfIn = self.trainSamples[0].outSubsetOfInIndices
        for s in self.trainSamples:
            self.outSubsetOfIn = set.intersection(self.outSubsetOfIn, s.outSubsetOfInIndices)
        """

        # Symmetries:
        # Are all outputs LR, UD, D1 or D2 symmetric?
        self.lrSymmetric = all([s.outMatrix.lrSymmetric for s in self.trainSamples])
        self.udSymmetric = all([s.outMatrix.udSymmetric for s in self.trainSamples])
        self.d1Symmetric = all([s.outMatrix.d1Symmetric for s in self.trainSamples])
        self.d2Symmetric = all([s.outMatrix.d2Symmetric for s in self.trainSamples])

        # Colors
        # How many colors are there in the input? Is it always the same number?
        # How many colors are there in the output? Is it always the same number?
        self.sameNumColors = all([s.sameNumColors for s in self.trainSamples])
        self.nInColors = [s.inMatrix.nColors for s in self.trainSamples] + [
            s.inMatrix.nColors for s in self.testSamples
        ]
        self.sameNInColors = self.allEqual(self.nInColors)
        self.nOutColors = [s.outMatrix.nColors for s in self.trainSamples]
        self.sameNOutColors = self.allEqual(self.nOutColors)
        # Which colors does the input have? Union and intersection.
        self.inColors = [
            s.inMatrix.colors for s in self.trainSamples + self.testSamples
        ]
        self.commonInColors = set.intersection(*self.inColors)
        self.totalInColors = set.union(*self.inColors)
        # Which colors does the output have? Union and intersection.
        self.outColors = [s.outMatrix.colors for s in self.trainSamples]
        self.commonOutColors = set.intersection(*self.outColors)
        self.totalOutColors = set.union(*self.outColors)
        # Which colors appear in every sample?
        self.sampleColors = [s.colors for s in self.trainSamples]
        self.commonSampleColors = set.intersection(*self.sampleColors)
        # Input colors of the test samples
        self.testInColors = [s.inMatrix.colors for s in self.testSamples]
        # Are there the same number of colors in every sample?
        self.sameNSampleColors = self.allEqual(
            [len(sc) for sc in self.sampleColors]
        ) and all(
            [
                len(s.inMatrix.colors | self.commonOutColors)
                <= len(self.sampleColors[0])
                for s in self.testSamples
            ]
        )
        # How many colors are there in total? Which ones?
        self.colors = self.totalInColors | self.totalOutColors
        self.nColors = len(self.colors)
        # Does the output always have the same colors as the input?
        if self.sameNumColors:
            self.sameIOColors = all(
                [i == j for i, j in zip(self.inColors, self.outColors)]
            )
        if self.sameIOShapes:
            # Do the matrices have the same color count?
            self.sameColorCount = all([s.sameColorCount for s in self.trainSamples])
            if self.sameColorCount:
                self.sameRowCount = all([s.sameRowCount for s in self.trainSamples])
                self.sameColCount = all([s.sameColCount for s in self.trainSamples])
            # Which color changes happen? Union and intersection.
            cc = [set(s.changedPixels.keys()) for s in self.trainSamples]
            self.colorChanges = set.union(*cc)
            self.commonColorChanges = set.intersection(*cc)
            # Does any color always change? (to and from)
            self.changedInColors = [s.changedInColors for s in self.trainSamples]
            self.commonChangedInColors = set.intersection(*self.changedInColors)
            self.changedOutColors = [s.changedOutColors for s in self.trainSamples]
            self.commonChangedOutColors = set.intersection(*self.changedOutColors)
            self.commonOnlyChangedInColors = self.commonChangedInColors - set.union(
                *self.changedOutColors
            )
            # Complete color changes
            self.completeColorChanges = [
                s.completeColorChanges for s in self.trainSamples
            ]
            self.commonCompleteColorChanges = set.intersection(
                *self.completeColorChanges
            )
            self.allColorChangesAreComplete = all(
                [s.allColorChangesAreComplete for s in self.trainSamples]
            )
            # Are there any fixed colors?
            self.fixedColors = set.intersection(
                *[s.fixedColors for s in self.trainSamples]
            )
            self.fixedColors2 = (
                set.union(*[s.fixedColors for s in self.trainSamples])
                - set.union(*[s.changedInColors for s in self.trainSamples])
                - set.union(*[s.changedOutColors for s in self.trainSamples])
            )
            # Does any color never change?
            if self.commonChangedInColors == set(self.changedInColors[0]):
                self.unchangedColors = set(range(10)) - self.commonChangedInColors
            else:
                self.unchangedColors = [s.unchangedColors for s in self.trainSamples]
                self.unchangedColors = set.intersection(*self.unchangedColors)

        # Is the number of pixels changed always the same?
        """
        if self.sameIOShapes:
            self.sameChanges = self.allEqual([s.diffPixels for s in self.trainSamples])
        """

        # R: is output a shape in the input
        self.outIsInMulticolorShapeSize = False
        self.outIsInMulticolorDShapeSize = False

        if all(
            [
                (
                    hasattr(s, "outIsInMulticolorShapeSize")
                    and s.outIsInMulticolorShapeSize
                )
                for s in self.trainSamples
            ]
        ):
            self.outIsInMulticolorShapeSize = True
        if all(
            [
                (
                    hasattr(s, "outIsInMulticolorDShapeSize")
                    and s.outIsInMulticolorDShapeSize
                )
                for s in self.trainSamples
            ]
        ):
            self.outIsInMulticolorDShapeSize = True

        self.nCommonInOutShapes = min(len(s.commonShapes) for s in self.trainSamples)
        self.nCommonInOutDShapes = min(len(s.commonDShapes) for s in self.trainSamples)
        # self.nCommonInOutShapesNoColor = min(len(s.commonShapesNoColor) for s in self.trainSamples)
        # self.nCommonInOutDShapesNoColor = min(len(s.commonDShapesNoColor) for s in self.trainSamples)
        self.nCommonInOutMulticolorShapes = min(
            len(s.commonMulticolorShapes) for s in self.trainSamples
        )
        self.nCommonInOutMulticolorDShapes = min(
            len(s.commonMulticolorDShapes) for s in self.trainSamples
        )
        # self.nCommonInOutMulticolorShapesNoColor = min(len(s.commonMulticolorShapesNoColor) for s in self.trainSamples)
        # self.nCommonInOutMulticolorDShapesNoColor = min(len(s.commonMulticolorDShapesNoColor) for s in self.trainSamples)

        """
        if len(self.commonInColors) == 1 and len(self.commonOutColors) == 1 and \
        next(iter(self.commonInColors)) == next(iter(self.commonOutColors)):
            self.backgroundColor = next(iter(self.commonInColors))
        else:
            self.backgroundColor = -1
        """

        """
        # Shape features
        self.shapeFeatures = []
        for s in self.trainSamples:
            self.shapeFeatures += s.shapeFeatures
        """

        if self.sameIOShapes:
            self.fixedShapes = []
            for s in self.trainSamples:
                for shape in s.fixedShapes:
                    self.fixedShapes.append(shape)
            self.fixedShapeFeatures = []
            nFeatures = len(self.trainSamples[0].inMatrix.shapes[0].boolFeatures)
            for i in range(nFeatures):
                self.fixedShapeFeatures.append(True)
            for shape in self.fixedShapes:
                self.fixedShapeFeatures = [
                    shape.boolFeatures[i] and self.fixedShapeFeatures[i]
                    for i in range(nFeatures)
                ]

        # Grids:
        self.inputIsGrid = all(
            [s.inMatrix.isGrid for s in self.trainSamples + self.testSamples]
        )
        self.outputIsGrid = all([s.outMatrix.isGrid for s in self.trainSamples])
        self.hasUnchangedGrid = all([s.gridIsUnchanged for s in self.trainSamples])
        if all([hasattr(s, "gridCellIsOutputShape") for s in self.trainSamples]):
            self.gridCellIsOutputShape = all(
                [s.gridCellIsOutputShape for s in self.trainSamples]
            )
        if all([hasattr(s, "gridCellIsInputShape") for s in self.trainSamples]):
            self.gridCellIsInputShape = all(
                [s.gridCellIsInputShape for s in self.trainSamples]
            )
        if self.hasUnchangedGrid:
            self.gridCellsHaveOneColor = all(
                [s.gridCellsHaveOneColor for s in self.trainSamples]
            )
            self.outGridCellsHaveOneColor = all(
                [s.outMatrix.grid.allCellsHaveOneColor for s in self.trainSamples]
            )
        # Asymmetric grids
        self.inputIsAsymmetricGrid = all(
            [s.inMatrix.isAsymmetricGrid for s in self.trainSamples + self.testSamples]
        )
        self.hasUnchangedAsymmetricGrid = all(
            [s.asymmetricGridIsUnchanged for s in self.trainSamples]
        )
        if self.hasUnchangedAsymmetricGrid:
            self.assymmetricGridCellsHaveOneColor = all(
                [s.asymmetricGridCellsHaveOneColor for s in self.trainSamples]
            )
            self.outAsymmetricGridCellsHaveOneColor = all(
                [
                    s.outMatrix.asymmetricGrid.allCellsHaveOneColor
                    for s in self.trainSamples
                ]
            )

        # Background color

        # Is there always a background color? Which one?
        if (
            self.allEqual([s.inMatrix.backgroundColor for s in self.trainSamples])
            and self.trainSamples[0].inMatrix.backgroundColor
            == self.testSamples[0].inMatrix.backgroundColor
        ):
            self.backgroundColor = self.trainSamples[0].inMatrix.backgroundColor
        elif self.hasUnchangedAsymmetricGrid and all(
            [s.inMatrix.asymmetricGrid.nCells > 6 for s in self.trainSamples]
        ):
            self.backgroundColor = self.trainSamples[0].inMatrix.asymmetricGrid.color
            for sample in self.trainSamples:
                sample.inMatrix.backgroundColor = self.backgroundColor
                sample.outMatrix.backgroundColor = self.backgroundColor
            for sample in self.testSamples:
                sample.inMatrix.backgroundColor = self.backgroundColor
        else:
            self.backgroundColor = -1

        self.orderedColors = self.orderColors()

        # Shapes:
        # Does the task ONLY involve changing colors of shapes?
        if self.sameIOShapes:
            self.onlyShapeColorChanges = True
            for s in self.trainSamples:
                nShapes = s.inMatrix.nShapes
                if s.outMatrix.nShapes != nShapes:
                    self.onlyShapeColorChanges = False
                    break
                for shapeI in range(nShapes):
                    if not s.inMatrix.shapes[shapeI].hasSameShape(
                        s.outMatrix.shapes[shapeI]
                    ):
                        self.onlyShapeColorChanges = False
                        break
                if not self.onlyShapeColorChanges:
                    break

            # Get a list with the number of pixels shapes have
            if self.onlyShapeColorChanges:
                nPixels = set()
                for s in self.trainSamples:
                    for shape in s.inMatrix.shapes:
                        nPixels.add(shape.nPixels)
                self.shapePixelNumbers = list(nPixels)

        # R: Are there any common input shapes accross samples?
        self.commonInShapes = []
        for sh1 in self.trainSamples[0].inMatrix.shapes:
            if sh1.color == self.trainSamples[0].inMatrix.backgroundColor:
                continue
            addShape = True
            for s in range(1, self.nTrain):
                if not any(
                    [
                        sh1.pixels == sh2.pixels
                        for sh2 in self.trainSamples[s].inMatrix.shapes
                    ]
                ):
                    addShape = False
                    break
            if addShape and sh1 not in self.commonInShapes:
                self.commonInShapes.append(sh1)

        self.commonInDShapes = []
        for sh1 in self.trainSamples[0].inMatrix.dShapes:
            if sh1.color == self.trainSamples[0].inMatrix.backgroundColor:
                continue
            addShape = True
            for s in range(1, self.nTrain):
                if not any(
                    [
                        sh1.pixels == sh2.pixels
                        for sh2 in self.trainSamples[s].inMatrix.dShapes
                    ]
                ):
                    addShape = False
                    break
            if addShape:
                self.commonInDShapes.append(sh1)
        # Does the task use the information of isolated pixels?
        # if all(s.inMatrix.nIsolatedPixels)
        # Does the input always consist in two shapes?
        self.twoShapeTask = (False, False, False, False, 1)
        if all(len(s.inMatrix.multicolorDShapes) == 2 for s in self.trainSamples):
            self.twoShapeTask = (True, True, True, False, 1)
            if all(
                s.inMatrix.multicolorDShapes[0].shape
                == s.inMatrix.multicolorDShapes[1].shape
                for s in self.trainSamples
            ):
                self.twoShapeTask = (True, True, True, True, 1)

        elif all(len(s.inMatrix.multicolorShapes) == 2 for s in self.trainSamples):
            self.twoShapeTask = (True, True, False, False, 1)
            if all(
                s.inMatrix.multicolorShapes[0].shape
                == s.inMatrix.multicolorShapes[1].shape
                for s in self.trainSamples
            ):
                self.twoShapeTask = (True, True, False, True, 1)
        elif all(len(s.inMatrix.dShapes) == 2 for s in self.trainSamples):
            self.twoShapeTask = (True, False, True, False, 1)
            if all(
                s.inMatrix.dShapes[0].shape == s.inMatrix.dShapes[1].shape
                for s in self.trainSamples
            ):
                self.twoShapeTask = (True, False, True, True, 1)
        if self.inputIsGrid:
            if all(s.inMatrix.grid.nCells == 2 for s in self.trainSamples):
                self.twoShapeTask = (True, False, False, True, 2)
        elif all(
            (
                s.inMatrix.shape[0] * 2 == s.inMatrix.shape[1]
                or s.inMatrix.shape[0] == s.inMatrix.shape[1] * 2
            )
            for s in self.trainSamples
        ):
            self.twoShapeTask = (True, False, False, True, 3)

        # Are all output matrices equal mod nonBackgroundColor?
        if self.sameOutShape:
            self.sameOutDummyMatrix = all(
                np.all(
                    self.trainSamples[0].outMatrix.dummyMatrix
                    == s.outMatrix.dummyMatrix
                )
                for s in self.trainSamples
            )
        # Frames
        self.hasFullFrame = all(
            [len(s.inMatrix.fullFrames) > 0 for s in self.trainSamples]
        )
        self.hasPartialFrame = all(
            [len(s.inMatrix.partialFrames) > 0 for s in self.trainSamples]
        )
        # Is the task about filling a blank?
        self.fillTheBlank = all([hasattr(s, "blankToFill") for s in self.trainSamples])

        # Do all output matrices follow a pattern?
        self.followsRowPattern = all(
            [s.followsRowPattern != False for s in self.trainSamples]
        )
        self.followsColPattern = all(
            [s.followsColPattern != False for s in self.trainSamples]
        )
        if self.followsRowPattern:
            self.rowPatterns = [
                s.outMatrix.followsRowPattern() for s in self.trainSamples
            ]
        if self.followsColPattern:
            self.colPatterns = [
                s.outMatrix.followsColPattern() for s in self.trainSamples
            ]

        # Full Borders / Requires vertical-horizontal rotation
        if self.sameIOShapes:
            if self.submission:
                self.hasOneFullBorder = all(
                    [len(s.commonFullBorders) == 1 for s in self.trainSamples]
                )
            else:
                self.hasOneFullBorder = all(
                    [
                        hasattr(s, "commonFullBorders")
                        and len(s.commonFullBorders) == 1
                        for s in self.trainSamples + self.testSamples
                    ]
                )
            self.requiresHVRotation = False
            if not (
                self.allEqual([s.isHorizontal for s in self.trainSamples])
                or self.allEqual([s.isVertical for s in self.trainSamples])
            ):
                self.requiresHVRotation = all(
                    [s.isHorizontal or s.isVertical for s in self.trainSamples]
                )

    def allEqual(self, x):
        """
        x is a list.
        Returns true if all elements of x are equal.
        """
        if len(x) == 0:
            return False
        return x.count(x[0]) == len(x)

    def orderColors(self):
        orderedColors = []
        # 1: Colors that appear in every sample, input and output, and never
        # change. Only valid if t.sameIOShapes
        if self.sameIOShapes:
            for c in self.fixedColors:
                if all([c in sample.inMatrix.colors for sample in self.testSamples]):
                    orderedColors.append(c)
            # 2: Colors that appear in every sample and are always changed from,
            # never changed to.
            for c in self.commonChangedInColors:
                if c not in self.commonChangedOutColors:
                    if all(
                        [c in sample.inMatrix.colors for sample in self.testSamples]
                    ):
                        if c not in orderedColors:
                            orderedColors.append(c)
            # 3: Colors that appear in every sample and are always changed to,
            # never changed from.
            for c in self.commonChangedOutColors:
                if not all(
                    [c in sample.inMatrix.colors for sample in self.trainSamples]
                ):
                    if c not in orderedColors:
                        orderedColors.append(c)
        # 4: Add the background color.
        if self.backgroundColor != -1:
            if self.backgroundColor not in orderedColors:
                orderedColors.append(self.backgroundColor)
        # 5: Other colors that appear in every input.
        for c in self.commonInColors:
            if all([c in sample.inMatrix.colors for sample in self.testSamples]):
                if c not in orderedColors:
                    orderedColors.append(c)
        # 6: Other colors that appear in every output.
        for c in self.commonOutColors:
            if not all([c in sample.inMatrix.colors for sample in self.trainSamples]):
                if c not in orderedColors:
                    orderedColors.append(c)

        # TODO Dealing with grids and frames

        return orderedColors


# %% Setup
# This first cell needs to be executed before doing anything else
import json
import numpy as np
import pandas as pd
from pathlib import Path
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors

import Task
import Utils
from collections import Counter

# Load all the data. It needs to be in the folder 'data'
#df = pd.read_csv('info.csv', index_col=0)

# index is a list containing the 800 ids of the tasks
#index = list(allTasks.keys())


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
    def __init__(self, Candidate1, Candidate2, Candidate3):
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
    b3c = Best3Candidates(c, c, c)

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
