# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

cube_instruct = """You are a virtual expert in solving a 2x2 Pocket Cube. Your task is to restore a scrambled 2x2 Rubik's Cube to its original state. All the given problems can be solved in 1 to 4 moves. You cannot exceed more than 11 moves. Provide the sequence of moves required for the restoration. Please follow the instructions and rules below to complete the solving:
1. A 2x2 Pocket Cube has six faces, namely: [Upper, Front, Bottom, Left, Right, Back] Each consisting of a 2x2 grid of squares, with each square having its own color.
2. Colors in the Cube are represented in numbers: [0, 1, 2, 3, 4, 5]

You must make move to the Cube to achieve a Restored State, not limited to the above one. Note that we just need each face to have same numbers, no matter which face has which color.

5. You are only allowed to use following moves [U, U', U2, R, R', R2, F, F', F2]. 

Now strictly follow the above process to form Restoration Moves.
[STATEMENT]
As initial state of the cube, I have that
[Initial Cube State]
Upper:
4 5
4 4
Front:
5 1
5 0
Down:
0 0
2 0
Left:
1 1
3 2
Right:
2 2
4 3
Back:
3 3
1 5
[Process]:
[Step 1]
[Move] R
[Current Cube State]
Upper:
4 0
4 0
Front:
5 5
0 1
Down:
0 1
2 2
Left:
1 1
3 3
Right:
2 2
4 3
Back:
4 3
5 5
[Step 2]
[Move] U'
[Current Cube State]
Upper:
0 0
4 4
Front:
0 1
0 1
Down:
2 2
2 2
Left:
1 1
3 3
Right:
4 3
4 3
Back:
5 5
5 5
[Step 3]
[Move] F'
[Current Cube State]
Upper:
0 0
0 0
Front:
1 1
1 1
Down:
2 2
2 2
Left:
3 3
3 3
Right:
4 4
4 4
Back:
5 5
5 5

[STATEMENT]
As initial state of the cube, I have that
[Initial Cube State]:
{init_state}
[Process]:
[Step 1]
[Move]
"""
