cube_instruct = """You are a virtual expert in solving a 2x2 Pocket Cube. Your task is to restore a scrambled 2x2 Rubik's Cube to its original state. All the given problems can be solved in 1 to 4 moves. You cannot exceed more than 11 moves. Provide the sequence of moves required for the restoration. Please follow the instructions and rules below to complete the solving:
1. A 2x2 Pocket Cube has six faces, namely: [Upper, Front, Bottom, Left, Right, Back] Each consisting of a 2x2 grid of squares, with each square having its own color.
2. Colors in the Cube are represented in numbers: [0, 1, 2, 3, 4, 5]
3. You must make move to the Cube to achieve a Restored State. Note that we just need each face to have same numbers, no matter which face has which color.
4. A restoration of a Pocket Cube is to move squares in each face to have same numbers.
5. You are only allowed to use following moves [U, U', U2, R, R', R2, F, F', F2]. 

Now strictly follow the above process to form Restoration Moves.

[STATEMENT]
As initial state of the cube, I have that
[Initial Cube State]:
{init_state}
[Process]:
[Step 1]
[Move]
"""
