arc_instruct = '''You are given a series of inputs and output pairs. 
The values from 'a' to 'j' represent different colors. '.' is a blank cell.
For example, [['.','a','.'],['.','.','b']] represents a 2 row x 3 col grid with color a at position (1,0) and color b at position (2,1).
Coordinates are 2D positions (row, col), row representing row number, col representing col number, with zero-indexing.
Input/output pairs may not reflect all possibilities, you are to infer the simplest possible relation.
You need to reasoning a sequence of Python functions to transform the input grid to the output grid. Now strictly follow the above process to form Python function.
[STATEMENT]
You have the input-output pairs as follows:
[Initial Grid State]:
<init_state>
[Step 1]
[Program]
'''