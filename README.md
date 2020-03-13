# AStar vs HillClimbing

This was done as a part of Artificial Intelligence course.<br />

The problem statement inolved a vaiant of nQueens problem. Some of the queen pieces are very heavy, and therefore difficult to move. We assume here that each column of the chessboard can have only one queen. Queens have a weight ranging from 1 to 9 (integers). The cost to move a queen is the number of tiles you move it times its weight squared. So to move a queen weighing 6 upwards by 4 squares in a column costs 6^2 * 4 = 144. The goal is to find a sequence of moves that help us attain a board configuration with 0 attacking pairs of queens with the least possible cost.
(Note:Attack on a queen includes direct and indirect attack)<br />

For A* following heuristics were defined and tested:<br />
<ul>
<li>H1 : The square of the lightest Queen across all pairs of Queens attacking each other.</li>
<li>H2 : Sum across squared weight of the lightest Queen in every pair of attacking Queens.</li>
<li>H3 : The maximum of minimum cost required to clear attack for each queen. This has been explained in more detail in the report.</li>
</ul>

For Hill Climbing we are using it with sideways movements and random restarts.<br />

The time limit to obtain a result is 10 secs. A comparative study was done between A* and Hill Climbing to see how they perform with in the given time limit and a given board complexity.<br />

A report is also present explaing the process and approach followed during this study.<br />

Instructions to use the python script:<br />
-------------------------

The script needs to be run from the terminal using python3 with the follwing arguments<br />
Arguments : [#_queens] [BoardConfigFile] [Method] [Heuristic] [TimeLimit] <br />
| Argument | Description |
| --- | --- |
| #_queens | No. of queens in the board (>4) |
| BoardConfigFile | .csv file name or random for a random board |
| Method | AS - A Star, HC - Hill Climbing, Both - Both the methods |
| Heuristic | Heuristic number (1/2/3) |
| TimeLimit | Time Limit in secs |

Example command line input:<br />
-------------------------

python3 nQueens_v8_final.py 5 board.csv Both 3 10<br />
>This runs the 'nQueens_v8_final.py' script for a '5x5' board using the 'board.csv' configuration file. 'Both' A* and Hill Climbing algorithms are run with the Heuristic 3 with a time limit of 10 secs<br />

python3 nQueens_v8_final.py 5 random Both 3 10<br />
>This is same as the previous example but uses a randomly generated board<br />



