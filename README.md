# 8puzzlepy
8 puzzle algorithm code in python

**Uninformed Best First Searches**
* breadth-first (using priority queue - criteria for priority is depth)
* depth-first (hits recursion limit)
* depth-limited (with limits of 10, 20, 25, 31)
* iterative-deepening

**Other Uninformed Searches**
* breadth-first (using FIFO queue, same move count, but should be faster)

**Informed Searches**
* A-star (misplaced tiles heuristic)
* A-star (Manhattan distance heuristic)

**Moves per Algorithm**

| **Algorithm**                   | **e1** | **e2** | **e3** | **e4** | **e5** |
|---------------------------------|--------|--------|--------|--------|--------|
| **Breadth First - Best First**  | 23     | 0      | 7      | 20     | 31     |
| **Breadth First**               | 23     | 0      | 7      | 20     | 31     |
| **Depth First**                 | re     | 0      | re     | re     | re     |
| **Depth Limited (10)**          | -1     | 0      | 7      | -1     | -1     |
| **Depth Limited (20)**          | -1     | 0      | 17     | 20     | -1     |
| **Depth Limited (25)**          | 23     | 0      | 21     | 24     | -1     |
| **Depth Limited (31)**          | 31     | 0      | 31     | 30     | 31*    |
| **Iterative Deepening**         | 23     | 0      | 7      | 20     | ?*     |
| **A-star (misplaced tiles)**    | 41     | 0      | 7      | 106    | 103    |
| **A-star (Manhattan distance)** | 31     | 0      | 7      | 106    | 103    |

e1 = EightPuzzle((1, 4, 2, 0, 7, 5, 3, 6, 8))\
e2 = EightPuzzle((1, 2, 3, 4, 5, 6, 7, 8, 0)) // already at goal\
e3 = EightPuzzle((4, 0, 2, 5, 1, 3, 7, 8, 6)) // easy one\
e4 = EightPuzzle((7, 2, 4, 5, 0, 6, 8, 3, 1))\
e5 = EightPuzzle((8, 6, 7, 2, 5, 4, 3, 0, 1)) // max optimal solution is 31 moves\

re: recursion depth reached\
-1: hit limit before finding solution\
\*: take a long-ass time

**Too hard / takes too long to implement**
- bidirectional search

**Branching factor**

0xx
xxx = 2 x 4 cases
xxx

x0x
xxx = 3 x 4 cases
xxx

xxx
x0x = 4 x 1 case
xxx

8 + 12 + 4 = 24 / 9 = 2.667

1 + b + b?? ... max depth of an optimal solution is 31

states = ~2.57 x 10^13


## 4/23/22
1. Determine which algorithms are possible to implement
    - go thru chapter 3 first
2. Determine the data structures needed
3. Prepare to convert to javascript
