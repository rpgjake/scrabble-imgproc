import numpy as np

letterScore = {
    'A' : 1,
    'B' : 3,
    'C' : 3,
    'D' : 2,
    'E' : 1,
    'F' : 4,
    'G' : 2,
    'H' : 4,
    'I' : 1,
    'J' : 8,
    'K' : 5,
    'L' : 1,
    'M' : 3,
    'N' : 1,
    'O' : 1,
    'P' : 3,
    'Q' : 10,
    'R' : 1,
    'S' : 1,
    'T' : 1,
    'U' : 1,
    'V' : 4,
    'W' : 4,
    'X' : 8,
    'Y' : 4,
    'Z' : 10,
    ''  : 0
}

board = np.array(
        [['T', '.', '.', 'd', '.', '.', '.', 'T', '.', '.', '.', 'd', '.', '.', 'T'],
         ['.', 'D', '.', '.', '.', 't', '.', '.', '.', 't', '.', '.', '.', 'D', '.'],
         ['.', '.', 'D', '.', '.', '.', 'd', '.', 'd', '.', '.', '.', 'D', '.', '.'],
         ['d', '.', '.', 'D', '.', '.', '.', 'd', '.', '.', '.', 'D', '.', '.', 'd'],
         ['.', '.', '.', '.', 'D', '.', '.', '.', '.', '.', 'D', '.', '.', '.', '.'],
         ['.', 't', '.', '.', '.', 't', '.', '.', '.', 't', '.', '.', '.', 't', '.'],
         ['.', '.', 'd', '.', '.', '.', 'd', '.', 'd', '.', '.', '.', 'd', '.', '.'],
         ['T', '.', '.', 'd', '.', '.', '.', 'D', '.', '.', '.', 'd', '.', '.', 'T'],
         ['.', '.', 'd', '.', '.', '.', 'd', '.', 'd', '.', '.', '.', 'd', '.', '.'],
         ['.', 't', '.', '.', '.', 't', '.', '.', '.', 't', '.', '.', '.', 't', '.'],
         ['.', '.', '.', '.', 'D', '.', '.', '.', '.', '.', 'D', '.', '.', '.', '.'],
         ['d', '.', '.', 'D', '.', '.', '.', 'd', '.', '.', '.', 'D', '.', '.', 'd'],
         ['.', '.', 'D', '.', '.', '.', 'd', '.', 'd', '.', '.', '.', 'D', '.', '.'],
         ['.', 'D', '.', '.', '.', 't', '.', '.', '.', 't', '.', '.', '.', 'D', '.'],
         ['T', '.', '.', 'd', '.', '.', '.', 'T', '.', '.', '.', 'd', '.', '.', 'T']])

def score(before, after):
    testTiles = np.logical_not(np.equal(before, after))
    testTiles = np.logical_and(testTiles, (before == '?'))
    testRow = -1
    rowCount = 0
    testCol = -1
    colCount = 0
    for i in range(0, 15):
        testCount = np.count_nonzero(testTiles[i].astype(np.uint8))
        if testCount > rowCount:
            testRow = i
            rowCount = testCount
        testCount = np.count_nonzero(testTiles[:, i].astype(np.uint8))
        if testCount > colCount:
            testCol = i
            colCount = testCount
    words = []
    if rowCount > colCount:
        indices = []
        for i in range(0, 15):
            if testTiles[testRow, i]:
                indices.append((testRow, i))

                # Get list of new words
                newIndices = []
                for j in range(testRow, 15):
                    if testTiles[j, i]:
                        newIndices.append((j, i))
                    else:
                        break
                for j in range(testRow, -1, -1):
                    if testTiles[j, i]:
                        newIndices.append((j, i))
                    else:
                        break
                if len(newIndices) > 1:
                    words.append(newIndices)
        words.append(indices)
    score = 0
    for word in words:
        wordScore = 0
        multiplier = 1
        for index in word:
            if index in words[0]:
                if board[index] == 'd':
                    wordScore += 2 * letterScore[after[index]]
                elif board[index] == 't':
                    wordScore += 3 * letterScore[after[index]]
                elif board[index] == 'D':
                    multiplier *= 2
                elif board[index] == 'T':
                    multiplier *= 3
        wordScore *= multiplier
        score += wordScore
    return score