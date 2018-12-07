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
    letterBoard = after
    before = np.array(before, dtype=np.character).view(np.uint8)
    after = np.array(after, dtype=np.character).view(np.uint8)
    testTiles = np.logical_not(before == after)
    testTiles = np.logical_and(testTiles, (before == ord('?')))
    print("Comparing:")
    print(testTiles)
    # testRow = -1
    # rowCount = 0
    # testCol = -1
    # colCount = 0
    # for i in range(0, 15):
    #     testCount = np.count_nonzero(testTiles[i].astype(np.uint8))
    #     if testCount > rowCount:
    #         testRow = i
    #         rowCount = testCount
    #     testCount = np.count_nonzero(testTiles[:, i].astype(np.uint8))
    #     if testCount > colCount:
    #         testCol = i
    #         colCount = testCount
    # print(rowCount)
    # print(colCount)

    newLetters = []
    for i in range(0, 15):
        for j in range(0, 15):
            index = (i, j)
            if testTiles[index]:
                newLetters.append(index)

    potentialWords = []
    for (i, j) in newLetters:
        lowI = i
        while lowI > 0 and letterBoard[lowI - 1][j] != '?':
            lowI -= 1
        highI = i
        while highI < 15 and letterBoard[highI][j] != '?':
            highI += 1
        if highI - lowI > 1:
            newWord = []
            for newI in range(lowI, highI):
                newWord.append((newI, j))
            potentialWords.append(newWord)

        lowJ = j
        while lowJ > 0 and letterBoard[i][lowJ - 1] != '?':
            lowJ -= 1
        highJ = i
        while highJ < 15 and letterBoard[i][highJ] != '?':
            highJ += 1
        if highJ - lowJ > 1:
            newWord = []
            for newI in range(lowJ, highJ):
                newWord.append((i, highJ))
            potentialWords.append(newWord)

    # Prune out duplicates in potential words
    words = []
    for potentialWord in potentialWords:
        # Flag to see if new word is already in list
        outerFlag = True
        for word in words:
            # Flag to compare new word to each word
            flag = True
            for letter in newWord:
                flag &= (letter in word)
            if flag:
                outerFlag = False
        # If flag is false, then the word already exists
        if outerFlag:
            words.append(newWord)

    print(words)

    score = 0
    for word in words:
        wordScore = 0
        multiplier = 1
        for (i, j) in word:
            if (i, j) in newLetters:
                if board[i][j] == 'd':
                    wordScore += 2 * letterScore[letterBoard[i][j]]
                elif board[i][j] == 't':
                    wordScore += 3 * letterScore[letterBoard[i][j]]
                else:
                    wordScore += letterScore[letterBoard[i][j]]

                if board[i][j] == 'D':
                    multiplier *= 2
                elif board[i][j] == 'T':
                    multiplier *= 3
            else:
                wordScore += letterScore[letterBoard[i][j]]
        wordScore *= multiplier
        score += wordScore
    return score