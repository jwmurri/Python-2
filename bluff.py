from random import randint
import math

singularnums = {1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six'}
pluralnums = {1:'ones', 2:'twos', 3:'threes', 4:'fours', 5:'fives', 6:'sixes'}

printmode = True

def fact(n):
    '''
    n: an integer. returns factorial of n
    '''
    assert type(n) == int
    if n == 0:
        return 1
    return fact(n-1) * n
    
def nCr(n, r):
    '''
    n: number of options
    r: number of choices
    returns nCr, the number of combinations
    '''
    return fact(n) / (fact(r)*fact(n-r))
    
def binomProb(n, r, p):
    '''
    n: number of experiments
    r: number of successes desired
    p: float, probability of success for each experiment
    returns probability of r successes in n experiments
    '''
    return nCr(n,r)*(p**r)*((1-p)**(n-r))

def binomProbCumul(n, r, p):
    '''
    n: number of experiments
    r: number of successes desired
    p: float, probability of success for each experiment
    returns probability of at least r successes in n experiments
    '''
    prob = 0.0
    for i in range(r, n+1):
        prob += binomProb(n,i,p)
    return prob

def rollHand(n):
    '''
    n: int. Rolls a hand of n 6-sided dice
    returns dict: value --> num of dice with value
    '''
    hand = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    for i in range(n):
        hand[randint(1,6)] += 1
    return hand
    
def handSize(hand):
    '''
    Determines the size of a hand.
    '''
    count = 0
    for key in hand.keys():
        count += hand[key]
    return count

def probGuessIsRight(ndice, guess, hand, printmode=False):
    '''
    ndice: total number of dice
    guess: tuple; (num_val, val)
    hand: dict; value --> num of dice with value
    '''
    size = handSize(hand) # number of dice in player's own hand
    adjnum = 0 # number of dice other players must have to meet guess
    prob = 0 # probability that guess is correct
    
    # Probability readout:
    if printmode:
        if guess[0] == 1:
            print 'Probability that there is 1 ' + singularnums[guess[1]],
        else:
            print 'Probability that there are ' + str(guess[0]) + ' ' + pluralnums[guess[1]],
        print 'in ' + str(ndice) + ' dice.'
        
    if guess[1] == 1: # if the guess is about ones:
        adjnum = guess[0] - hand[1] # players are required to have all the ones that the player does not have
        prob = binomProbCumul(ndice - size, adjnum, 1.0/6.0) # required probability given p(roll one) = 1/6
        
        if printmode:
            print 'Player has ' + str(hand[1]),
            if not hand[1] == 1:
                print 'ones',
            else:
                print 'one',
            print ', so other players must have at least ' + str(adjnum)
            if adjnum == 1:
                print singularnums[1] + '.'
            else:
                print pluralnums[1] + '.'
            print 'Probability of this occuring is ' + str(prob) + '.'
            
    else: # if the guess is not about ones:
        adjnum = guess[0] - (hand[guess[1]] + hand[1]) # players are required to have all the dice the player doesn't
        def getProbAtNum(x):
            result = 0
            for i in range(x + 1):
                result += binomProb(ndice - size, x - i , 1.0/6.0) * binomProb(ndice - size, i, 1.0/6.0)
            return result
        for i in range(adjnum, ndice - size + 1):
            prob += getProbAtNum(i)
        
        if printmode:
            print 'Player has ' + str(hand[1]),
            if not hand[1] == 1:
                print 'ones',
            else:
                print 'one',
            print 'and '+ str(hand[guess[1]]),
            if hand[guess[1]] == 1:
                print singularnums[guess[1]],
            else:
                print pluralnums[guess[1]],
            print ', so other players must have at least ' + str(adjnum),
            if adjnum == 1:
                print singularnums[1] + ' and ' + singularnums[guess[1]] + '.'
            else:
                print pluralnums[1] + ' and ' + pluralnums[guess[1]] + '.'
            print 'Probability of this occuring is ' + str(prob) + '.'
                
    return prob
        
    

def makeInitGuessv1(nplayers, ndice, hand, targetrisk):
    '''
    nplayers: int, number of players currently playing the game
    ndice: total number of dice in the game
    hand: dict: value --> num of dice with value
    targetrisk: float between 0 and 1, probability that guess is wrong
    '''
    size = handSize(hand)
    expected = (ndice - size)/3.0
    max_quantity = 0
    for key in range(2,7):
        if hand[key] >= max_quantity:
            max_quantity = hand[key]
    maxquantity += hand[1]
    
        
    
    