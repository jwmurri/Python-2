from math import *
from decimal import *
from random import *
from time import clock

def place(string):
    position = 40-(len(string)/2)
    line = '###' + '-'*(position-3)
    line += string
    line += '-'*(77-len(line)) + '###'
    assert len(line)==80
    return line

def timef(func, arg):
    begin = clock()
    func(arg)
    time1 = clock() - begin
    print('func took', time1, 's')
    return time1
    
def isInt(l):
    for i in l:
        if type(i)!=int and type(i)!=long:
            return False
    return True
    
def randommag(n):
    return int(10**(n-1+random()))

###-------------------------------DIVISIBILITY-------------------------------###

from integers import isInt
from math import *
from polynomials import *

def rem(a,b):
    '''
    Gives the remainder r obtained by dividing a by b. 0 <= r < b
    ex. rem(5,4) -> 1
    '''
    if type(a)==int:
        return a%b
    
    
def quo(a,b):
    '''
    Gives the integer quotient obtained by dividing a by b.
    ex. quo(5,2) -> 2
    '''
    if type(a)==int:
        return a/b
    
        
def qr(a,b):
    if type(a)==int:
        return (a/b,a%b)


def gcd(a,b):
    '''
    Returns the greatest common divisor of a and b, obtained using the 
    Euclidean algorithm.
    ex. gcd(2501,1891) -> 61
    '''
    while b != 0:
        q = quo(a,b)
        r = rem(a,b)
        a = b
        b = r
    return abs(a)
    
        
        
def lcm(a,b):
    '''
    Returns the least common multiple of a and b.
    '''
    if a != 0 and b != 0:
        return abs((a*b)/gcd(a,b))
    elif a != 0:
        return abs(a)
    elif b != 0:
        return abs(b)
    else:
        return 0
    
def multgcd(l):
    '''
    Returns the gcd of all the integers in the list l.
    '''
    if len(l)==2:
        return gcd(l[0], l[1])
    else:
        return gcd(l[0], multgcd(l[1:]))
        
def multlcm(l):
    '''
    Returns the lcm of all the integers in the list l.
    '''
    if len(l)==2:
        return lcm(l[0], l[1])
    else:
        return lcm(l[0], multlcm(l[1:]))
        
        
    
def axby(a,b):
    '''
    Returns (x,y) such that ax+by = gcd(a,b).
    '''
    c = gcd(a,b)
    assert isInt([a,b])
    x = 0
    y = 0
    while a*x+b*y != c:
        if a*x+b*y < c:
            if randint(0,1)==0:
                x += 1
            else:
                y += 1
        else:
            if randint(0,1)==0:
                x -= 1
            else:
                y -= 1
    return (x,y)
    
def diophantine(a,b,c):
    '''
    Returns integers (x,y) such that ax+by = c.
    '''
    assert isInt([a,b,c])
    if rem(c, gcd(a,b))!=0:
        return 'No solutions'
    x = 0
    y = 0
    while a*x+b*y != c:
        if a*x+b*y < c:
            if randint(0,1)==0:
                x += 1
            else:
                y += 1
        else:
            if randint(0,1)==0:
                x -= 1
            else:
                y -= 1
    return (x,y)
    
def diophantinePos(a,b,c):
    '''
    Returns positive integers (x,y) such that ax+by = c.
    '''
    assert isInt([a,b,c])
    if rem(c, gcd(a,b))!=0:
        return 'No solutions'
    x = 0
    y = quo(c,b)
    while a*x+b*y != c:
        if x>quo(c,a) or y<0:
            return 'No solutions'
        if a*x+b*y < c:
            x += 1
        else:
            y -= 1
        
    return (x,y)
    

def indivisible(p, l):
    '''
    Returns True if none of the elements in l (a list) divide p.
    ex. indivisible(4, [3,5,8]) -> True
    '''
    assert isInt([p]) and type(l)==list
    for i in l:
        if p%i==0:
            return False
    return True
    
###-------------------------------COMBINATORICS------------------------------###

def fact(n):
    '''
    Returns n factorial.
    ex. fact(4) -> 24
    '''
    assert isInt([n])
    if n==0 or n==1:
        return 1
    return n*fact(n-1)
    
def choose(n,k):
    '''
    Returns the number of combinations of n objects taken r at a time.
    ex. choose(6,4) -> 15
    '''
    assert isInt([n,k])
    return fact(n)/(fact(k)*fact(n-k))
    
def permute(n,k):
    '''
    Returns the number of permutations of n objects taken k at a time.
    ex. permute(6,4) -> 360
    '''
    assert isInt([n,k])
    return fact(n)/fact(n-k)
    
###-------------------------------NUMBER THEORY------------------------------###

def isPrime(p):
    '''
    Returns True iff p is prime. 
    ex. isPrime(2**31-1) -> True
    '''
    assert isInt([p]) and p > 1
    return indivisible(p, range(2,int(sqrt(p)+1)))
    
def primesToN(n):
    '''
    Returns a list of all primes <= n.
    '''
    assert isInt([n]) and n > 1
    l = []
    for i in range(2,n+1):
        if isPrime(i):
            l.append(i)
    return l

def factorization(n):
    '''
    Returns a list of prime factors of n in ascending order. 
    ex. factorization(12) -> [2,2,3]
    '''
    assert isInt([n]) and n > 1
    primeList = primesToN(n)
    factors = []
    for p in primeList:
        while n%p==0:
            n /= p
            factors.append(p)
    return factors
    
def factorlist(n):
    '''
    Returns all integers which divide n.
    '''
    assert isInt([n])
    factorlist = []
    for i in range(n+1):
        if i != 0:
            if rem(n,i) == 0:
                factorlist.append(i)
                factorlist.append(-i)
    return factorlist
    
def factordict(n):
    '''
    Returns a dictionary mapping the prime factors of n to their multiplicity.
    ex. factordict(360) -> {2:3, 3:2, 5:1}
    '''
    assert isInt([n]) and n > 1
    primeList = primesToN(n)
    multiplicity = {}
    for p in primeList:
        while n%p==0:
            n /= p
            if p not in multiplicity.keys():
                multiplicity[p] = 1
            else:
                multiplicity[p] += 1
    return multiplicity
    
def ord(n,r):
    '''
    Returns multiplicative order of n (mod r), (O_r(n)), the smallest integer 
    k such that n^k = 1 (mod r).
    ex. ord(4,7) -> 3 because 4^3 = 1 (mod 7)
    '''
    assert all([isInt([n,r]), gcd(n,r)==1, r > 1])
    k = 1
    while (n**k)%r != 1:
        k += 1
    return k
    
def totient(n):
    '''
    Returns Euler's totient function of n, the number of positive integers <n 
    which are relatively prime to n.
    ex. totient(9) -> 6
    '''
    assert isInt([n]) and n > 1
    num = 0
    for i in range(1,n+1):
        if gcd(n,i)==1:
            num += 1
    return num
    
def coprime(a,b):
    '''
    Returns True iff a and b are coprime.
    ex. coprime(6,35) -> True
    '''
    assert isInt([a,b])
    return gcd(a,b)==1
            
def smallestcoprime(n,a=1):
    '''
    Returns smallest number coprime to n which is >a. a defaults to 1.
    '''
    assert isInt([n,a]) and a >= 1
    current = a+1
    while True:
        if coprime(current,n):
            return current
        current += 1
    
def numDivsForm(n):
    '''
    Gives the number of integer divisors >= 1 and <= n of n by brute force.
    '''
    assert isInt([n]) and n > 1
    product = 1
    d = factordict(n)
    for prime in d.keys():
        product *= (d[prime]+1)
    return product
    
###-------------------------------NUMBER BASES-------------------------------###

def stringcheck(string):
    '''
    Returns true iff input is appropriate for base functions below.
    '''
    if type(string)==str:
        for char in string.lower():
            if char not in 'abcdefghijklmnopqrstuvwxyz0123456789':
                return False
        return True
    if type(string)==int or type(string)==long:
        if string > 0:
            return True
        else:
            return False
        
def to10(numstr, b):
    '''
    Converts a string of digits in base b to base 10. A = 10, B = 11, and so on.
    Valid for 1 < b <= 36
    Returns: integer in base 10
    '''
    assert isInt([b]) and 1<b<=36
    if b <= 10:
        if type(numstr)==int or type(numstr)==long:
            numstr = str(numstr)
    assert type(numstr)==str and stringcheck(numstr)
    result = 0
    numstr = numstr.lower()
    index = len(numstr)-1
    chardict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'a':10,'b':11,'c':12,'d':13,'e':14,'f':15,'g':16,'h':17,'i':18,'j':19,'k':20,'l':21,'m':22,'n':23,'o':24,'p':25,'q':26,'r':27,'s':28,'t':29,'u':30,'v':31,'w':32,'x':33,'y':34,'z':35}
    for char in numstr:
        assert chardict[char] < b
        result += chardict[char]*(b**index)
        index -=1
    return result
     
def from10(num, b):
    '''
    Converts a number in base 10 to base b. A = 10, B = 11, and so on.
    Valid for 1 < b <= 36
    Returns: string of digits in base b
    '''
    assert isInt([num,b]) and 1<b<=36  
    result = ''
    charlist = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    n = int(log(num,b))
    index = n
    for i in range(n+1):
        if num - b**index >= 0:
            a = quo(num,b**index)
            result += charlist[a]
            num -= a*(b**index)
        else:
            result += '0'
        index -= 1
    return result
    
def changebase(numstr,a,b):
    '''
    Converts a string of digits in base a to base b. If base is 10, can use int.
    Valid for 1 < a,b <= 36
    Returns: string of digits in base b (or int if b=10)
    '''
    assert all([isInt([a,b]), 1<a<=36, 1<b<=36, stringcheck(numstr)])
    if a==10:
        if type(numstr)==str:
            numstr = int(numstr)
        assert isInt([numstr])
    elif a <= 10:
        if isInt([numstr]):
            numstr = str(numstr)
        assert type(numstr)==str
    else:
        assert type(numstr)==str
        
    if a==10:
        if b==10:
            return numstr
        else:
            return from10(numstr,b)
    else:
        inbase10 = to10(numstr,a)
        if b==10:
            return inbase10
        else:
            return from10(inbase10,b)
            
def baseAdd(x,y,b):
    '''
    Adds the two numbers x and y in base b.
    '''
    assert all([stringcheck(x), stringcheck(y), isInt([b]), 1<b<=36])
    return from10(to10(x,b)+to10(y,b),b)
    
def baseMultiply(x,y,b):
    '''
    Adds the two numbers x and y in base b.
    '''
    assert all([stringcheck(x), stringcheck(y), isInt([b]), 1<b<=36])
    return from10(to10(x,b)*to10(y,b),b)
            
###---------------------------------FIBONACCI--------------------------------###

def fibRecu(n):
    '''
    Returns nth number of fibonacci sequence using the recursive definition.
    ex. fibRecu(10) -> 55
    '''
    assert isInt([n])
    if n==1 or n==2:
        return 1
    return fibRecu(n-1)+fibRecu(n-2)
    
def fibIter(n):
    '''
    Returns nth number of fibonacci sequence using iteration.
    ex. fibIter(10) -> 55
    '''
    assert isInt([n])
    fibList = [0, 1, 1]
    while len(fibList) < n+1:
        fibList.append(fibList[-1]+fibList[-2])
    return fibList[-1]
    
def fibExpl(n):
    '''
    Returns nth number of fibonacci sequence using the explicit definition.
    Accurate until 72nd fib number.
    ex. fibExpl(10) -> 55
    '''
    assert isInt([n])
    phi1 = (1+sqrt(5))/2
    phi2 = (1-sqrt(5))/2
    return int((1/sqrt(5)) * (phi1**n - phi2**n))

def timeFibs(n):
    '''
    Tests and times three fib functions on an input n.
    ex.
    '''
    assert isInt([n])
    
    begin = clock()
    fibRecu(n)
    t1 = clock() - begin
    
    print 'fibRecu took', t1*1000, 'ms'
    
    begin = clock()
    fibIter(n)
    t2 = clock() - begin
    
    print 'fibIter took', t2*1000, 'ms'
    
    begin = clock()
    fibExpl(n)
    t3 = clock() - begin
    
    print 'fibExpl took', t3*1000, 'ms'
    
    if min(t1,t2,t3)==t1:
        print 'fibRecu was fastest by a factor of', min(t2,t3)/t1
        return 'r' 
         
    elif min(t1,t2,t3)==t2:
        print 'fibIter was fastest by a factor of', min(t1,t3)/t2
        return 'i'
        
    else:
        print 'fibExpl was fastest by a factor of', min(t1,t2)/t3
        return 'e'

def testFibs(n, times):
    wins = []
    windict = {}
    for i in range(times)  :
        wins.append(timeFibs(n))
    windict['fibRecu'] = wins.count('r')
    windict['fibIter'] = wins.count('i')
    windict['fibExpl'] = wins.count('e')
    for key in ['fibRecu','fibIter','fibExpl']:
        print key, 'won', str(windict[key])+'/'+str(times), 'times.'
    
    

    
