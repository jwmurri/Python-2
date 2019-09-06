# -*- coding: utf-8 -*-
# Computational Mathematical Tools v1.0
# by Jacob Murri

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
    print 'func took', time1, 's'
    return time1
    
def isInt(*args):
    ''' Checks if arguments are integers.
    '''
    if type(args[0])==list:
        args = args[0]
    for i in args:
        if type(i)!=int and type(i)!=long:
            return False
    return True
    
def randommag(n):
    return int(10**(n-1+random()))

def rem(a,b):
    ''' a, b integers. Gives the remainder r obtained by dividing a by b. 0 <= r < b.
    '''
    assert isInt(a,b)
    return a%b
    
def quo(a,b):
    ''' a, b integers. Gives the integer quotient obtained by dividing a by b.
    '''
    assert isInt(a,b)
    return a/b
        
def qr(a,b):
    ''' a, b integers. Returns (q,r) (quotient and remainder) such that a = qb + r.
    '''
    assert isInt(a,b)
    return (a/b,a%b)

def gcd(*args):
    ''' args integers. Returns the greatest common divisor of all args using the Euclidean algorithm.
    '''
    if type(args[0])==list:
        args = args[0]
    else:
        args = list(args)
    assert isInt(args)
    if len(args)==2:
        a,b = args[0],args[1]
        while b != 0:
            q,r = qr(a,b)
            a = b
            b = r
        return abs(a)
    else:
        return gcd(args[0], gcd(args[1:]))
        
def lcm(a,b):
    ''' args integers. Returns the least common multiple of all args using the Euclidean algorithm.
    '''
    if type(args[0])==list:
        args = args[0]
    else:
        args = list(args)
    assert isInt(args)
    if len(args)==2:
        a,b = args[0],args[1]
        if a != 0 and b != 0:
            return abs((a*b)/gcd(a,b))
        elif a != 0:
            return abs(a)
        elif b != 0:
            return abs(b)
        else:
            return 0
    else:
        return lcm(args[0], lcm(args[1:]))
    
def axby(a,b):
    ''' a, b integers. Returns (x,y) such that ax+by = gcd(a,b).
    '''
    assert isInt(a,b)
    x,y,c = 0,0,gcd(a,b)
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
    ''' a,b,c integers. Returns integers (x,y) such that ax+by = c.
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
    ''' a,b,c integers. Returns positive integers (x,y) such that ax+by = c.
    '''
    assert isInt(a,b,c)
    if rem(c, gcd(a,b)) != 0:
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
    ''' Returns True if none of the elements in list l divide integer p.
    '''
    assert isInt([p]) and type(l) == list
    return all([p%i!=0 for i in l])

def fact(n):
    ''' n integer. Returns n factorial.
    '''
    assert isInt(n)
    total = 1
    for i in range(1,n+1):
        total *= i
    return total
    
def choose(n,k):
    ''' Returns the number of combinations of n objects taken k at a time.
    '''
    assert isInt(n,k)
    return fact(n)/(fact(k)*fact(n-k))
    
def permute(n,k):
    ''' Returns the number of permutations of n objects taken k at a time.
    '''
    assert isInt(n,k)
    return fact(n)/fact(n-k)

def isPrime(p):
    ''' Returns True iff p is prime. 
    '''
    assert isInt(p) and p > 1
    return indivisible(p, range(2,int(sqrt(p)+1)))
    
def primesToN(n):
    ''' Returns a list of all primes <= n.
    '''
    assert isInt(n) and n > 1
    return [ i for i in range(2,n+1) if isPrime(i) ]

def factorization(n):
    ''' Returns a list of prime factors of n in ascending order. Factors with multiplicity >1 are repeated.
    '''
    assert isInt(n) and n > 1
    primeList = primesToN(n)
    factors = []
    for p in primeList:
        while n%p == 0:
            n /= p
            factors.append(p)
    return factors
    
def factorlist(n):
    ''' Returns all integers which divide n.
    '''
    assert isInt(n)
    factorlist = []
    for i in range(n+1):
        if i != 0:
            if rem(n,i) == 0:
                factorlist.append(i)
                factorlist.append(-i)
    return factorlist
    
def factordict(n):
    ''' Returns a dictionary mapping the prime factors of n to their multiplicity.
    '''
    assert isInt(n) and n > 1
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
    ''' Returns multiplicative order of n (mod r), (O_r(n)), the smallest integer k such that n^k = 1 (mod r).
    '''
    assert all([isInt(n,r), gcd(n,r)==1, r > 1])
    k = 1
    while (n**k)%r != 1:
        k += 1
    return k
    
def totient(n):
    ''' Returns Euler's totient function of n, the number of positive integers < n which are relatively prime to n.
    '''
    assert isInt(n) and n > 1
    num = 0
    for i in range(1,n+1):
        if gcd(n,i) == 1:
            num += 1
    return num
    
def coprime(a,b):
    ''' Returns True iff a and b are coprime.
    '''
    assert isInt(a,b)
    return gcd(a,b) == 1
            
def smallestcoprime(n,a=1):
    '''
    Returns smallest number coprime to n which is > a. a defaults to 1.
    '''
    assert isInt(n,a) and a >= 1
    current = a+1
    while True:
        if coprime(current,n):
            return current
        current += 1
    
def AKSprimality(n):
    ''' Python implementation of a polynomial-time primality tester explored by Agrawal, Kayal, and Saxena.
    '''
    assert isInt(n) and n > 1
    begin = clock()
    print "Beginning at time t =", begin
    
    # Step 1: If n = a^b for a>1 and b>1, n is composite
                
    b = 2
    while b <= log(n,2):
        if abs(n**(1.0/b)-round(n**(1.0/b)))<1e-14:
            a = n**(1.0/b)
            if n==a**b:
                print "Algorithm concluded at time t =", clock()-begin
                return False
        b += 1
    print "Step 1 complete at time t =", clock()-begin
                
    # Step 2: Find the smallest r such that ord(n,r) > (log_2(n))^2
    
    print "Beginning step 2 at time t =", clock()-begin
    r = 2 
    lowbound = (log(n,2))**2
    while True:
        d = gcd(n,r)
        if d != 1 and d != n:
            print 'gcd of n and', d, 'is not OK'
            return False # n is composite
        ordnr = 1
        while (n**ordnr)%r != 1:
            ordnr += 1
        if ordnr <= lowbound: 
            r += 1
        else:
            break # smallest r such that ord(n,r) > (log_2(n))^2 has been found
    print "Step 2 complete at time t =", clock()-begin
    
    # Step 3: If 1 < gcd(a,n) < n for some a <= r, n is composite
    
    for a in range(1,r+1): 
        if 1 < gcd(a,n) < n:
            print "Algorithm concluded at time t =", clock()-begin
            return False
    print "Step 3 complete at time t =", clock()-begin
            
    # Step 4: If n <= r, n is prime
            
    if n<=r:
        print "Algorithm concluded at time t =", clock()-begin
        return True
    print "Step 4 complete at time t =", clock()-begin
        
    # Step 5: If (x+a)^n != x^n + a (mod n) for any 
    #       : 1 <= a <= floor(phi(r)*log_2(n)), n is composite
        
    for a in range(1, int(sqrt(totient(r))*log(n,2))+1):
        for i in range(1,n): # all coefficients of polynomials must be same
            if (choose(n,i)*a**(n-i))%n != 0: 
                print "Algorithm concluded at time t =", clock()-begin
                return False 
    print "Step 5 complete at time t =", clock()-begin
                
    # Step 6: At this point, n must be prime
    
    print "Algorithm concluded at time t =", clock()-begin
    return True 
    
def numDivsForm(n):
    ''' Gives the number of integer divisors >= 1 and <= n of n by brute force.
    '''
    assert isInt(n) and n > 1
    product = 1
    d = factordict(n)
    for prime in d.keys():
        product *= (d[prime]+1)
    return product
    
###-------------------------------NUMBER BASES-------------------------------###

def stringcheck(string):
    ''' Returns true iff input is appropriate for base-change functions below.
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
    ''' Converts a string of digits in base b to base 10. A = 10, B = 11, and so on.
        Valid for 1 < b <= 36
        Returns: integer in base 10
    '''
    assert isInt(b) and 1 < b <= 36
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
    ''' Converts a number in base 10 to base b. A = 10, B = 11, and so on.
        Valid for 1 < b <= 36
        Returns: string of digits in base b
    '''
    assert isInt(num,b) and 1<b<=36  
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
    ''' Converts a string of digits in base a to base b. If base is 10, can use int.
        Valid for 1 < a,b <= 36
        Returns: string of digits in base b (or int if b=10)
    '''
    assert all([isInt(a,b), 1<a<=36, 1<b<=36, stringcheck(numstr)])
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
    ''' Adds the two numbers x and y in base b.
    '''
    assert all([stringcheck(x), stringcheck(y), isInt(b), 1<b<=36])
    return from10(to10(x,b)+to10(y,b),b)
    
def baseMultiply(x,y,b):
    ''' Adds the two numbers x and y in base b.
    '''
    assert all([stringcheck(x), stringcheck(y), isInt(b), 1<b<=36])
    return from10(to10(x,b)*to10(y,b),b)
            
###---------------------------------FIBONACCI--------------------------------###

def fibRecu(n):
    ''' Returns nth number of fibonacci sequence using the recursive definition.
        ex. fibRecu(10) -> 55
    '''
    assert isInt(n)
    if n==1 or n==2:
        return 1
    return fibRecu(n-1)+fibRecu(n-2)
    
def fibIter(n):
    ''' Returns nth number of fibonacci sequence using iteration.
        ex. fibIter(10) -> 55
    '''
    assert isInt(n)
    fibList = [0, 1, 1]
    while len(fibList) < n+1:
        fibList.append(fibList[-1]+fibList[-2])
    return fibList[-1]
    
def fibExpl(n):
    ''' Returns nth number of fibonacci sequence using the explicit definition.
        Accurate until 72nd fib number.
        ex. fibExpl(10) -> 55
    '''
    assert isInt(n)
    phi1 = (1+sqrt(5))/2
    phi2 = (1-sqrt(5))/2
    return int((1/sqrt(5)) * (phi1**n - phi2**n))

def timeFibs(n):
    ''' Tests and times three fib functions on an input n.
    '''
    assert isInt(n)
    
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
    ''' Races the three Fibonacci algorithms
    '''
    wins = []
    windict = {}
    for i in range(times)  :
        wins.append(timeFibs(n))
    windict['fibRecu'] = wins.count('r')
    windict['fibIter'] = wins.count('i')
    windict['fibExpl'] = wins.count('e')
    for key in ['fibRecu','fibIter','fibExpl']:
        print key, 'won', str(windict[key])+'/'+str(times), 'times.'
        
class Rational(object):
    
    def __init__(self, numerator, denominator=1):
        ''' Initializes rational number. 
        '''
        if isinstance(numerator, Rational) or isinstance(denominator, Rational):
            a = numerator/denominator
            self.num = (a.num[0],a.num[1])
        elif isinstance(numerator, Complex) or isinstance(denominator, Complex):
            return ComplexRational(numerator/denominator)
        else:
            assert denominator != 0
            a,b = numerator, denominator
            while  b != 0:
                q = a/b
                r = a%b
                a = b
                b = r
            d = abs(a) # = gcd(numerator,denominator)
            if d != 1:
                numerator /= d
                denominator /= d
            self.num = (numerator, denominator)
            if denominator < 0:
                self.num = (-numerator, -denominator)
            if numerator == 0:
                self.num = (0,1)
            
    real = lambda self: float(self.num[0])/float(self.num[1])
        
    def __str__(self):
        if self.num[1] != 1:
            return str(self.num[0]) + '/' + str(self.num[1])
        else:
            return str(self.num[0])
            
    __repr__ = lambda self: self.__str__()
        
    def __add__(self, other):
        if isinstance(other, Rational):
            return Rational(self.num[0]*other.num[1]+self.num[1]*other.num[0], self.num[1]*other.num[1])
        elif type(other)==int:
            return Rational(self.num[0]+other*self.num[1], self.num[1])
        else:
            return self.real()+other
            
    __radd__ = lambda self, other: self.__add__(other)
        
    def __mul__(self, other):
        if isinstance(other, Rational):
            return Rational(self.num[0]*other.num[0], self.num[1]*other.num[1])
        elif type(other)==int:
            return Rational(self.num[0]*other, self.num[1])
        elif isinstance(other, Polynomial):
            return other*self
        elif isinstance(other, Vector):
            return other*self
        elif isinstance(other, Matrix):
            return other*self
        else:
            return self.real()*other
            
    __rmul__ = lambda self, other: self.__mul__(other)
        
    __sub__ = lambda self, other: self.__add__(other*-1)
        
    __rsub__ = lambda self, other: other + self.__mul__(-1)
        
    def __div__(self, other):
        if isinstance(other, Rational):
            return Rational(self.num[0]*other.num[1], self.num[1]*other.num[0])
        elif type(other)==int:
            return Rational(self.num[0], self.num[1]*other)
        else:
            return self.real()/other
            
    __rdiv__ = lambda self, other: other * Rational(1,1)/self
        
    __neg__ = lambda self: Rational(-self.num[0],self.num[1])
        
    __pos__ = lambda self: self
        
    __abs__ = lambda self: Rational(abs(self.num[0]), self.num[1])
    
    __pow__ = lambda self, other: Rational(self.num[0].__pow__(other), self.num[1].__pow__(other))
                    
    def __lt__(self, other):
        if isinstance(other, Rational):
            return self.num[0]*other.num[1] < self.num[1]*other.num[0]
        else:
            return self.real() < other

    def ___le__(self, other):
        if isinstance(other, Rational):
            return self.num[0]*other.num[1] <= self.num[1]*other.num[0]
        else:
            return self.real() <= other
        
    def __eq__(self, other):
        if isinstance(other, Rational):
            return self.num[0]*other.num[1] == self.num[1]*other.num[0]
        else:
            return self.real() == other

    def __ne__(self, other):
        if isinstance(other, Rational):
            return self.num[0]*other.num[1] != self.num[1]*other.num[0]
        else:
            return self.real() != other

    def __gt__(self, other):
        if isinstance(other, Rational):
            return self.num[0]*other.num[1] > self.num[1]*other.num[0]
        else:
            return self.real() != other

    def __ge__(self, other):
        if isinstance(other, Rational):
            return self.num[0]*other.num[1] >= self.num[1]*other.num[0]
        else:
            return self.real() >= other
            
class Surd(object):
    square_root = '√'
    def __init__(self):
        pass
        
class Complex(object):
    
    def __init__(self, real, imag=0):
        self.real = real
        self.imag = imag
        
    def __str__(self):
        string = ''
        if self.real != 0:
            string += str(self.real)
        if self.imag != 0:
            if self.real != 0:
                if str(self.imag)[0] != '-':
                    string += '+'
            if str(self.imag)[0] == '-':
                    string += '-'
            if abs(self.imag) != 1:
                string += str(abs(self.imag))
            string += 'i'
        if string == '':
            return '0'          
        if self.imag != 0 and self.real != 0:
            string = '(' + string + ')'      
        return string
        
    __repr__ = lambda self: self.__str__()
    
    conj = lambda self: Complex(self.real, -self.imag)
    
    abs2 = mag2 = lambda self: self.real*self.real + self.imag*self.imag
    
    __abs__ = lambda self: sqrt(self.mag2())
        
    def __add__(self, other):
        if isinstance(other, Complex):
            return Complex(self.real+other.real, self.imag+other.imag)
        else:
            return Complex(self.real+other, self.imag)
            
    __radd__ = lambda self, other: self.__add__(other)
        
    def __mul__(self, other):
        if isinstance(other, Complex):
            return Complex(self.real*other.real-self.imag*other.imag, self.imag*other.real+self.real*other.imag)
        else:
            return Complex(self.real*other, self.imag*other)
            
    __rmul__ = lambda self, other: self.__mul__(other)
        
    __sub__ = lambda self, other: self.__add__(other*-1)
        
    __rsub__ = lambda self, other: other + self.__mul__(-1)
        
    def __div__(self, other):
        if isinstance(other, Rational):
            return Complex(Rational(self.real*other.real-self.imag*other.imag, other.mag2()), Rational(self.imag*other.real-self.real*other.imag, other.mag2()))
        else:
            return Complex(Rational(self.real, other), Rational(self.real, other))
            
    def totheminusone(self):
        return Complex(Rational(self.real, self.mag2()), Rational(-self.imag, self.mag2()))
            
    __rdiv__ = lambda self, other: other * self.totheminusone()
        
    __neg__ = lambda self: Complex(-self.real, -self.imag)
        
    __pos__ = lambda self: self
    
    def __pow__(self, other):
        if other == 0:
            return Complex(1)
        else:
            return self*self.__pow__(other-1)
        
    def __eq__(self, other):
        if isinstance(other, Complex):
            return self.real == other.real and self.imag == other.imag
        else:
            return self.real == other and self.imag == 0

    def __ne__(self, other):
        if isinstance(other, Complex):
            return self.real != other.real or self.imag != other.imag
        else:
            return self.real != other or self.imag != 0       
        
class ComplexRational(Complex):
    
    def __init__(self, *args):
        if len(args)==1:
            if isinstance(args[0], Rational):
                self.real = args[0]
                self.imag = Rational(0,1)
            else:
                self.real = Rational(args[0], 1)
                self.imag = Rational(0,1)
        elif len(args)==2:
            if isinstance(args[0], Rational) and isinstance(args[1], Rational):
                self.real = args[0]
                self.imag = args[1]
            elif isinstance(args[0],Rational):
                self.real = args[0]
                self.imag = Rational(args[1], 1)
            elif isinstance(args[1],Rational):
                self.real = Rational(args[0], 1)
                self.imag = args[1]
            else:
                self.real = Rational(args[0],1)
                self.imag = Rational(args[1],1)
        elif len(args)==3:
            if isinstance(args[0], Rational):
                self.real = args[0]
                self.imag = Rational(args[1],args[2])
            elif isinstance(args[2], Rational):
                self.real = Rational(args[0],args[1])
                self.imag = args[2]
            else:
                self.real = Rational(args[0],args[1])
                self.imag = Rational(args[2], 1)
        elif len(args)==4:
            self.real = Rational(args[0],args[1])
            self.imag = Rational(args[2],args[3])
        
    def __str__(self):
        string = ''
        if self.real != 0:
            string += str(self.real)
        if self.imag != 0:
            if self.real != 0:
                if str(self.imag)[0] != '-':
                    string += '+'
            if str(self.imag)[0] == '-':
                    string += '-'
            if abs(self.imag) != 1:
                string += str(abs(self.imag))
            string += 'i'
        if string == '':
            return '0'          
        if self.imag != 0 and self.real != 0:
            string = '(' + string + ')'      
        return string
        
    __repr__ = lambda self: self.__str__()
    
    conj = lambda self: ComplexRational(self.real, -self.imag)
    
    abs2 = mag2 = lambda self: self.real*self.real + self.imag*self.imag
    
    __abs__ = lambda self: sqrt(self.mag2().real())
        
    def __add__(self, other):
        if isinstance(other, ComplexRational):
            return ComplexRational(self.real+other.real, self.imag+other.imag)
        else:
            return ComplexRational(self.real+other, self.imag)
            
    __radd__ = lambda self, other: self.__add__(other)
        
    def __mul__(self, other):
        if isinstance(other, Complex):
            return ComplexRational(self.real*other.real-self.imag*other.imag, self.imag*other.real+self.real*other.imag)
        else:
            return ComplexRational(self.real*other, self.imag*other)
            
    __rmul__ = lambda self, other: self.__mul__(other)
        
    __sub__ = lambda self, other: self.__add__(other*-1)
        
    __rsub__ = lambda self, other: other + self.__mul__(-1)
        
    def __div__(self, other):
        if isinstance(other, ComplexRational):
            return ComplexRational(self.real*other.real+self.imag*other.imag, self.imag*other.real-self.real*other.imag).__div__(other.mag2())
        if isinstance(other, Complex):
            return self.__div__(ComplexRational(other.real, other.imag))
        else:
            return ComplexRational(self.real/other, self.imag/other)
            
    def totheminusone(self):
        return ComplexRational(self.real, -self.imag).__div__(self.mag2())
            
    __rdiv__ = lambda self, other: other * self.totheminusone()
        
    __neg__ = lambda self: Complex(-self.real, -self.imag)
        
    __pos__ = lambda self: self
    
    def __pow__(self, other):
        if other == 0:
            return Complex(1)
        else:
            return self*self.__pow__(other-1)
        
    def __eq__(self, other):
        if isinstance(other, ComplexRational) or isinstance(other, Complex):
            return self.real == other.real and self.imag == other.imag
        else:
            return self.real == other and self.imag == 0

    def __ne__(self, other):
        if isinstance(other, ComplexRational) or isinstance(other, Complex):
            return self.real != other.real or self.imag != other.imag
        else:
            return self.real != other or self.imag != 0       
        
Real = real = Re = re = lambda z: z.real
Imag = imag = Im = im = lambda z: z.imag

Conj = conj = lambda z: z.conj()

def newton(F, f, epsilon=0.01, guess=0, time=1):
    '''
    F, f functions such that f is the derivative of F. Finds a zero (within epsilon) of F. By default will stop after 1 sec.
    '''
    begin = clock()
    current_val = guess
    while abs(F(current_val)) >= epsilon:
        current_val -= float(F(current_val))/float(f(current_val))
        if clock()-begin >= 1:
            return "answer not found"
    return current_val

class Polynomial(object):
    
    def __init__(self, *args):
        
        if type(args[0]) == list:
            self.coeffs = args[0]
        else:
            self.coeffs = list(args)
        while self.coeffs[-1] == 0 and len(self.coeffs) > 1:
            self.coeffs = self.coeffs[:-1]
        self.degree = len(self.coeffs)-1
        
    get = lambda self: self.coeffs
    # Get the internal coordinate list of a vector for easier manipulation.
    
    deg = lambda self: self.degree
        
    def __str__(self, dummy='x'):
        ''' Get the string representation of the polynomial with dummy variable x.
        '''
        string = ''
        x = dummy
        for i in range(self.degree+1)[::-1]:
            coeff = self.coeffs[i]
            if str(coeff)[0] == '-':
                if i == self.degree:
                    string += '-'
                else:
                    string += ' - '
            else:
                if i != self.degree and coeff != 0:
                    if str(coeff)[0] != '-':
                        string += ' + '
            if coeff != 0:
                if abs(coeff) != 1 or i == 0:
                    string += str(abs(coeff))
                if i != 0:
                    string += x
                    if i != 1:
                        string += '^'+str(i)     
        if string == '':
            return '0'                
        return string
        
    __repr__ = lambda self: self.__str__()
        
    def __add__(self, other):
        ''' Adds two polynomials. Assumes that deg self >= deg other.
        '''
        if isinstance(other, Polynomial):
            if self.degree >= other.degree:
                l = []
                for i in range(self.degree+1):
                    if i > other.degree:
                        l.append(self.get()[i])
                    else:
                        l.append(self.get()[i]+other.get()[i])
                return Polynomial(l)
            else:
                return other.__add__(self)
        else:
            l = self.coeffs[:]
            l[0] = l[0] + other
            return Polynomial(l)
        
    __radd__ = lambda self, other: self.__add__(other)
        
    def __mul__(self, other):
        ''' Returns product of self and other.
        '''
        if isinstance(other, Polynomial):
            l = [ 0 for i in range(self.degree+other.degree+2) ]
            for i in range(self.degree+1):
                for j in range(other.degree+1):
                    l[i+j] = l[i+j] + self.coeffs[i]*other.coeffs[j]
            return Polynomial(l)
        else:
            return Polynomial([ self.get()[i]*other for i in range(self.degree+1) ])
            
    __rmul__ = lambda self, other: self.__mul__(other)
    
    __sub__ = lambda self, other: self.__add__(other.__mul__(-1))
        
    __rsub__ = lambda self, other: other.__add__(self.__mul__(-1))
        
    __getitem__ = lambda self, i: self.coeffs[i]

    __iter__ = lambda self: self.coeffs 
    
    __eq__ = lambda self, other: all([ (self-other).get()[i] == 0 for i in range((self-other).degree+1) ])
    
    __neq__ = lambda self, other: not self.__eq__(other)
    
    leadingcoeff = lambda self: self.coeffs[self.degree]
    
    def monic(self):
        if self == 0:
            return self
        elif self.leadingcoeff() == 1:
            return self
        else:
            return self*Rational(1, self.leadingcoeff())
    
    def __abs__(self):
        print 'WHY IS IT CALLING THIS'
        print self
        return self
    
    def makemonic(self):
        self = self.monic()
        
    def eval(self, x):
        result = 0
        for index,value in enumerate(self.coeffs):
            result += value*(x**index)
        return result

    def derivative(self):
        l = [ 0 for i in range(self.degree) ]
        for index,coeff in enumerate(self.coeffs):
            if index != 0:
                l[index-1] = value*index
        return Polynomial(l)
        
    def qr(self, other):
        a = self
        b = other
        assert b != 0
        (q,r) = (Polynomial(0),a) # at each step a = b * q + r
        while r != 0 and deg(r) >= deg(b):
            term = singleterm(deg(r)-deg(b))*Rational(leadingcoeff(r),leadingcoeff(b)) # divide the leading terms
            q = q + term
            r = r - (term * b)
        return (q,r)
        
    quo = lambda self, other: self.qr(other)[0]
        
    rem = lambda self, other: self.qr(other)[1]
    
    def gcd(self,other):
        a = self
        b = other
        while not b == 0:
            q = a.quo(b)
            r = a.rem(b)
            print a, b, q, r
            a = b
            b = r
        return monic(a)
    
    def __div__(self,other):
        if self.rem(other) == 0:
            return self.quo(other)
        
deg = lambda p: p.degree
        
monic = lambda p: p.monic()

lead = leadingcoeff = lambda p: p.coeffs[p.degree]

polygcd = lambda p, q: p.gcd(q)

def polygcd(*args):
    ''' args polynomials. Returns the greatest common divisor of all args using the division algorithm.
    '''
    if type(args[0])==list:
        args = args[0]
    else:
        args = list(args)
    if len(args)==2:
        return args[0].gcd(args[2])
    else:
        return polygcd(args[0], polygcd(args[1:]))
        
def polylcm(a,b):
    ''' args polynomials. Returns the least common multiple of all args using the division algorithm.
    '''
    if type(args[0])==list:
        args = args[0]
    else:
        args = list(args)
    assert isInt(args)
    if len(args)==2:
        a,b = args[0],args[1]
        if a != 0 and b != 0:
            return monic((a*b)/polygcd(a,b))
        elif a != 0:
            return monic(a)
        elif b != 0:
            return monic(b)
        else:
            return 0
    else:
        return lcm(args[0], lcm(args[1:]))

def singleterm(d):
    l = [0 for i in range(d+1)]
    l[d] = 1
    return Polynomial(l)

def factors(p, c=False, analytical=False):
    if deg(p) == 0:
        return []
    elif deg(p) == 1:
        return [p]
    elif deg(p) == 2 and analytical:
        c,b,a = tuple(p.coeffs)
        if b**2 - 4*a*c < 0 and sqrt(b**2-4*a*c) == int(sqrt(b**2-4*a*c)):
            if c:
                root1 = Complex(Rational(-b,2*a), Rational(int(sqrt(b**2-4*a*c), 2*a)))
                return [Polynomial(-root1,1), Polynomial(-conj(root1),1)]
            else:
                return [p]
        elif b**2 - 4*a*c == 0:
            root1 = Rational(-b, 2*a)
            return [Polynomial(-root1,1),Polynomial(-root1,1)]
        elif sqrt(b**2-4*a*c) == int(sqrt(b**2-4*a*c)):
            root1 = Rational(-b, 2*a) + Rational(int(sqrt(b**2-4*a*c), 2*a))
            root2 = Rational(-b, 2*a) - Rational(int(sqrt(b**2-4*a*c), 2*a))
            return [Polynomial(-root1,1),Polynomial(-root2,1)]
        else:
            raise NotImplementedError
    else:
        roots = []
        leading = lead(p)
        constant = p[0]
        for a in factorlist(constant):
            for b in factorlist(leading):
                if p.eval(Rational(a,b)) == 0:
                    if all([ not factor == Polynomial(-Rational(a,b),1) for factor in roots ]):
                        roots.append(Polynomial(-Rational(a,b),1))
        return roots
        
# LINEAR ALGEBRA

# Basic utilities --------------------------------------------------------------

ltoarray = lambda n, m, l: [ [ l[j] for j in range(m*i, m*(i+1)) ] for i in range(n) ]
# Converts a one-dimensional len n*m list l to a nested list n*m array structure.
    
arraytol = lambda array: [ array[i][j] for i in range(len(array)) for j in range(len(array[0])) ] 
# Converts a nested list n*m array structure into a one-dimensional len n*m list.
    
flip = lambda array: [ [ array[i][j] for i in range(len(array)) ] for j in range(len(array[0])) ]
# Transposes an n*m array to an m*n array.
    
class Vector(object):
    ''' Vector class that stores vectors internally as a list called self.vector.
    '''
    def __init__(self, *args):
        ''' Initializes vector with coordinates given by coorlist. Dimensionality is implied by length of coorlist.
        '''
        if type(args[0]) == list:
            self.coords = args[0]
        else:
            self.coords = list(args)
        self.dim = len(self.coords)
        
    get = lambda self: self.coords
    # Get the internal coordinate list of a vector for easier manipulation.
        
    def __str__(self):
        ''' Get the string representation of the vector with angular braces.
        '''
        string = ''
        for i in range(self.dim):
            string += str(self.coords[i])
            if i != self.dim - 1: # add commas for all except last coordinate
                string += ', '                            
        return '<'+string+'>' 
        
    __repr__ = lambda self: self.__str__()
        
    def __add__(self, other):
        ''' Adds a vector to another vector.
        '''
        assert self.dim == other.dim
        return Vector([ self.get()[i] + other.get()[i] for i in range(self.dim) ])
        
    __radd__ = lambda self, other: self.__add__(other)
        
    def __mul__(self, other):
        ''' If other is a vector, takes dot product. If other is a scalar, multiplies by scalar.
        '''
        if isinstance(other, Vector):
            assert self.dim == other.dim
            return sum([ self.get()[i] * other.get()[i] for i in range(self.dim) ])
        else:
            return Vector([ self.get()[i]*other for i in range(self.dim) ])
            
    def __rmul__(self, other):
        ''' If other is a vector, takes dot product. If other is a scalar, multiplies by scalar.
        '''
        if isinstance(other, Vector):
            assert self.dim == other.dim
            return sum([ self.get()[i] * other.get()[i] for i in range(self.dim) ])
        else:
            return Vector([ self.get()[i]*other for i in range(self.dim) ])

    __div__ = lambda self, other: Vector([ Rational(self.get()[i],other) for i in range(self.dim) ])

    __sub__ = lambda self, other: self.__add__(other.__mul__(-1))
        
    __rsub__ = lambda self, other: other.__add__(self.__mul__(-1))
        
    __len__ = lambda self: self.dim
        
    __getitem__ = lambda self, i: self.coords[i]

    __iter__ = lambda self: self.coords
        
    __reversed__ = lambda self: reversed(self.coords)
        
    __getslice__ = lambda self, i, j: Vector(self.coords[i:j]) 
    
    __eq__ = lambda self, other: all([ (self-other).get()[i] == 0 for i in range(self.dim) ])      

def cross(a, b):
    ''' Takes the cross product of two 3-dim vectors a and b.
    '''
    assert a.dim == b.dim == 3
    (a1, a2, a3) = (a.get()[0], a.get()[1], a.get()[2])
    (b1, b2, b3) = (b.get()[0], b.get()[1], b.get()[2])
    return Vector([a2*b3-a3*b2, a3*b1-a1*b3, a1*b2-a2*b1])
    
mag = lambda v: sqrt(float(sum([v.get()[i]**2 for i in range(v.dim)])))
# Returns magnitude of a vector v.

iszero = lambda v: all([ v.get()[i] == 0 for i in range(v.dim) ])
# Returns true if vector v is the zero vector
    
def only1(v):
    ''' Returns true if vector v has only one 1 in its entries, all rest are zero.
    '''
    num_ones = 0
    for entry in v.get():
        if entry == 1:
            num_ones += 1
        elif entry != 0:
            return False
    return num_ones == 1
    
def numzeros(v):
    ''' Returns the number of zeros in the vector.
    '''
    num = 0
    for entry in v.get():
        if entry == 0:
            num += 1
    return num
    
costheta = lambda a, b: float(a*b)/(mag(a)*mag(b))
# Returns cosine of angle between two equal dimensional vectors a and b.

def sintheta(a, b):
    ''' Returns sine of angle between two vectors a and b.
    '''
    assert a.dim == b.dim
    if a.dim == 3:
        return float(cross(a, b))/(mag(a)*mag(b))
    return sqrt(1-costheta(v1,v2)**2)
    
theta = lambda a, b : acos(costheta(v1, v2))
# Returns angle (in radians) between two equal dimensional vectors a and b.
        
class Matrix(Vector):
    ''' Matrix class that stores matrices internally as two dimensions and a list, and also a list of lists.
    '''
    
    def __init__(self,*args):
        ''' Creates a matrix.
            Method 1: args: nested list. Will create matrix where rows are the sublists.
            Method 2: args: list, rows, cols. Will create rows*cols matrix where the entries are in list. Make sure len(l) = rows*cols
            Method 3: Like method 2, but the entries are not in a list. So args: entry1, entry2, ... , entryn, rows, cols. Make sure n = rows*cols.
        '''
        
        if type(args[0]) == list:
            l = args[0]
            if type(l[0]) == list:
                self.rows = len(l)
                self.cols = len(l[0])
                assert len(l) == self.rows and len(l[0]) == self.cols
                for row in l:
                    assert len(row) == len(l[0])
                self.list = arraytol(l)
                self.array = l
            
            else:
                assert type(args[1]) == type(args[2]) == int
                assert len(l) == args[1]*args[2]
                self.rows = args[1]
                self.cols = args[2]
                self.list = l
                self.array = ltoarray(self.rows, self.cols, l)
        else:
            l = list(args)
            assert len(l) == (l[-2]*l[-1])+2 and type(l[-1]) == type(l[-2]) == int
            self.rows = l[-2]
            self.cols = l[-1]
            self.list = l[:-2]
            self.array = ltoarray(self.rows, self.cols, self.list)
            
    def fractionize(self):
        ''' Makes all the elements in a matrix complex rational numbers.
        '''
        temp = self.list
        self.list = []
        for i in temp:
            if isinstance(i, ComplexRational):
                self.list.append(i)
            elif isinstance(i, Rational):
                self.list.append(ComplexRational(i,0))
            elif isinstance(i, Complex):
                self.list.append(ComplexRational(i.real, i.imag))
            else:
                self.list.append(ComplexRational(i,1,0))
        self.array = ltoarray(self.rows, self.cols, self.list)
        
    def defractionize(self):
        ''' Un-rationalizes everything.
        '''
        temp = []
        for n in self.list:
            if isinstance(n, ComplexRational):
                if n.real.num[1] == 1 and n.imag.num[1] == 1:
                    temp.append(Complex(n.real.num[0], n.imag.num[0]))
                elif n.real.num[1] == 1:
                    temp.append(Complex(n.real.num[0], n.imag.real()))
                elif n.imag.num[1] == 1:
                    temp.append(Complex(n.real.real(), n.imag.num[0]))
                else:
                    temp.append(Complex(n.real.real(), n.imag.real()))
            elif isinstance(n, Rational):
                if n.num[1] == 1:
                    temp.append(n.num[0])
                else:
                    temp.append(n.real())
        self.list = temp
        self.array = ltoarray(self.rows, self.cols, self.list)
            
    def __str__(self):
        ''' Prepares a matrix for printing
        '''
        total = ''
        colwidth = max([ len(str(self.list[i])) for i in range(self.rows*self.cols) ])
        for i in range(self.rows):
            string = ''
            for j in range(self.cols):
                partial = str(self.array[i][j])
                partial += ' '*(colwidth-len(str(self.array[i][j])))
                if j!=self.cols-1:
                    partial += ', '
                string += partial
            total += '|'+string+'|\n'
        return total
        
    copy = lambda self: Matrix(self.array[:])
    # Returns an exact copy of the matrix.
            
    #__str__ = lambda self: str(self.array)
    # String representation of matrix as list of lists.
        
    __repr__ = lambda self: self.__str__()
    # Representation of matrix as list of lists.
        
    def getrow(self, row):
        ''' Slices a row out of a matrix. Row number > 0. Returns vector.
        '''
        assert 0 < row <= self.rows
        return Vector(self.array[row-1])
        
    def getcol(self, col):
        ''' Slices a column out of a matrix. Col number > 0. Returns vector.
        '''
        assert 0 < col <= self.cols
        return Vector([self.array[i][col-1] for i in range(self.rows)])
        
    getrows = lambda self: [ self.getrow(i+1) for i in range(self.rows) ]
    # Returns the list of row vectors in the matrix.
        
    getcols = lambda self: [ self.getcol(i+1) for i in range(self.cols) ]
    # Returns the list of column vectors in the matrix.
        
    getlist = lambda self: self.list
    # Returns a list of values in the matrix.
        
    getarray = lambda self: self.array
    # Returns a structured list of values in the matrix.
        
    def replacerow(self, n, row):
        ''' Replaces the matrix's current nth row (n > 0) with the given vector/list.
        '''
        if type(row) == list:
            assert len(row) == self.cols
        else:
            assert row.dim == self.cols
            row = row.get() # make it a list
        self.array[n-1] = row
        self.list = arraytol(self.array)
            
    def replacecol(self, n, col):
        ''' Replaces the matrix's current nth col (n > 0) with the given vector/list.
        ''' 
        if type(col) == list:
            assert len(col) == self.rows
        else:
            assert col.dim == self.rows
            col = col.get() # make it a list
        for i in range(self.rows):
            self.list[n-1 + cols*i] = col[i]
        self.array = [ [ self.list[j] for j in range(self.rows*i, self.rows*i + self.cols) ] for i in range(self.rows) ]
        
    def swap(self, row1, row2, printmode=False):
        ''' Swaps row1 and row2 in the matrix. Result is row-equivalent to the original matrix.
        '''
        if printmode: print "------------------------------------------------------------------------------------------------------------------------------------------------"
        if printmode: print "Type 1 operation: swap rows " + str(row1) + " and " + str(row2) + ". Before:"
        if printmode: self.pr()
        row_1 = self.getrow(row1)
        row_2 = self.getrow(row2)
        self.replacerow(row1, row_2)
        self.replacerow(row2, row_1)
        if printmode: print "After:"
        if printmode: self.pr()
        
    def scale(self, n, k, printmode=False):
        ''' Scales nth row by k. Result is row-equivalent to the original matrix.
        '''
        if printmode: print "------------------------------------------------------------------------------------------------------------------------------------------------"
        if printmode: print "Type 2 operation: scale row " + str(n) + " by " + str(k) + ". Before:"
        if printmode: self.pr()
        self.replacerow(n, self.getrow(n)*k) # the key line
        if printmode: print "After:"
        if printmode: self.pr()
        
    def reduce(self, n, k, printmode=False):
        ''' Reduces nth row by an integer factor of k. Make sure all values in row are divisible by k.
        '''
        if printmode: print "------------------------------------------------------------------------------------------------------------------------------------------------"
        if printmode: print "Type 2 operation: reduce row " + str(n) + " by " + str(k) + ". Before:"
        if printmode: self.pr()
        self.replacerow(n, Vector([self.getrow(n).get()[i]/k for i in range(self.getrow(n).dim)])) # the key line
        if printmode: print "After:"
        if printmode: self.pr()
        
    def addreplace(self, n, m, k, printmode=False):
        ''' Replaces row n with row n + k * row m. Result is row-equivalent to the original matrix.
        '''
        if printmode: print "------------------------------------------------------------------------------------------------------------------------------------------------"
        if printmode: print "Type 3 operation: replace row " + str(n) + " with row " + str(n) +' + '+ str(k) +' * '+ "row " + str(m) + ". Before:"
        if printmode: self.pr()
        self.replacerow(n, self.getrow(n) + self.getrow(m)*k) # the key line
        if printmode: print "After:"
        if printmode: self.pr()
        
    def __add__(self, other):
        '''Adds two equal-dimensional matrices.
        '''
        assert self.rows == other.rows and self.cols == other.cols
        return Matrix([self.list[i]+other.list[i] for i in range(self.rows*self.cols)], self.rows, self.cols)
        
    __radd__ = lambda self, other: self.__add__(other)
        
    def __mul__(self,other):
        ''' Computes product self*other. Must have appropriate dimensionality.
        '''
        if isinstance(other, Matrix):
            assert self.cols == other.rows
            return Matrix([ other.getcol(j+1)*self.getrow(i+1) for i in range(self.rows) for j in range(other.cols)], self.rows, other.cols)
        elif isinstance(other, Vector):
            assert self.cols == other.dim
            return Vector([ self.getrow(i+1)*other for i in range(self.rows) ])
        else:
            return Matrix([ i*other for i in self.list ], self.rows, self.cols)
            
    def __rmul__(self,other):
        ''' Computes product other*self. Must have appropriate dimensionality.
        '''
        if isinstance(other, Matrix):
            return other.__mul__(self)
        elif isinstance(other, Vector):
            assert self.rows == 1
            return self.getrow(1)*other
        else:
            return Matrix([ i*other for i in self.list ], self.rows, self.cols)
            
    def __div__(self,other):
        '''Computes self/other. 
        '''
        if isinstance(other, Matrix):
            return self.__mul__(inv1(other))
        else:
            return Matrix([ i/other for i in self.list ], self.rows, self.cols)
    
    __rdiv__ = lambda self,other: other.__div__(self)
    
    __eq__ = lambda self, other: all([ (self-other).list[i] == 0 for i in range(self.rows*self.cols) ])  
    
    __neg__ = lambda self: Matrix([-i for i in self.list], self.rows, self.cols)
    
    dim = lambda A: str(A.rows)+'x'+str(A.cols)

def identity(n):
    ''' Returns identity matrix of size n*n.
    '''
    l = [ 0 for i in range(n*n) ]
    for i in range(n):
        l[i*(n+1)] = 1
    return Matrix(l, n, n)
    
I = lambda n: identity(n)

def zero(n):
    ''' Returns zero matrix of size n*n.
    '''
    return Matrix([ 0 for i in range(n*n) ],n,n)

J = lambda n: combineV(combineH(zero(n),identity(n)),combineH(-identity(n),zero(n)))
    

def randomM(n, m):
    ''' Forms an n*m matrix with entries randomly selected from the single-digit integers.
    '''
    return Matrix([ randint(-9,9) for i in range(n*m) ], n, m)
    
def rM(*args):
    ''' Construct a matrix with a list of row vectors.
    '''
    if type(args[0]) == list:
        rowvecl = args[0]
    else:
        rowvecl = list(args)
    for vec in rowvecl:
        assert vec.dim == rowvecl[0].dim
    return Matrix([ vec.get() for vec in rowvecl ])
    
def cM(*args):
    ''' Construct a matrix with a list of column vectors.
    '''
    if type(args[0]) == list:
        colvecl = args[0]
    else:
        colvecl = list(args)
    for vec in colvecl:
        assert vec.dim == colvecl[0].dim
    return Matrix(flip([ vec.get() for vec in colvecl ]))

transpose = lambda A: Matrix(flip(A.getarray()))
# Returns the transpose of a matrix A.
    
def combineH(A, B):
    ''' Horizontally combines two matrices A and B.
    '''
    assert A.rows == B.rows
    return cM(A.getcols()+B.getcols())
    
def combineV(A,B):
    ''' Vertically combines two matrices A and B.
    '''
    assert A.cols == B.cols
    return transpose(cM(transpose(A).getcols()+transpose(B).getcols()))
    
def augment(A, *args):
    ''' Augments matrix A with a list of additional column vectors (essentially splicing two matrices).
    '''
    if type(args[0]) == list:
        addcols = args[0]
    else:
        addcols = list(args)
    assert A.rows == addcols.rows
    return combineH(A, cM(addcols))
        
def cut(A, n, m):
    ''' Returns cols n through m of the matrix, 0 < n <= m.
    '''
    assert 0 < n <= m <= A.cols
    return cM(A.getcols()[n-1:m])
    
def tr(A):
    ''' Returns trace of matrix A.
    '''
    assert A.cols == A.rows
    return sum([ A.array[i][i] for i in range(A.cols)])
    
def maketracezero(A):
    ''' Makes trace zero by changing the first entry.
    '''
    if tr(A) == 0:
        return A
    elif tr(A) < 0:
        A.array[0][0] += 1
        return maketracezero(A)
    else:
        A.array[0][0] -= 1
        return maketracezero(A)
        
def isHamiltonian(A):
    ''' Checks whether a matrix is Hamiltonian.
    '''
    assert A.cols == A.rows 
    assert A.cols % 2 == 0
    J = J(A.cols/2)
    return A*J == transpose(A*J)
    
def pivot(row):
    ''' Returns tuple containing (value of leading coefficient, position of leading coefficient).
    '''
    assert not iszero(row)
    for i in range(row.dim):
        if row.get()[i] != 0:
            return (row.get()[i], i)
            
def type1m(n, i, j):
    ''' Returns an n*n elementary matrix that switches rows i and j. i,j > 0
    '''
    l = [ 0 for a in range(n*n) ]
    for index in range(n*n):
        row = index / 5
        col = index % 5
        if row == i-1:
            if col == j-1:
                l[index] = 1
        elif row == j-1:
            if col == i-1:
                l[index] = 1
        elif row == col:
            l[index] = 1
    return Matrix(l, n, n)
    
def type2m(n, i, k):
    ''' Returns an n*n elementary matrix that multiplies row i by k. i > 0, k != 0
    '''
    assert k != 0
    l = [ 0 for a in range(n*n) ]
    for index in range(n*n):
        if index / 5 == index % 5:
            if index / 5 == i-1:
                l[index] = k
            else:
                l[index] = 1
    return Matrix(l, n, n)
    
def type3m(n, i, j, k):
    ''' Returns an n*n elementary matrix that replaces row i by row i + k * row j. i,j > 0
    '''
    assert i != j
    l = [ 0 for a in range(n*n) ]
    for index in range(n*n):
        row = index / 5
        col = index % 5
        if row == col:
            l[index] = 1
        elif row == i-1 and col == j-1:
            l[index] = k
    return Matrix(l, n, n)
            
def isref(A):
    ''' Checks if matrix A is in row-echelon form. True if yes, False if no.
    '''
    allzero_rows = []
    nonzero_rows = []
    for i in range(A.rows):
        if iszero(A.getrows()[i]):
            allzero_rows.append(i)
        else:
            nonzero_rows.append(i)
    if not allzero_rows == []:
        if min(allzero_rows) < max(nonzero_rows):
            return False # not all nonzero rows are above all allzero rows
    pivots = []
    for row_index in nonzero_rows:
        pivots.append(pivot(A.getrows()[row_index])[1])
    current_col = -1
    for col in pivots:
        if col <= current_col:
            return False # one pivot is not to the right of a pivot above it
        current_col = col
    return True # all tests have been passed  
    
def isrref(A):
    ''' Checks if matrix A is in reduced row-echelon form. True if yes, False if no.
    '''
    if not isref(A):
        return False # matrix is not ref
    pivots = []
    nonzero_rows = []
    for i in range(A.rows):
        if not iszero(A.getrows()[i]):
            nonzero_rows.append(i)
    for row_index in nonzero_rows:
        pivots.append((row_index, pivot(A.getrows()[row_index])[1], pivot(A.getrows()[row_index])[0]))
    for p in pivots:
        if p[2] != 1:
            return False # value of pivot is not 1
        if not only1(A.getcol(p[1]+1)):
            return False # pivot is not only nonzero entry in its column
    return True # all conditions have been satisfied
        
def ref(A, calcasfraction=True, showasfraction=True, hijackfordet=False):
    ''' Computes the row-echelon form of A. Non-destructive.
    '''
    B = A.copy() # copy the matrix. the alg is non-destructive
    if calcasfraction: B.fractionize()
    if hijackfordet: prod = 1
    for i in range(B.cols)[:B.rows-1]:
        # for each column in the matrix, do the following:
        if not iszero(B.getcol(i+1)[i:]): # if the column is all zero, we don't care about it. Move on
            # select a row which has a nonzero entry in this column
            for j in range(B.rows)[i:]:
                if B.array[j][i] != 0: 
                    selected_row = j # just pick the first one
                    break  
            # for each row, subtract a multiple of selected row (if necessary) so that the entry in this column is zero
            for j in range(B.rows)[i:]:
                if B.array[j][i] != 0 and j != selected_row:
                    B.addreplace(j+1, selected_row+1, -B.array[j][i]/B.array[selected_row][i]) # cancel out the rows based on the ith column entry
            # make the row with the non-zero entry the top row
            if selected_row != i:
                B.swap(selected_row+1, i+1) 
                if hijackfordet: prod *= -1
    if calcasfraction and not showasfraction:  B.defractionize()
    if not hijackfordet: return B
    else: return prod
    
def rref(A, calcasfraction=True, showasfraction=True):
    ''' Computes the unique reduced row-echelon form matrix which is row-equivalent to A. Non-destructive.
    '''
    B = ref(A,calcasfraction,showasfraction)
    if calcasfraction: B.fractionize()
    # scale all of the rows to 1 based on their pivot
    for j in range(B.rows): 
        if (not iszero(B.getrow(j+1))) and (pivot(B.getrow(j+1))[0] != 1): # if the row is all zero, don't do anything
            B.scale(j+1, 1/pivot(B.getrow(j+1))[0]) # otherwise, scale such that the pivot is 1
    pivots = [] # a list of pivots (row, col)
    for j in range(B.rows): 
        if not iszero(B.getrow(j+1)): # for each nonzero row, 
            pivots.append((j, pivot(B.getrow(j+1))[1])) # add to the list the position of the pivot
    pivots = [ (j, pivot(B.getrow(j+1))[1]) for j in range(B.rows) if not iszero(B.getrow(j+1))] 
    for row,col in pivots[::-1]: # look at all of the pivots in reverse
        # for each column, subtract out copies of the pivot row as allowable
        for j in range(B.rows)[:row]: # for each row strictly above, subtract a copy of pivot row (if necessary) so that the entry in this column is zero
            if B.array[j][col] != 0:
                B.addreplace(j+1, row+1, -B.array[j][col]) 
    if calcasfraction and not showasfraction: B.defractionize()
    return B
        
def minor(A, row, col):
    ''' Returns the minor matrix obtained by striking out a given row and column in matrix A. Row, col numbers > 0.
    '''
    l = []
    for i in range(A.rows*A.cols):
        if (i / A.rows != row-1) and (i % A.cols != col-1): # if the entry is not in the selected row or col:
            l.append(A.getlist()[i])
    assert len(l) == (A.rows-1)*(A.cols-1) # quality check
    return Matrix(l, A.rows-1, A.cols-1)
    
def multiply(l):
    ''' Takes the product of each of the elements in l.
    '''
    prod = 1
    for i in l:
        prod *= i
    return prod
    
diagonalprod = lambda A: multiply([ A.array[i][i] for i in range(A.rows) ])
    
def det(A, method = 'elementary'):
    ''' Computes the determinant of a square matrix recursively, either by cofactor method or by elementary product. Default: elementary.
    '''
    assert A.rows == A.cols
    if method == 'cofactor':
        d = 0
        if A.rows == 1:
            return A.getlist()[0]
        num_zeros = -1
        for i in range(A.rows): # select the row/col with the most zeros
            if numzeros(A.getrow(i+1)) > num_zeros:
                direction = 'h'
                selected = i
        for j in range(A.cols):
            if numzeros(A.getcol(j+1)) > num_zeros:
                direction = 'v'
                selected = j
        if direction == 'h':
            for j in range(A.cols): # for each column, sum:
                d += A.array[selected][j] * (-1)**(selected+j) * det(minor(A, selected+1, j+1), method='cofactor')
        elif direction == 'v':
            for i in range(A.rows): # for each row, sum:
                d += A.array[i][selected] * (-1)**(selected+i) * det(minor(A, i+1, selected+1), method='cofactor')
        return d
    elif method == 'firstrow':
        d = 0
        if A.rows == 1:
            return A.getlist()[0]
        for j in range(A.cols): # for each column, sum:
                d = d + A.array[0][j] * (-1)**(j) * det(minor(A, 1, j+1), method='firstrow')
        return d
    elif method == 'elementary': return ref(A, hijackfordet = True) * diagonalprod(ref(A))     
    else:
        pass
    
cofactor = lambda A, row, col: (-1)**(row+col)*det(minor(A, row, col))
# Computes the cofactor for the given row and column in matrix A
 
cofactorM = lambda A: Matrix([ [ cofactor(A, i+1, j+1) for j in range(A.cols) ] for i in range(A.rows) ])
# Computes the cofactor matrix for A.

adj = lambda A: transpose(cofactorM(A))
# Computes the adjugate of matrix A

def inv(A, method = 'Gaussian'):
    ''' Computes the inverse of a nonsingular square matrix A using the Gaussian method or the cofactor method. Default is Gaussian.
    '''
    assert A.rows == A.cols and det(A) != 0
    if method == 'Gaussian':
        return cut(rref(combineH(A, identity(A.rows))), A.cols+1, 2*A.cols)
    elif method == 'cofactor':
        B = A.copy()
        B.fractionize()
        return adj(B)/det(B)
    else:
        pass
    
def rank(M):
    ''' Computes the rank of a matrix M.
    '''
    R = rref(M)
    rank = R.rows
    for j in range(R.rows): 
        if iszero(R.getrow(j+1)):
            rank -= 1
    return rank
    
def LU(A):
    ''' Returns LU = A for lower-triangular L and upper-triangular U. Assumes A is nonsingular
    '''
    U = A.copy()
    n = B.cols
    L = I(n)
    for i in range(n):
        # for each column in the matrix, do the following:
        if not iszero(U.getcol(i+1)[i:]): # if the column is all zero, we don't care about it. Move on
            # select a row which has a nonzero entry in this column
            for j in range(i,n):
                if U.array[j][i] != 0: 
                    selected_row = j # just pick the first one
                    break  
            # for each row, subtract a multiple of selected row (if necessary) so that the entry in this column is zero
            for j in range(i,n):
                if U.array[j][i] != 0 and j != selected_row:
                    E = type3m(n, j+1, selected_row+1, -B.array[j][i]/B.array[selected_row][i]) # cancel out the rows based on the ith column entry
                    U = E * U
                    L = L * inv(E)
            # make the row with the non-zero entry the top row
            if selected_row != i:
                E = type1m(n, selected_row+1, i+1)
                U = E * U
                L = L * inv(E)
    for j in range(n): 
        if (not iszero(U.getrow(j+1))) and (pivot(U.getrow(j+1))[0] != 1): # if the row is all zero, don't do anything
            E = type2m(n, j+1, 1/pivot(B.getrow(j+1))[0]) # otherwise, scale such that the pivot is 1
            U = E * U
            L = L * inv(E)
    
    return (L,U)
    
def LI(*args):
    ''' Determines whether the vectors inputted are linearly independent.
    '''
    if not isinstance(args[0], Vector):
        args = args[0]
    else:
        args = list(args)
    pivots = [] # a list of pivots (row, col)
    A = rref(cM(args))
    for row in A.getrows():
        if not iszero(row):
            pivots.append(pivot(row)[1])
    for i in range(A.cols):
        if i not in pivots:
            return False  
    return True
    
def findDepVector(*args):
    ''' Finds a dependent vector
    '''
    if not isinstance(args[0], Vector):
        args = args[0]
    else:
        args = list(args)
    assert not LI(args)
    pivots = [] # a list of pivots (row, col)
    A = rref(cM(args))
    for row in A.getrows():
        if not iszero(row):
            pivots.append(pivot(row)[1])
    for i in range(A.cols):
        if i not in pivots:
            return args[i]
    
def basis(*args):
    ''' Returns a basis that spans the span of the input vectors.
    '''
    if not isinstance(args[0], Vector):
        args = args[0]
    else:
        args = list(args)
    if LI(args):
        return args
    else:
        newargs = []
        for vec in args:
            if vec != findDepVector(args):
                newargs.append(vec)
        return basis(newargs)
        
def aminusxi(A):
    ''' Returns a polynomial A - xI.
    '''
    return A - I(A.rows)*singleterm(1)

def characteristic(A):
    ''' Returns the characteristic polynomial for A.
    '''
    return det(I(A.rows)*singleterm(1)-A, method='firstrow')
    
def iseigenvector(A, v):
    ''' Checks if v is an eigenvector of A.
    '''
    return det(cM(A*v, v)) == 0
    
def iseigenvalue(A, l):
    ''' Checks if l is an eigenvalue of A.
    '''
    return characteristic(A).eval(l) == 0
    
def eigenvalues(A):
    ''' Finds all the rational eigenvalues of A.
    '''
    roots = []
    p = characteristic(A)
    leading = lead(p)
    constant = p[0]
    for a in factorlist(constant):
        for b in factorlist(leading):
            if p.eval(Rational(a,b)) == 0:
                if all([ factor != Rational(a,b) for factor in roots ]):
                    roots.append(Rational(a,b))
    return roots
    
def diagonalize(A):
    ''' Returns P, D diagonal, inv(P) s.t. P*D*inv(P) = A
    '''
    pass
    
def proj(u, v):
    ''' Returns projection of v onto L, the line through the origin and u.
    '''
    return Rational(u*v,u*u)*u
    
def isOrthogonal(*args):
    ''' Tells if a set of vectors are orthogonal.
    '''
    if type(args[0]) == list:
        args = args[0]
    else:
        args = list(args)
    for i in range(len(args)):
        for j in range(len(args))[i+1:]:
            if args[i]*args[j] != 0:
                return False
    return True
    
def orthogonalbasis(*args):
    ''' Creates an orthogonal basis spanning the span of the args.
    '''
    if type(args[0]) == list:
        args = args[0]
    else:
        args = list(args)
    out = []
    args = basis(args)
    for vec1 in args:
        new_vec = vec1
        for vec2 in out:
            new_vec = new_vec - proj(vec2,vec1)
        out.append(new_vec)
    return out
    
def orthonormalbasis(*args):
    ''' Creates an orthonormal basis spanning the span of the args.
    '''
    if type(args[0]) == list:
        args = args[0]
    else:
        args = list(args)
    out = []
    for vec1 in args:
        new_vec = vec1
        for vec2 in out:
            new_vec = new_vec - proj(vec2,vec1)
        out.append(new_vec)
    for i in range(len(out)):
        out[i] = out[i]/mag(out[i])
    return out
    
def QR(A):
    ''' Finds orthonormal Q and upper-triangular R such that QR = A.
    '''
    basis = orthonormalbasis(A.getcols())
    return (cM(basis),transpose(Q)*A)
    
        
    
    



        

    


    

