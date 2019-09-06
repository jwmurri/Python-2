from complex import *
from rationals import *

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
        
        
            