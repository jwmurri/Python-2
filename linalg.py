from math import *
from random import *
from rationals import *
from polynomials import *
from complex import *
from complexrationals import *

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
    return V([a2*b3-a3*b2, a3*b1-a1*b3, a1*b2-a2*b1])
    
mag = lambda v: sqrt(sum([v.get()[i]**2 for i in range(v.dim)]))
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
    
dim = lambda A: str(A.rows)+'x'+str(A.cols)

def identity(n):
    ''' Returns identity matrix of size n*n.
    '''
    l = [ 0 for i in range(n*n) ]
    for i in range(n):
        l[i*(n+1)] = 1
    return Matrix(l, n, n)
    
I = lambda n: identity(n)

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
    return rM(A.getrows(), B.getrows())
    
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
    B = ref(A)
    if calcasfraction:
        B.fractionize()
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
    if calcasfraction and not showasfraction:
        B.defractionize()
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
    # elif method == 'elementary': return ref(A, hijackfordet = True) * diagonalprod(ref(A))     
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
    return det(aminusxi(A), method='firstrow')
    
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
    for i in range(len(args)):
        new_vec = args[i]
        for j in range(len(args))[:i]:
            new_vec = new_vec - proj(args[j],args[i])
        out.append(new_vec)
    return out
    
    
    


        



            
            
        
        
        
        
        
    
    
    
    