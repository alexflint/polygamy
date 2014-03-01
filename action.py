import numpy as np
import scipy.linalg

from polysolve import *

F = parse('x**2 + y**2 - 1', 'x-y')
Ls = [ [], [(1,0), (0,1)] ]
basis = { (0,1), (0,0) }
num_vars = 2
var = 0

print 'F:'
for f in F:
    print '  ',f

F_ext = list(F)
for f,L in zip(F,Ls):
    for monomial in L:
        F_ext.append(f * monomial)

print 'F_ext:'
for f in F_ext:
    print '  ',f

var_monomial = tuple(i==var for i in range(num_vars))
monomials = set(term.monomial for f in F for term in f.terms)
products = set(multiply_monomial(var_monomial, b) for b in basis)
required = products.difference(basis)
nuissance = monomials.difference(set.union(basis, required))

print '\nNuissance:'
for x in nuissance:
    print '  ',as_polynomial(x, num_vars)
print 'Required:'
for x in required:
    print '  ',as_polynomial(x, num_vars)
print 'Basis:'
for x in basis:
    print '  ',as_polynomial(x, num_vars)

nn = len(nuissance)
nr = len(required)
nb = len(basis)

ordering = list(nuissance) + list(required) + list(basis)

C,X = matrix_form(F_ext, ordering)
C = C.astype(float)

print '\nC:'
print C

print '\nX:'
for x in X:
    print '  ',x

P,L,U = scipy.linalg.lu(C)

print '\nU:'
print U

C2 = U[ nn:, nn: ]

print '\nC2:'
print C2

C_R2 = C2[ -nr:, :nr  ]
C_B2 = C2[ -nr:,  nr: ]

print '\nC_R2:'
print C_R2
print '\nC_B2:'
print C_B2

A = - np.dot(np.linalg.inv(C_R2), C_B2)

print '\nA:'
print A

eigvals,eigvecs = scipy.linalg.eig(A)
print '\nEigenvalues:'
print eigvals
print '\nEigenvectors:'
print eigvecs

