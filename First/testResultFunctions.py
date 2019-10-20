# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 16:35:48 2019

@author: simon
"""

import processData as prd

#to print the result of the test
# inp1, inp2 input
# out1, out2 expected output
# res1, res2 actual output
def print_result(inp1, inp2, out1, out2, res1, res2):
    if (out1,out2) == (res1,res2):
        print("ok")
    else:
        print("error")
        print(inp1," - ",inp2)
        print(out1," - ",out2)
        print(res1," - ",res2)
        
# to test the first_present_second_not function
def test_present_not(a,b, e01, e10):
    res1 = prd.first_present_second_not(a, b)
    res2 = prd.first_present_second_not(b, a)
    print_result(a,b, e01, e10, res1,res2)

# to test the find_error_values function
def test_error_values(inp1, inp2, out1, out2):
    ern, eri = prd.find_error_values(inp1,inp2)
    print_result(inp1, inp2, out1, out2, ern,eri)

# each test that should work
def test_that_work():
    test_error_values([],[],0,[])
    test_error_values([1],[0],1,[0])
    test_error_values([1],[1],0,[])
    test_error_values([1,1],[0,1],1,[0])
    test_error_values([1,1],[0,0],2,[0,1])
    test_error_values([1,1],[1,1],0,[])
    test_error_values([0],[0],0,[])
    test_error_values([1,1,3,4,5,6,2,3,4],
                      [1,3,2,1,3,5,1,4,4],7,[1,2,3,4,5,6,7])
    print()
    test_present_not([0,1,2,3,4],[1,2,3],2,0)
    test_present_not([1,4,6,8],[0,4,7],3,2)
    test_present_not([2,3,5,6,7,9],[2,3,4,5],3,1)
    test_present_not([1,2,3,4],[1,2],2,0)
    
# each test should fail
def test_that_fail():
    test_error_values([],[],2,[])
    test_error_values([1],[0],1,[2])
    test_error_values([1],[1],0,[2])
    test_error_values([1,1],[0,1],12,[0])
    test_error_values([1,1],[0,0],2,[0,21])
    test_error_values([1,1],[1,1],0,[2])
    test_error_values([0],[0],0,[3])
    test_error_values([1,1,3,4,5,6,2,3,4],
                      [1,3,2,1,3,5,1,4,4],9,[1,2,3,4,5,6,10])
    print()
    test_present_not([0,1,2,3,4],[1,2,3],3,0)
    test_present_not([1,4,6,8],[0,4,7],3,1)
    test_present_not([2,3,5,6,7,9],[2,3,4,5],2,1)
    test_present_not([1,2,3,4],[1,2],2,2)
    
    
