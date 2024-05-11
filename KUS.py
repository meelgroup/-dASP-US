# /***********[KUS.py]
# Copyright (c) 2018 Rahul Gupta, Shubham Sharma, Subhajit Roy, Kuldeep Meel
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# ***********/

import argparse
import pickle
import random
import time
import os

import numpy as np
import pydot


class Node():
    def __init__(self,label=None,children=[],decision=None):
        self.label = label
        self.children = children
        self.models = 1
        self.decisionat = decision

class Sampler():
    '''Main class which defines parsing, graph drawing, counting and sampling functions'''
    def __init__(self):
        self.totalvariables = None
        self.treenodes = []
        self.useList = False
        self.graph = None
        self.samples = None
        self.drawnnodes = {}
        self.num_var_in_residual = None
        self.num_clause_in_residual = None
        self.clause_in_residual = []
    
    def drawtree(self,root):
        '''Recursively draws tree for the d-DNNF'''
        rootnode = pydot.Node(str(root.label)+" "+str(root.models))
        self.graph.add_node(rootnode)
        self.drawnnodes[root.label] = rootnode
        for ch in root.children:
            if ch.label not in self.drawnnodes:
                node = self.drawtree(ch)
                self.graph.add_edge(pydot.Edge(rootnode,node))
            else:
                self.graph.add_edge(pydot.Edge(rootnode,self.drawnnodes[ch.label]))
        return rootnode

    def parse(self,inputnnffile):
        '''Parses the d-DNNF tree to a tree like object'''
        with open(inputnnffile) as f:
            treetext = f.readlines()
        nodelen = 0
        for node in treetext:
            node = node.split()
            if node[0] == 'c':
                continue
            elif node[0] == 'nnf':
                self.totalvariables = int(node[3])
            elif node[0] == 'L':
                self.treenodes.append(Node(label=int(node[1])))
                nodelen+=1
            elif node[0] == 'A':
                if node[1] == '0':
                    self.treenodes.append(Node(label='T ' + str(nodelen)))
                else:
                    andnode = Node(label='A '+ str(nodelen))
                    andnode.children = list(map(lambda x: self.treenodes[int(x)],node[2:]))
                    self.treenodes.append(andnode)
                nodelen+=1
            elif node[0] == 'O':
                if node[2] == '0':
                    self.treenodes.append(Node(label='F '+ str(nodelen)))
                else:
                    ornode = Node(label='O '+ str(nodelen),decision = int(node[1]))
                    ornode.children = list(map(lambda x: self.treenodes[int(x)],node[3:]))
                    self.treenodes.append(ornode)
                nodelen+=1

    def counting(self,root):
        '''Computes Model Counts'''
        if(str(root.label)[0] == 'A'):
            root.models = 1
            finalbitvec = set()
            for ch in root.children:
                finalbitvec.update(self.counting(ch)) 
                root.models = root.models * ch.models    
            return finalbitvec
        elif(str(root.label)[0] == 'O'):
            bitvecs = []
            bitvecs.append(self.counting(root.children[0]))
            bitvecs.append(self.counting(root.children[1]))
            # set difference to find out uncommon variables
            bitvec2_1 = bitvecs[1] - bitvecs[0]
            bitvec1_2 = bitvecs[0] - bitvecs[1]
            if (not root.children[0].models):
                model1 = 0
            else:
                # accomodating cylinders from uncommon variables in model counts
                model1 = root.children[0].models * (2 ** len(bitvec2_1))
            if (not root.children[1].models):
                model2 = 0
            else:
                model2 = root.children[1].models * (2 ** len(bitvec1_2))
            root.models = model1 + model2
            root.children[0].models = model1
            root.children[1].models = model2
            return bitvecs[0].union(bitvec2_1)
        else:
            bitvec = set()
            try:
                int(root.label)
                bitvec.add(abs(root.label))
                root.models = 1
            except:
                if (str(root.label)[0] == 'F'):
                    root.models = 0
                elif (str(root.label)[0] == 'T'):                    
                    root.models = 1
            return bitvec 

    def getsamples(self,root,indices):
        '''Generates Uniform Independent Samples'''
        if(not indices.shape[0]):
            return
        if(str(root.label)[0] == 'O'):
            z0 = root.children[0].models
            z1 = root.children[1].models
            p = (1.0*z0)/(z0+z1)
            tosses = np.random.binomial(1, p, indices.shape[0])
            self.getsamples(root.children[0],np.array(indices[np.where(tosses==1)[0]]))
            self.getsamples(root.children[1],np.array(indices[np.where(tosses==0)[0]]))
        elif(str(root.label)[0] == 'A'):
            for ch in root.children:
                self.getsamples(ch,indices)
        else:
            try:
                int(root.label)
                for index in indices:
                    if (self.useList):
                        self.samples[index][abs(root.label)-1] = root.label
                    else:
                        self.samples[index] += str(root.label)+' '
            except:
                pass

def random_assignment(totalVars, solution, useList):
    '''Takes total number of variables and a partial assignment
    to return a complete assignment'''
    literals = set()
    if useList:
        solutionstr = ''
        for literal in solution:
            if literal: #literal is not 0 ie unassigned
                literals.add(abs(int(literal)))
        for i in range(1,totalVars+1):
            if i not in literals:
                solutionstr += str(((random.randint(0,1)*2)-1)*i)+" "
            else:
                solutionstr += str(int(solution[i-1]))+" "
    else:
        solutionstr = solution
        for x in solution.split():
            literals.add(abs(int(x)))
        for i in range(1,totalVars+1):
            if i not in literals:
                solutionstr += str(((random.randint(0,1)*2)-1)*i)+" "
    return solutionstr

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  
    parser.add_argument("--outputfile", type=str, default="samples.txt", help="output file for samples", dest='outputfile')
    parser.add_argument("--drawtree", type=int, default = 0, help="draw nnf tree", dest='draw')
    parser.add_argument("--samples", type=int, default = 10, help="number of samples", dest='samples')
    parser.add_argument("--useList", type=int, default = 0, help="use list for storing samples internally instead of strings", dest="useList")
    parser.add_argument("--randAssign", type=int, default = 1, help="randomly assign unassigned variables in a model with partial assignments", dest="randAssign")
    parser.add_argument("--savePickle", type=str, default=None, help="specify name to save Pickle of count annotated dDNNF for incremental sampling", dest="savePickle")
    parser.add_argument("--printStats", type=int, default=0, help="print d-DNNF compilation stats", dest="printStats")
    parser.add_argument("--seed", type=int, default=0, help="seed for random number generator", dest="seed")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dDNNF', type=str, help="specify dDNNF file", dest="dDNNF")
    group.add_argument('--countPickle', type=str, help="specify Pickle of count annotated dDNNF", dest="countPickle")
    group.add_argument('DIMACSCNF', nargs='?', type=str, default="", help='input cnf file')
    
    args = parser.parse_args()
    random.seed(args.seed)
    draw = args.draw
    totalsamples = args.samples
    useListInt = args.useList
    randAssignInt = args.randAssign
    dDNNF = False
    countPickle = False
    DIMACSCNF = ""
    RESIDUALCNF = ""
    printCompilerOutput = False
    if args.DIMACSCNF:
        DIMACSCNF = args.DIMACSCNF
        RESIDUALCNF = DIMACSCNF.replace("model_", "residual_")
    elif args.dDNNF:
        dDNNF = args.dDNNF
    elif args.countPickle:
        countPickle = args.countPickle
    if args.printStats:
        printCompilerOutput = args.printStats
    savePickle = args.savePickle
    useList = False
    if (useListInt == 1):
        useList = True
    randAssign = False
    if (randAssignInt == 1):
        randAssign = True
    sampler = Sampler()
    sampler.useList = useList
    if DIMACSCNF:
        DIMACSCNF = args.DIMACSCNF
        with open(DIMACSCNF, "r") as f:
            text = f.read()
            f.close()
        dDNNF = DIMACSCNF + ".nnf"
        cmd = "./d4 " + DIMACSCNF + " -out=" + dDNNF
        if not printCompilerOutput:
            cmd += " > /dev/null 2>&1"
        else:
            print("The stats of dDNNF compiler: ")
        start = time.time()
        os.system(cmd)
        if not printCompilerOutput:
            print("Time taken for dDNNF compilation: ", time.time() - start)
    if dDNNF:
        start = time.time()
        sampler.parse(dDNNF)
        print("Time taken to parse the nnf text:", time.time() - start)
        if (not sampler.totalvariables):
            print("Formula is UNSAT! The generated d-DNNF is empty.")
            exit()
        start = time.time()
        bitvec = sampler.counting(sampler.treenodes[-1])
        sampler.treenodes[-1].models = sampler.treenodes[-1].models * (2**(sampler.totalvariables - len(bitvec)))
        print("Time taken for Model Counting:", time.time()-start)
        timepickle = time.time()
        if savePickle:
            fp = open(savePickle, "wb")
            pickle.dump((sampler.totalvariables,sampler.treenodes), fp)
            fp.close()
            print("Count annotated dDNNF pickle saved to:", savePickle)
            print("Time taken to save the count annotated dDNNF pickle:", time.time() - timepickle)
    else:
        timepickle = time.time()
        fp = open(countPickle, "rb")
        (sampler.totalvariables,sampler.treenodes) = pickle.load(fp)
        fp.close()
        print("Time taken to read the pickle:", time.time() - timepickle)
        if savePickle:
            fp = open(savePickle, "wb")
            pickle.dump((sampler.totalvariables,sampler.treenodes), fp)
            fp.close()
            print("Time taken to save the count annotated dDNNF pickle:", time.time() - timepickle)

    print("Model Count:",sampler.treenodes[-1].models)
    if draw:
        sampler.graph = pydot.Dot(graph_type='digraph')
        sampler.drawtree(sampler.treenodes[-1])
        sampler.graph.write_png('d-DNNFgraph.png')
    if (useList):
        sampler.samples = np.zeros((totalsamples,sampler.totalvariables), dtype=np.int32)
    else:
        sampler.samples = []
        for i in range(totalsamples):
            sampler.samples.append('')

    ## start working with residual formula
    for line in open(RESIDUALCNF, 'r'):
        l = line.split()
        if line.startswith("p cnf"):
            # number of variables and clauses
            sampler.num_var_in_residual = int(l[-2])
            sampler.num_clause_in_residual = int(l[-1])
        elif line.startswith("c"):
            continue
        else:
            c = [int(_) for _ in line.split() if int(_) != 0]
            sampler.clause_in_residual.append(c)

    start = time.time()
    sampler.getsamples(sampler.treenodes[-1],np.arange(0,totalsamples))
    print("Time taken by sampling:", time.time()-start)
    # f = open(args.outputfile,"w+")
    # if randAssign:
    #     sampler.samples = list(map(lambda x: random_assignment(sampler.totalvariables, x, sampler.useList), sampler.samples))
    #     for i in range(totalsamples):
    #         f.write(str(i+1) + ", " + sampler.samples[i] + "\n")
    #     f.close()
    # else:
    #     if useList:
    #         for i in range(totalsamples):
    #             f.write(str(i+1) + ", " + " ".join(map(str,sampler.samples[i])) + "\n")
    #         f.close()
    #     else:        
    #         for i in range(totalsamples):
    #             f.write(str(i+1) + ", " + sampler.samples[i] + "\n")
    #         f.close()
    # print("Samples saved to", args.outputfile)
    found_answer_set = 0
    for i in range(totalsamples):
        print("Checking models {0}".format(i + 1))
        f = open("temp_" + RESIDUALCNF, 'w')
        assignment = [int(_) for _ in sampler.samples[i].split()]
        positive_assignments = []
        negative_assignments = []
        # get assignment of the current sample
        for var_index in range(1, sampler.num_var_in_residual + 1):
            if var_index in assignment:
                positive_assignments.append(var_index)
            elif -var_index in assignment: 
                negative_assignments.append(var_index)
            else:
                assert("A variable is not assigned to any value" and False)

        f.write("p cnf {0} {1}\n".format(sampler.num_var_in_residual, sampler.num_clause_in_residual + len(negative_assignments) + 1))
        # ordinal clauses
        for each_clause in sampler.clause_in_residual:
            f.write("".join(str(_) + " " for _ in each_clause) + "0\n")

        # negative assignment 
        for each_assign_to_false in negative_assignments:
            assert(each_assign_to_false <= sampler.num_var_in_residual)
            f.write(str(-each_assign_to_false) + " 0\n")
        
        # blocking clause
        f.write("".join(str(-_) + " " for _ in positive_assignments) + " 0\n")
        # checking whether satisfiable or not
        # for each_assign_to_true in positive_assignments:
        #     assert(each_assign_to_true <= sampler.num_var_in_residual)
        #     f.write(str(each_assign_to_true) + " 0\n")
        f.close()

        cmd = './cadical {0} > result-{0}'.format("temp_" + RESIDUALCNF)
        os.system(cmd)

        with open('result-{0}'.format("temp_" + RESIDUALCNF)) as f:
            treetext = f.readlines()
        unsat = False
        for result in treetext:
            if "s UNSATISFIABLE" in result:
                unsat = True
                break
        if unsat:
            found_answer_set += 1

    print("Total samples: {0} and answer sets: {1}".format(totalsamples, found_answer_set))

if __name__== "__main__":
    main()
