# -*- coding: utf-8 -*-
"""
AN HMM MODEL FOR SARS-CoV 2 GENE FINDING 

Created on Fri Nov 10 10:50:59 2023
@author: Alexandre de Carvalho Assunção
"""

import numpy as np
from itertools import product

NUCLEOTIDES = ['A','C','G','T']
CODONS = [''.join(codon) for codon in product(NUCLEOTIDES,repeat=3)]
MAX_ITERATIONS = 10
SARS_CoV2_FASTA = 'C:/Users/alexc/OneDrive/Documentos/HMM/COVID.txt'
CODON_CHART = 'C:/Users/alexc/OneDrive/Documentos/HMM/codon-table-grouped.csv'

class HMM(object):

    def __init__(self):
        
        self.States = {'SC':0,'C':1,'STOP':2,'NC':3}
        
        '''
        Emission Probabilities For Each State Of The HMM:
        '''
        ##################################################
        #
        # 1.1- Create the Dictionaries:
        #
        self.SC = {codon: 0 for codon in CODONS}
        self.C = {codon: 0 for codon in CODONS}
        self.STOP = {codon: 0 for codon in CODONS}
        self.NC = {codon: 0 for codon in CODONS}
        #
        # 1.2- Fill The Prior Start Codon Emission Probabilities
        #
        for codon in CODONS:
            if codon == 'ATG':
                self.SC[codon] = 0.99
            else:
                self.SC[codon] = 0.01/63
        #
        # 1.3- Fill The Prior Coding Region Emission Probabilities
        #
        for codon in CODONS:
            self.C[codon] = 1/64
        #
        # 1.4- Fill The Prior Stop Codon Emission Probabilities
        #
        for codon in CODONS:
            if codon in ['TAA','TAG','TGA']:
                self.STOP[codon] = 0.33
            else:
                self.STOP[codon] = 0.01/63
        #
        # 1.5- Fill The Prior Non-Coding Region Emission Probabilities
        #
        for codon in CODONS:
            self.NC[codon] = 1/64
        #
        # 1.6- Join The Emission Probabilities Into A List
        #
        self.E = [self.SC,self.C,self.STOP,self.NC]
        
        '''
        HMM's Zero Time Distribution:
        '''
        self.state_prior = [0.1,0.4,0.1,0.4]
        
        '''
        Prior Transition Probabilities
        '''
        #
        # A[k][l] : Probability Of Going From k To l
        #
        self.A = np.zeros((4,4))
        self.A = [[0.04,0.8,0.15,0.01],[0.01,0.8,0.15,0.04],[0.006,0.001,0.002,0.991],[0.15,0.005,0.025,0.82]]


    def Viterbi(self,X):
         
         ####################################################
         # Initializes Dynammic Programming Matrix          #
         #                                                  #
         n,m = len(self.States),len(X)
         V = np.zeros((n,m))
         V[:,0] = [np.log(self.state_prior[i]) + np.log(self.E[i][X[0]]) for i in range(n)]

         ####################################################
         # Recursive Step of Dynammic Programming           #
         #                                                  #
         for j in range(1,m):
             for i in range(n):
                 
                try:
                    Rec = [V[k][j-1] + np.log(self.A[k][i]) for k in range(n)]
                    V[i][j] = np.log(self.E[i][X[j]]) + max(Rec)
                except FloatingPointError:
                    print("WARNING: Zero Prob.")
                    V[i][j] = np.NINF
                         
         ###################################################
         # Decodes Optimal Path                            #
         #                                                 #
         Path = [np.argmax(V[:,i]) for i in range(m)]
        
         return V,Path
    
    
    def Viterbi_EM(self,X):
        
        
        n,m = len(self.States),len(X)  # Aux Variables to use in loops
        reg = 1e-7                     # Regularization Parameter to Avoid Zero Probs And Infinite Logs
        Iterations = []                # Results of each Iteration
        
        for I in range(MAX_ITERATIONS):
            
        
            ##################################################
            ##################################################
            ''' 
                1-               E STEP      
            '''
            #
            # 1.1 - RUNS VITERBI ALGORITHM
            #
            V,Path = self.Viterbi(X)
            #
            # 1.2 - CALCULATES THE LOG-LIKELIHOOD
            #
            likelihood = np.log(self.E[Path[0]][X[0]]) + np.log(self.state_prior[Path[0]])
            likelihood += sum([np.log(self.E[Path[i]][X[i]]) + np.log(self.A[Path[i-1]][Path[i]]) for i in range(1,m)])
           
            Iterations.append(likelihood)
           
            ##################################################
            ##################################################
            '''  
                2-               M STEP
            '''
            #
            # 2.1 - UPDATE EMISSION PROBABILITIES
            #
            Codon_Count = {codon: [0,0,0,0] for codon in CODONS}
            
            Total_Emissions = np.zeros((n))
            
            # 2.1.a ) Counts Emissions Using State Vector Obtained in E STEP 
            for state in range(n):
                for c in CODONS:
                    Codon_Count[c][state] = sum([Path[i] == state and X[i] == c for i in range(m)])
                Total_Emissions[state] = sum([Path[i] == state for i in range(m)])
                
            # 2.1.b) Creates New Emission Dict and Updates it with the values obtain in 2.1.a
            observed_Emissions = self.E.copy()
            for state in range(n):
                for c in CODONS:
                    observed_Emissions[state][c] = Codon_Count[c][state]/Total_Emissions[state] + reg
                
            self.E = observed_Emissions.copy()
            
            #
            # 2.2 - UPDATE TRANSITION PROBABILITIES
            #
            observed_Transitions = np.zeros((n,n))
            total_Transitions = np.zeros((n))
            # 2.2.a) Counts Observed Transitions
            for i in range(m-1):
                observed_Transitions[Path[i]][Path[i+1]] += 1
            for i in range(m):
                total_Transitions[Path[i]] += 1
            # 2.2.b) Calculates New Transition Probabilities Based On Observed Data
            for i in range(n):
                for j in range(n):
                    observed_Transitions[i][j] = observed_Transitions[i][j]/total_Transitions[i] + reg
            
            self.A = observed_Transitions.copy()
            
        '''
        3- CREATES A LIST OF THE GENES FOUND
        '''
        Genes = []
        Current_Gene = []
        
        i = 0
        while(i < m):
            #
            # 3.1 - Finds A Start Codon
            #
            if Path[i] == 0:
                Current_Gene.append(X[i])
                i +=1 
                while(True):
                    
                    #
                    # 3.1.1 - Keeps Storing the Codons as Long As We Don't Find a Stop Codon
                    #
                    if Path[i] == 1 or Path[i] == 0:
                        Current_Gene.append(X[i])
                        i += 1
                        if(i >= m):
                            break
                    #
                    # 3.1.1 - Break The Loop When Finding A Stop Codon
                    #
                    if Path[i] == 2 or Path[i] == 3:
                        i += 1
                        break
                #
                # 3.2 - Store The Gene If It Has Reasonable Length
                #
                if len(Current_Gene) > 70:
                    Genes.append(Current_Gene)
                Current_Gene = []
            else:
                i += 1
        return Genes


##########################################################
'''
AUXILIARY FUNCTIONS TO HANDLE INPUT/OUTPUT
'''
##########################################################

def parseFasta(FILENAME):
    
    with open(FILENAME,'r') as data:
        lines = [line.strip() for line in data]        
    genome = ''.join(lines[1:])
    return genome


def GenomeToORF(G):
    
    #
    # Breaks Genome G into three reading frames.
    #
    
    while(len(G)%3 != 2):
        G.append('A')
    
    ORF1 = [G[i:i+3] for i in range(0,len(G)-2,3)]
    ORF2 = [G[i:i+3] for i in range(1,len(G)-1,3)]
    ORF3 = [G[i:i+3] for i in range(2,len(G),3)]
    
    return [ORF1,ORF2,ORF3]

def Translation(G):
    
    '''
    Input: G(String List)--> A list containing the codons of several genes
    Output: Proteome(String List)--> A list of sequences of amino acids
    '''
    
    with open(CODON_CHART,'r') as data:
           lines = [line.strip() for line in data]
    
    T_Dict = {}
    
    for i in range(1,len(lines)):
        T_Dict[lines[i][2]+lines[i][3]+lines[i][4]] = lines[i][0]
    
    #
    # Translates SARS-CoV2 Genes Into Proteins
    #
    Proteome = []
    Protein = ""
    
    for g in G:
        for i in range(len(g)):
           if g[i] in ['TAA','TGA','TAG']:
               continue
           Protein += T_Dict[g[i]]
        Proteome.append(Protein)
    
    return Proteome



####################################################################
##           MAIN BODY OF THE PROGRAM                      #########
####################################################################
#
# a) Reads Genome From Fasta File
#
G = parseFasta(SARS_CoV2_FASTA)
#
# b) Breaks Genome Into Reading Frames
#
ORF = GenomeToORF(G)
#
# c) Creates a Hidden Markov Model To Find The Genes
#
My_HMM = HMM()

Genes = [] # Stores the Codons of Genes
P = []     # Stores the amino Acids

'''
RUNS THE HMM MODEL
'''
for i in range(len(ORF)):
    Genes.append(My_HMM.Viterbi_EM(ORF[i]))
    P.append(Translation(Genes[i]))
    
'''
SAVES THE RESULT INTO OUTPUT FILE
'''
OUT = open("C:/Users/alexc/OneDrive/Documentos/HMM/SARS_COV_2_PROTEOME.txt",'w')
for i in range(len(P)):
    for j in range(len(P[i])):
        OUT.write('> Protein:\n')
        OUT.write('> P len: ' + str(len(P[i][j])) + '\n')
        OUT.write(P[i][j])
        OUT.write('\n')
OUT.close()

####################################
######### END ######################
####################################
####################################