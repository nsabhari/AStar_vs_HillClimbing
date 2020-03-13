#!/usr/bin/env python 

import sys, time, random, csv
import numpy as np
import os.path
from os import path
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)

'''
Functions for calculating Heuristic 0,1,2,3
H0 : Number of attacking pairs (Direct/Indirect)
H1 : Least weight among the attacking queens
H2 : Sum of least weight among the attacking queens of each pair
H3 : Max of min cost to remove attack from each of the queens
'''
def calcHeuristic(mode,n,pos,weight):
    global board
    Heuristic = 0
    nPairs = 0
    if mode == 1: Heuristic = 10
    if mode == 3: AttackingQueensWt = np.zeros(n+1,dtype='int')
        
    for i in range(1,n):
        for j in range(i+1,n+1):
            slope = abs((pos[i]-pos[j])/(i-j))
            if slope == 0 or slope == 1:
                nPairs += 1
                if mode == 0:   Heuristic+=1
                elif mode == 1: Heuristic = min(weight[i],weight[j],Heuristic)
                elif mode == 2: Heuristic += min(weight[i],weight[j])**2
                elif mode == 3:
                    AttackingQueensWt[i] += weight[j]**2
                    AttackingQueensWt[j] += weight[i]**2

        if mode == 3: AttackingQueensWt[i] = min(AttackingQueensWt[i],weight[i]**2)

    if nPairs == 0: Heuristic = 0
    elif mode == 1: Heuristic = Heuristic**2      
    elif mode == 3:
        AttackingQueensWt[i+1] = min(AttackingQueensWt[i+1],weight[i+1]**2)
        Heuristic = np.max(AttackingQueensWt[np.nonzero(AttackingQueensWt)])
        
    return Heuristic

'''
Function to draw board
'''
def draw_board(n,pos,weight,title,pos_orig,flag,data,plotID):
    global fig, method

    if method == 'Both':
        board = fig.add_subplot(1,2,plotID)
    else:
        board = fig.add_subplot(1,1,1)
    # Clearing the board
    board.cla()
    board.axis([0.5,n,0.5,n])
    board.set_xticks(np.arange(0.5,n+1,1))
    board.set_yticks(np.arange(0.5,n+1,1))
    board.xaxis.tick_top()
    board.invert_yaxis()
    board.grid(True)
    board.set_title(title,fontsize=15)

    # Plotting the queens
    for i in range(1,n+1):
        board.text(i,pos[i],weight[i],
                   color='white', size = 17,
                   horizontalalignment = 'center',
                   verticalalignment = 'center',
                   bbox = dict(boxstyle='round', facecolor='black', alpha=0.5))
        board.text(i-0.25,pos_orig[i]-0.25,weight[i],
                   color='black', size = 10,
                   horizontalalignment = 'center',
                   verticalalignment = 'center',
                   bbox = dict(boxstyle='round', facecolor='yellow', alpha=0.5))
                
    # Plotting the attacks
    for i in range(1,n):
        for j in range(i+1,n+1):
            slope = abs((pos[i]-pos[j])/(i-j))
            if slope == 0 or slope == 1:
                board.plot([i,j],[pos[i], pos[j]])

    # Plotting heuristics
    if flag == 1:
        for i in range(1,n+1):
            for j in range(1,n_queens+1):
                if pos[i] != j:
                    board.text(i,j,data[i,j], size = 12,
                            horizontalalignment = 'center',
                            verticalalignment = 'center')
    plt.show(block=False)
    
'''
Hill Climbing (Steepest Ascent)
'''
def HillClimbing(n,pos,weight,H,TLimit,maxSideways):
    flag        = 0
    nExplored   = 0
    nSideways   = 0
    cost = 0
    EffBF       = np.array([0, 0],dtype='int')
    path        = np.array([0,0,0],dtype='int')
    
    posStart    = pos.copy()
    H_curr      = calcHeuristic(H,n,pos,weight)         # Finding current heuristic
    bestTillNow = np.hstack((pos,H_curr,0))
    H_neigh     = np.zeros((n+1,n+1),dtype='float')     # Array to store neighbour heuristics
    start_time  = time.time()
    while H_curr != 0:
        BF = 0
        H_neigh[:,:] = 9*10**6
        # Calculating Heuristic for neighbours
        for i in range(1,n+1):
            for j in range(1,n+1):
                nodeNeibr = pos.copy()
                if pos[i] != j:
                    nodeNeibr[i] = j
                    H_neigh[i,j] = calcHeuristic(H,n,nodeNeibr,weight)
                    BF += 1

        nExplored   += 1
        EffBF[0]    += 1
        EffBF[1]    += BF

        minHeu = H_neigh.min()
        if minHeu > H_curr:     # We have reached a local minima
            flag = 2
            break
        elif minHeu == H_curr:  # Sideways movement
            nSideways += 1
            if nSideways > maxSideways:     # We are stuck
                flag = 2
                break
            
        choices     = np.where((H_neigh == H_neigh.min()))                  # Finding locations of min heuristics in the board ([Queen no], [Row pos])
        selected    = random.randint(0,np.shape(choices)[1]-1)              # Selected randomly from available choices
        cost       += abs(pos[choices[0][selected]] - choices[1][selected]) * weight[choices[0][selected]]**2
        path        = np.vstack((path,[choices[0][selected],pos[choices[0][selected]],choices[1][selected]]))
        
        pos[choices[0][selected]] = choices[1][selected]                    # Updating the board
        H_curr      = H_neigh[choices[0][selected],choices[1][selected]]    # Updating current heuristic
        
        if H_curr < bestTillNow[n+1] or ((H_curr == bestTillNow[n+1] and cost <= bestTillNow[n+2])):
            bestTillNow = np.hstack((pos,H_curr,cost))
            
        end_time = round(time.time()-start_time,4)
        if end_time > TLimit:                               # Time Limit reached
            flag = 1
            break

    end_time    = round(time.time()-start_time,4)
    return flag,bestTillNow.astype(int),nExplored,end_time,EffBF,path
    
'''
A star algorithm
'''
def AStar(n,pos,weight,H,TLimit):
    flag    = 0
    # notAdm  = 0
    # Builing the encoder, used to spped up seaching
    encoder = np.zeros(n+1,dtype='int')
    for i in range(0,n+1):  encoder[i] = (n+1)**i
        
    EffBF   = np.array([[0, 0, 0],[0, 0, 0]],dtype='float')
    nNodes  = 0

    H_curr  = calcHeuristic(H,n,pos,weight)
    nodeActive = np.hstack((pos,H_curr,0,0,0)) # Board config, Heuristic, Cost, H+C, Depth Level

    nodesOpen   = np.array([],dtype='int')
    nodesClosed = np.array(np.sum(np.multiply(pos[0:n+1],encoder)))
    nodesOpenAndClosed = np.array(np.sum(np.multiply(pos[0:n+1],encoder)))

    start_time = time.time()
    while nodeActive[n+1] != 0:
        BF = 0
        for i in range(1,n+1):
            for j in range(1,n+1):
                nodeNeibr = nodeActive.copy()
                if pos[i] != j:
                    nodeNeibr[i] = j
                    config = np.sum(np.multiply(nodeNeibr[0:n+1],encoder))
                                                                                                    
                    cost_orig = np.sum(np.multiply(abs(nodeNeibr[0:n+1] - pos[0:n+1]),weight**2))   # Calculating cost wrt to starting board
                    nodeNeibr[n+2] = abs(nodeActive[i]-j)*weight[i]**2 + nodeActive[n+2]            # Cost for this node
                    
                    if nodeNeibr[n+2] == cost_orig and not(np.any(nodesOpenAndClosed == config)):
                        nodesOpenAndClosed  = np.vstack((nodesOpenAndClosed,config))
                        nodeNeibr[n+1]      = calcHeuristic(H,n,nodeNeibr,weight)    # Heuristic
                        nodeNeibr[n+3]      = nodeNeibr[n+1] + nodeNeibr[n+2]        # Total Cost
                        nodeNeibr[n+4]      = nodeNeibr[n+4] + 1                     # Depth Level

                        if nodesOpen.size != 0: nodesOpen = np.vstack((nodesOpen,nodeNeibr))
                        else:                   nodesOpen = nodeNeibr.copy()

                        nNodes+=1
                        BF += 1
                       
        if nNodes > n**n:   print ('***',nNodes)
        if EffBF.shape[0] <= nodeActive[n+4]:  EffBF = np.vstack((EffBF,[0, 0, 0]))

        EffBF[nodeActive[n+4],0] += 1
        EffBF[nodeActive[n+4],1] += BF

        choices = np.where((nodesOpen[:,n+3] == np.amin(nodesOpen[:,n+3])))
        selected = random.randint(0,len(choices[0])-1)
        #selected = 0
        
        #if nodesOpen[choices[0][selected]][n+3] < nodeActive[n+3] and notAdm == 0:
        #    print("\n----------Heuristic Not Admissible------------")
        #    notAdm = 1
        
        nodeActive  = nodesOpen[choices[0][selected]]

        nodesOpen   = np.delete(nodesOpen,choices[0][selected],0)                                  # Popping out the active node from open list
        nodesClosed = np.vstack((nodesClosed,np.sum(np.multiply(nodeActive[0:n+1],encoder))))    # Adding the active node to closed list

        end_time = round(time.time()-start_time,4)
        if end_time > TLimit:
            flag = 1
            break

    #print ("Nodes Open = " , nodesOpen.shape[0] , ", Nodes Closed = " , nodesClosed.size,\
    #       ",\n% of Max nodes created = " , nNodes*100/(n**n-1))

    EffBF[:,2] = np.divide(EffBF[:,1],EffBF[:,0], out=np.zeros_like(EffBF[:,1]), where=EffBF[:,0]!=0)
    return flag, nodeActive, nodesClosed.size, end_time, EffBF

def help(text):
    print(text)
    print('\n-----N Queens Problem Help-----\n')
    print('Arguments : [#_queens] [BoardConfigFile] [Method] [Heuristic] [TimeLimit]')
    print('#_queens\t: No. of queens in the board (>4)')
    print('BoardConfigFile\t: .csv file name or \'random\' for a random board')
    print('Method\t\t: \'AS\' - A Star, \'HC\' - Hill Climbing,','\'Both\' - Both the methods')
    print('Heuristic\t: Heuristic number (1/2/3)')
    print('TimeLimit\t: Time Limit in secs')
    print('\n-----End Help-----\n')
    sys.exit()

'''
Code starts here
'''

if len(sys.argv) != 6: help("ERROR : Incorrent number of arguments")

n_queens = int(sys.argv[1])               # Reading number of queens
row = np.zeros(n_queens+1,dtype='int')    # Row position of each queen
wt = np.zeros(n_queens+1,dtype='int')     # Weight of each queen

if n_queens < 4: n_queens = 4

# Reading from csv or creating a random board
if sys.argv[2] == 'random':
    for i in range(1,n_queens+1):
        row[i] = random.randint(1,n_queens)
        wt[i] = random.randint(1,9)
else:
    if path.exists(sys.argv[2]):
        row_no = 0
        with open(sys.argv[2]) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for line in csv_reader:
                if len(line) != n_queens:   help("ERROR : Data in csv file is incorrect")
                for col_no in range(0,n_queens):
                    if line[col_no]!='' :
                        row[col_no+1] = row_no+1
                        wt[col_no+1] = int(line[col_no])
                row_no += 1
    else:
        help()

# Reading the method to be used
if sys.argv[3] != 'AS' and sys.argv[3] != 'HC' and sys.argv[3] != 'Both': help()
method = sys.argv[3]
# Creating plots to show the board
fig = plt.figure(method)
if method == 'Both':    fig.set_size_inches(15,7.5)
else:                   fig.set_size_inches(7.5,7.5)

# Reading the heuristic to be used
h_mode = int(sys.argv[4])
if h_mode <1 or h_mode > 3: help("ERROR : Incorrect heuristic number")

print('\nStarting board\t:',row[1:], '\nWeights\t\t:',wt[1:])
print('Heuristic\t:',calcHeuristic(h_mode,n_queens,row,wt),'\n')

if method == 'AS' or method == 'Both':
    print('**********STARTING A STAR SEARCH**********')
    result, end_state, nExpanded, run_time, EBF = AStar(n_queens,row.copy(),wt.copy(),h_mode,int(sys.argv[5]))

    print('\n-----A* search result using Heuristic',h_mode,'-----\n')
    if result == 1: print('Search UNSUCCESSFUL. Printing the current state\n')
    else:           print('Search SUCCESSFUL. Printing the result\n')

    print('Heuristic\t\t='     ,end_state[n_queens+1])
    print('End Position\t\t='  ,end_state[1:n_queens+1])
    print('# Nodes Expanded\t=',nExpanded)
    print('Cost\t\t\t='        ,end_state[n_queens+2])
    print('Run Time (sec)\t\t=',run_time)
    print('\nSequence of moves :')
    j = 1
    for i in range(1,n_queens+1):
        if row[i] != end_state[i]:
            print('Move',j,': Queen',i,'from',row[i],'to',end_state[i])
            j += 1
    print('\nEff. Branching factor\t='  ,round(np.sum(EBF[:,1])/np.sum(EBF[:,0]),3))
    print('Eff. Branching factor for each depth :')
    for i in range(EBF.shape[0]):
        print('Depth',i+1, '\t\t=',round(EBF[i,2],3))
    print('\n-----End Result-----\n')

    if end_state[n_queens+1] == 0:
        draw_text = "A Star : Heuristic = " + str(end_state[n_queens+1]) + ", Cost = " + str(end_state[n_queens+2])
        draw_board(n_queens,end_state[0:n_queens+1],wt,draw_text,row,0,'null',1)
    else:
        draw_text = "A Star : Heuristic = " + str(end_state[n_queens+1])
        HNeigh = np.zeros((n_queens+1,n_queens+1),dtype='float')
        for i in range(1,n_queens+1):
            for j in range(1,n_queens+1):
                nodeNeibr = end_state.copy()
                if end_state[i] != j:
                    nodeNeibr[i] = j
                    HNeigh[i,j] = calcHeuristic(h_mode,n_queens,nodeNeibr,wt)
        draw_board(n_queens,end_state[0:n_queens+1],wt,draw_text,row,1,HNeigh,1)
    

if method == 'HC' or method == 'Both':
    print('**********STARTING HILL CLIMBING SEARCH**********')
    TimeLimit = int(sys.argv[5])
    nRestart = 0
    result = 0
    nExpanded = 0
    EBF = [0,0]
    ResCtr = [0,0,0,0]
    TimeCtr = 0
    while result!=1 and TimeLimit>0:
        result, end_state, nExpanded_temp, run_time, EBF_temp, path_temp = HillClimbing(n_queens,row.copy(),wt.copy(),h_mode,TimeLimit,10)
        TimeCtr     += run_time
        TimeLimit   -= run_time
        
        if result == 2:
            #print('Search', nRestart+1, 'UNSUCCESSFUL\tRestarting... Time Remaining = ',round(TimeLimit,3))
            ResCtr[0] += 1
        elif result == 0:
            #print('Search', nRestart+1, 'SUCCESSFUL\tRestarting... Time Remaining = ',round(TimeLimit,3),'Cost =',end_state[n_queens+2])
            ResCtr[1] += 1
    
        nExpanded += nExpanded_temp
        EBF[0] += EBF_temp[0]
        EBF[1] += EBF_temp[1]
        if nRestart == 0:
            bestTillNow = end_state.copy()
            path = path_temp.copy()
            ResCtr[2] = nRestart+1
            ResCtr[3] = TimeCtr
        else:
            if end_state[n_queens+1] < bestTillNow[n_queens+1] or \
               (end_state[n_queens+1] == bestTillNow[n_queens+1] and end_state[n_queens+2] < bestTillNow[n_queens+2]):
                bestTillNow = end_state.copy()
                path = path_temp.copy()
                ResCtr[2] = nRestart+1
                ResCtr[3] = TimeCtr

        nRestart += 1
        
            
    print('\n-----Hill Climbing result using Heuristic',h_mode,'-----\n')
    if result == 1: print('Time Ended... Printing best available result\n')
    else          : print('Search SUCCESSFUL. Printing the result\n')
    print('Total Searches (Success %) = ',ResCtr[0] + ResCtr[1],'(',round(100*ResCtr[1]/(ResCtr[0] + ResCtr[1]),2),')')
    print('Best solution found in search no.:',ResCtr[2], '@ Time :', round(ResCtr[3],3),'sec')
    print('Heuristic\t\t='      ,bestTillNow[n_queens+1])
    print('End Position\t\t='   ,bestTillNow[1:n_queens+1])
    print('# Nodes Expanded\t=' ,nExpanded)
    print('Cost\t\t\t='         ,bestTillNow[n_queens+2])
    print('Run Time (sec)\t\t=' ,round(TimeCtr,3))
    print('\nSequence of moves :')
    for i in range(1,path.shape[0]):
        print('Move',i,': Queen',path[i,0],'from',path[i,1],'to',path[i,2])

    print('\nEff. Branching factor\t='  ,round(EBF[1]/EBF[0],3))
    print('\n-----End Result-----\n')

    if bestTillNow[n_queens+1] == 0:
        draw_text = "Hill Climbing : Heuristic = " + str(bestTillNow[n_queens+1]) + ", Cost = " + str(bestTillNow[n_queens+2])
        draw_board(n_queens,bestTillNow[0:n_queens+1],wt,draw_text,row,0,'null',2)
    else:
        draw_text = "Hill Climbing : Heuristic = " + str(bestTillNow[n_queens+1])
        HNeigh = np.zeros((n_queens+1,n_queens+1),dtype='float')
        for i in range(1,n_queens+1):
            for j in range(1,n_queens+1):
                nodeNeibr = bestTillNow.copy()
                if bestTillNow[i] != j:
                    nodeNeibr[i] = j
                    HNeigh[i,j] = calcHeuristic(h_mode,n_queens,nodeNeibr,wt)
        draw_board(n_queens,bestTillNow[0:n_queens+1],wt,draw_text,row,1,HNeigh,2)

input('Press any key to exit...')
    

