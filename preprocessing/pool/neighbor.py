import scipy.sparse
import numpy as np
class Graph(object):
    def __init__(self,*args,**kwargs):
        self.node_neighbors = {}
        self.visited = {}
    def add_nodes(self,nodelist):
        for node in nodelist:
            self.add_node(node)
    def add_node(self,node):
        if not node in self.nodes():
            self.node_neighbors[node] = []
    def add_edge(self,edge):
        u,v = edge
        if(v not in self.node_neighbors[u]) and ( u not in self.node_neighbors[v]):
            self.node_neighbors[u].append(v)

            if(u!=v):
                self.node_neighbors[v].append(u)
    def nodes(self):
        return self.node_neighbors.keys()
    def ifconnected(self,N):
        queue = []
        order = []
        #print k
        def bfs():
            while len(queue)> 0:
                node  = queue.pop(0)
                self.visited[node] = True
                for n in self.node_neighbors[node]:
                    if (not n in self.visited) and (not n in queue):
                        queue.append(n)
                        order.append(n)

        queue.append(0)
        order.append(0)
        bfs()
	if len(order)==N:
		print "Yes connceted!!!!!!!!!!!!!!!!!!!!!!!"
	else:
		print "No connceted!!!!!!!!!!!!!!!!!!!!!!!!!!"
		#print order
		#print len(order)
      		print "not visited:::::"
		for i in range(N):
			if (not i in self.visited):
				print i
    def clear(self):
        self.visited = {}
    def breadth_first_search(self,k,root):
        queue = []
        order = []
        #print k
        def bfs():
            while ((len(queue)> 0)and (len(order)< k+2)):
                node  = queue.pop(0)

                self.visited[node] = True
                for n in self.node_neighbors[node]:
                    if (not n in self.visited) and (not n in queue):
                        queue.append(n)
                        order.append(n)

        queue.append(root)
        order.append(root)
        bfs()

        #for node in self.nodes():
            #if not node in self.visited:
                #queue.append(node)
                #order.append(node)
                #bfs()
        #print order
	if len(order)<k+1:
		print "can't find "+str(k)+"neighbor"
		print order
		print queue
        return order
def sort_da(i,da,g,ifsort):
        N=len(da)
        d = np.zeros(N, np.float32)
        l = np.zeros(N, np.int32)
	for ii in range(N):
		d[ii]=(g[i,da[ii]])
		l[ii]=da[ii]
	###############
	if ifsort==1:
            idx=np.argsort(-d)
            d=d[idx]
            l=l[idx]
	###############
	return l,d
def bfs_build_neighbor(g,k,ifsort):
    new_g=Graph()
    idx_row, idx_col, val = scipy.sparse.find(g)
    perm = np.argsort(idx_row)
    rr = idx_row[perm]
    cc = idx_col[perm]
    nnz = rr.shape[0]#edges number
    N = rr[nnz-1] + 1#vertex number
    print "adj graph has vertices:"
    print N
    new_g.add_nodes([i for i in range(N)])
    rowstart = np.zeros(N, np.int32)
    rowlength = np.zeros(N, np.int32)
    oldval = rr[0]
    count = 0
    #print nnz
    for ii in range(nnz):#find every vertex's starting location,storge in rowstart,rowstart[0]....rowstart[N],rowlength[i] storges length
        rowlength[count] = rowlength[count] + 1
        if rr[ii] > oldval:
            oldval = rr[ii]
            rowstart[count+1] = ii
            count = count + 1
    rowlength[0]=rowlength[0]-1
    rowlength[N-1]=rowlength[N-1]+1
    line=0
    for i in range(N):
        rs = rowstart[i]
        for jj in range(rowlength[i]):
            nid =cc[rs+jj]
            new_g.add_edge((i,nid))
	    line=line+1
    #new_g.ifconnected(N)
    new_g.clear()
    neighbordata=np.zeros((N,k), np.int32)
    neighbordist=np.zeros((N,k), np.float32)
    for i in range(N):
        da=new_g.breadth_first_search(k,i)
	dd=[]
        new_g.clear()
        for j in range(k):
	    if j+1>=len(da):
		dd.append(N)
		#neighbordata[i,j]=N
	    else:
		dd.append(da[j+1])
                #neighbordata[i,j]=da[j+1]
	#paixusuanf!!!!!!!!!!!!
	idx,dist=sort_da(i,dd,g,ifsort)
	for j in range(k):
            neighbordata[i,j]=idx[j]
            neighbordist[i,j]=dist[j]
    return neighbordata,neighbordist
