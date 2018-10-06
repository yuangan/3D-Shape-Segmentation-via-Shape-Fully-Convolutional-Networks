import numpy as np
import os
import scipy.sparse
import coarsening_new
import graph_new
import neighbor
import math
if_sort=1
def write_mydata_dist(perm,idx,dist,g_dist_name,dh_angle,g_idx_name,read_file_adj):
        N,k=idx.shape
        assert k==3
        #graph_dist=np.zeros((N,k),np.float32)
        distname=open(g_dist_name, "w")
        distneighbor=open(g_idx_name, "w")
        fake_n=perm.index(N)
        offset_map=np.zeros(N, np.int32)
        for i in range(len(perm)):
                now=perm[i]
                if now<N:
                        offset_map[now]=i
        for i in range(len(perm)):
                m=perm[i]
                if m<N:
                        one_row_aj=read_file_adj[4*m+1:4*m+4]
                        one_row_dh=dh_angle[3*m:3*m+3]
                        for j in range(k):
                                now_idx=idx[m,j]
                                ok=offset_map[now_idx]
                                distneighbor.write(str(ok))
                                distneighbor.write(" ")
                                aj_dh_match=one_row_aj.index(str(now_idx))
                                angle=float(one_row_dh[aj_dh_match])
                                #angle=float(dh_angle[3*m+j])
                                new_dist=angle
                                distname.write(str(new_dist))
                                distname.write(" ")
                else:
                        for j in range(k):
                                distneighbor.write(str(fake_n))
                                distneighbor.write(" ")
                                distname.write("0.0")
                                distname.write(" ")
                distname.write("\n")
                distneighbor.write("\n")
        distneighbor.close()
        distname.close()
        
def readseg(seg,N):
        seglabel=np.zeros(N, np.int32)
        filename=open(seg, "r")
        read_file=filename.read().split()
        for i in range(N):
                seglabel[i]=int(read_file[i])
        return seglabel
def write_face_area(face_area,perm,name):
        N=len(face_area)
        facename=open(name, "w")
        for i in range(len(perm)):
                m=perm[i]
                if m<N:#m
                        facename.write(str(face_area[m]))
                else:
                        facename.write("0.0")
                facename.write("\n")
        facename.close()
def write_seg(seglabel,perm,name):
        N=len(seglabel)
        bigseg=np.zeros(len(perm), np.int32)
        segname=open(name, "w")
        for i in range(len(perm)):
                m=perm[i]
                if m<N:#m
                        segname.write(str(seglabel[m]))
                        bigseg[m]=seglabel[m]#m
                else:
                        segname.write("0")
                        bigseg[i]=0
                segname.write("\n")
        segname.close()
        return bigseg
def write_perm(perm,name):
        permname=open(name, "w")
        for i in range(len(perm)):
                m=perm[i]
                permname.write(str(m))
                permname.write("\n")
        permname.close()
def write_feature_and_offset(feature_NNNNNN,f,perm,idx,filename1,N):
        featurename=open(filename1, "w")
        #offsetname=open(filename2, "w")
        if N==len(perm):
                out=0
        else:
                out=perm.index(N)
        for i in range(len(perm)):
                m=perm[i]
                for n in range(feature_NNNNNN):#3 d features
                        if m<N:#m
                                featurename.write(str(f[m,n]))#m
                                featurename.write(" ")
                        else:
                                featurename.write("0.0")
                                featurename.write(" ")
                featurename.write("\n")
                #for j in range(3):#3 neighbors
                        #if m<N:
                                #n=idx[m,j]
                                #k=perm.index(n)
                                #offsetname.write(str(n))
                                #offsetname.write(" ")
                        #else:
                                #offsetname.write(str(out))
                                #offsetname.write(" ")
                #offsetname.write("\n")
        featurename.close()
        #offsetname.close()
def write_k_neighbor_offset(ne_matrix,neighbordist,perm,filename):
        neighborname=open(filename,'w')
        #neighbordistname=open(filename2,'w')
        N,k=ne_matrix.shape
        if N==len(perm):
                out=0
        else:
                out=perm.index(N)
                #print out
        offset_map=np.zeros(N, np.int32)
        for i in range(len(perm)):
                now=perm[i]
                if now<N:
                        offset_map[now]=i
        for i in range(len(perm)):
                m=perm[i]
                if m<N:
                        for j in range(k):
                                aa=ne_matrix[m,j]
                                idd=offset_map[aa]
                                neighborname.write(str(idd))
                                neighborname.write(" ")
                                #neighbordistname.write(str(neighbordist[m,j]))
                                #neighbordistname.write(" ")
                else:
                        for j in range(k):
                                #out=perm.index(m)
                                neighborname.write(str(out))
                                neighborname.write(" ")
                                #neighbordistname.write("0.0")
                                #neighbordistname.write(" ")
                neighborname.write("\n")
                #neighbordistname.write("\n")
        neighborname.close()
        #neighbordistname.close()
def get_distance(i,j,k,read_file1,read_file2):
        o=0
        for ii in range(k):
                a=float(read_file1[ii])-float(read_file2[ii])
                o=o+a**2
        return np.sqrt(o)


def get_neighbor(i,k,read_file):
        neighbor_list=[]
        neighbor_list.append(int(read_file[4*i+1]))
        neighbor_list.append(int(read_file[4*i+2]))
        neighbor_list.append(int(read_file[4*i+3]))
        return neighbor_list

def node_feature_build(N,k,read_file):
        g_feature=np.zeros((N,k), np.float32)
        for i in range(N):
                a=read_file[i].split()
                if not len(a)==k:
                        print "errrrrrrrrrrrrrrrrrrrr!!!"
                for j in range(k): 
                        g_feature[i,j]=float(a[j])
        return g_feature


def graph_build(N,k,read_file,read_file_adj):
        #idx=[]
        idx = np.zeros((N,3), np.int32)
        #dist=[]
        dist = np.zeros((N,3), np.float32)
        for i in range(N):
                neighbor=get_neighbor(i,k,read_file_adj)
                a=read_file[i].split()
                for j in range(len(neighbor)):
                        idx[i,j]=neighbor[j]
                        a_j=read_file[neighbor[j]].split()
                        dist[i,j]=get_distance(i,neighbor[j],k,a,a_j)
                        del a_j
                del a
	#print dist[1,0]
        A = graph_new.adjacency(dist, idx).astype(np.float32)
        return A,idx,dist

def compute(name,dir_all,rr_feature,rr_seg,rr_adj,rr_dh,feature_NNNNNN,neighbor_number,face_origin):
        #name="2"
        seg=rr_seg+name+".seg"
        file_dh_angle=rr_dh+name+"_Dist.txt"
        file_face=face_origin+name+"_FaceArea.txt"
        
        g_feature_name=dir_all+"mydata/"+name+".txt"
        g_seg_name=dir_all+"mydata_output/"+name+".txt"
        g_neighbor_name=[dir_all+"mydata_generate_1/"+name+".txt",dir_all+"mydata_generate_2/"+name+".txt",dir_all+"mydata_generate_3/"+name+".txt",dir_all+"mydata_generate_4/"+name+".txt",dir_all+"mydata_generate_5/"+name+".txt"]
        g_perm_name=[dir_all+"perm/"+name+".txt",dir_all+"perm/"+name+"_1.txt",dir_all+"perm/"+name+"_2.txt",dir_all+"perm/"+name+"_3.txt",dir_all+"perm/"+name+"_4.txt"]

        g_face_name=dir_all+"mydata_area/"+name+".txt"
        

        g_dist_name=dir_all+"mydata_dist/"+name+".txt"
        g_idx_name=dir_all+"mydata_idx/"+name+".txt"

        
        filename=open(rr_feature+name+".off.txt", "r")
        filename_adj=open(rr_adj+name+"_adjacentfaces.txt","r")
        filename_dh=open(file_dh_angle,"r")
        filename_face=open(file_face,"r")
        read_face_area=filename_face.read().split()
        get_dh=filename_dh.read().split()


        read_file=filename.read().split("\n")
        if (len(read_file[-1])==0):
                read_file=read_file[:-1]
        read_file_adj=filename_adj.read().split()
        
        num_faces = len(read_file_adj)/4
        N=int(num_faces)
        print N
        seglabel=readseg(seg,N)

        ##get mesh feature for each faces:the feature map
        graph_feature=node_feature_build(N,feature_NNNNNN,read_file)

        g,idx,dist=graph_build(N,feature_NNNNNN,read_file,read_file_adj)
        #find the around,conv.........  output the same graph,a new feature map

        filename.close()
        filename_adj.close()
        filename_dh.close()
        ####pooling preparing:get the pooling rule:perm 4 to 1,use max pooling
        ####two graphs:graphs[0] the same graph but add fake nodes(fakendoes index >len(graph's node) graphs[2]the graph after pooling,graph_feature_new the feature map after pooling,
        ####we only need graph_feature_new and the new graph
        graphs_pool1, perm_pool1, fakeN_pool1= coarsening_new.coarsen(g, levels=10, self_connections=False)
        write_mydata_dist(perm_pool1[0],idx,dist,g_dist_name,get_dh,g_idx_name,read_file_adj)
        write_feature_and_offset(feature_NNNNNN,graph_feature,perm_pool1[0],idx,g_feature_name,N)
        bigseg=write_seg(seglabel,perm_pool1[0],g_seg_name)
        write_face_area(read_face_area,perm_pool1[0],g_face_name)
        write_perm(perm_pool1[0],g_perm_name[0])
        for i in range(5):
                print i
                neighbordata,neighbordist=neighbor.bfs_build_neighbor(graphs_pool1[i*2],neighbor_number,if_sort)
                if (i==0):
                        print "write dist file"
                        #write_mydata_dist(perm_pool1[0],neighbordata,neighbordist,g_dist_name,get_dh,read_file_adj)
                write_k_neighbor_offset(neighbordata,neighbordist,perm_pool1[i*2],g_neighbor_name[i])
