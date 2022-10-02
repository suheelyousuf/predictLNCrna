import pandas as pd
import numpy as np
import os


from Bio import SeqIO

cols = ['sequence', 'label']
row=[]

with open("../data/Homo38.ncrna_training.fa") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        row.append([str(record.seq),'1'])
        #print(row)
        
        
df = pd.DataFrame(row, columns=cols)
#df.rename(columns={'Unnamed: 0': "sequence"}, inplace = True)
#df['label'] = 1
print(df.count)
print(df.shape)

row=[]
with open("../data/Human.coding_RNA_training.fa") as handle:
    for record in SeqIO.parse(handle, "fasta"):
        row.append([str(record.seq),'0'])
        #print(row)
        
        
df1 = pd.DataFrame(row, columns=cols)

df=df.append(df1,ignore_index=True)

print(df.count)
print(df.shape)

df=df.sample(frac=1)

print(df.count)
print(df.shape)

f=open('../data/random_samples.txt','w')

df.to_csv(f, sep='\t',index=False)


#df = pd.read_csv('../data/Homo38.ncrna_training.fa', sep = '>', )
#print(df.head())
