# coding:utf-8

with open('','r') as f_r:
    content=f_r.read().split('\n')
for i in content:
    i=i.split('\t')
content =[i.split('\t') for i in content]
#print(content)
smiles=[]
for i in content:
    print(i[-1])
    try:
        if 0.5<=float(i[-1])<0.6:
            smiles.append(i)
    except:
        continue
print(smiles)
with open('','w') as f_w:
    for i in smiles:
        f_w.write(i[0]+'\t'+i[-1]+'\n')
print('Done')