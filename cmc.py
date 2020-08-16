# Get distance between pair images
# q_g_dis is array 100*100 of distance between pair images
q_g_dis=np.zeros((100,100))
s=0
start_time = datetime.datetime.now()
for i in range(0,len(left_test),100):
  test_pair=[left_test[i:i+100],right_test[i:i+100]]
  probe=siamese_net.predict(test_pair)
  q_g_dis[s]=probe.T
  s=s+1
end_time = datetime.datetime.now()
print('Finish Operation',s)
print ('* total training time:', str(end_time-start_time))

# Display distance between pair images
pr=[]
# Select query index
d=0
# Display distance between selective query and gallery 
for c in range(0,100):
  pr.append('{0:.10f}'.format(float(q_g_dis[d][c]))) #round decimal number
pr = np.squeeze(np.array(pr)) #convert list to array
print(pr,'\n')
print(pr[1]) # Number horizontally from left to right
print(pr.shape)

rank=[]
start_time = datetime.datetime.now()
for k in range(20):
  n_correct=0
  for i in range(len(query_id)):
    ind_sort=np.argsort(-q_g_dis[i])
    a=query_id[i][0:4]
    for j in range(k+1):
      b=gallery_id[ind_sort[j]][0:4]
      if a==b:
        n_correct+=1
        break
  rank.append(100.0*n_correct/100)
end_time = datetime.datetime.now()
print(rank)
print ('* total training time:', str(end_time-start_time))