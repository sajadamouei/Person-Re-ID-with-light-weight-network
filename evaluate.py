
# Evaluate for binary classification
fake_lev=fake_ltarget[0:200]
fake_rev=fake_rtarget[0:200]
start_time = datetime.datetime.now()
res1 = siamese_net.evaluate([left_ev,right_ev],target_ev)
end_time = datetime.datetime.now()
print(res1)
print ('* total training time:', str(end_time-start_time))