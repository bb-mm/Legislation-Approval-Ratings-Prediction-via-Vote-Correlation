from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import math
import operator
import random as random
import time

training_sample = 200
v_gamma = 100
num_v = 50
shuffle = True

def loadVoteToBill():
	f = open('bill_vote.dat','r')
	
	ret = {}

	g = f.readline()
	while (len(g.strip()) > 0):
		bill_id = g.strip()

		tmp = f.readline().split()

		for x in tmp:
			ret[x] = bill_id
		
		g = f.readline()

	return dict(ret)


def loadVoteRecord():
	f = open('vote-voter.dat','r')
	
	ret = []

	g = f.readline()
	while (len(g.strip()) > 0):
		inp = g.split()
		vote_id = inp[0]

		tmp = f.readline().split()

		tmp_list = []
		for i in range(len(tmp) / 2):
			if (int(tmp[i * 2 + 1]) == -1):
				tmp_list.append((tmp[i * 2], 0))
			else:
				tmp_list.append((tmp[i * 2], int(tmp[i * 2 + 1])))


		if (len(tmp_list) > 0):
			vote_dict = dict(tmp_list)
			ret.append((vote_id, vote_dict))

		g = f.readline()

	return dict(ret)

def loadVoterRecord():
	f = open('voter-vote.dat','r')
	
	ret = []

	g = f.readline()
	while (len(g.strip()) > 0):
		inp = g.split()
		voter_id = inp[0]
		voter_num = int(inp[1])

		tmp = f.readline().split()

		tmp_list = []
		if (len(tmp) > 0):
			for i in range(len(tmp) / 2):
				if (int(tmp[i * 2 + 1]) == -1):
					tmp_list.append((tmp[i * 2], 0))
				else:
					tmp_list.append((tmp[i * 2], int(tmp[i * 2 + 1])))
			voter_dict = dict(tmp_list)

			ret.append((voter_id, voter_dict))
		
		g = f.readline()

	return dict(ret)

def loadVoteProperty():
	f = open('vote_composition.txt','r')
	
	ret = []

	g = f.readline()
	g = f.readline()
	while (len(g.strip()) > 0):
		vote_property = g.split()
		vote_property.pop(0)
		vote_id = vote_property.pop(0)
		tmp = vote_id.split('/')
		vote_id = tmp[len(tmp) - 1]

		tmp = []
		for i in range(len(vote_property) / 2):
			tmp.append((int(vote_property[i * 2]), float(vote_property[i * 2 + 1])))
		vote_dict = dict(tmp)
		ret.append((vote_id, vote_dict))
		g = f.readline()

	return dict(ret)


def loadBillProperty():
	f = open('text_composition.txt','r')
	
	ret = []

	g = f.readline()
	g = f.readline()
	while (len(g.strip()) > 0):
		bill_property = g.split()
		bill_property.pop(0)
		bill_id = bill_property.pop(0)
		tmp = bill_id.split('/')
		bill_id = tmp[len(tmp) - 1]

		tmp = []
		for i in range(len(bill_property) / 2):
			tmp.append((int(bill_property[i * 2]), float(bill_property[i * 2 + 1])))
		bill_dict = dict(tmp)
		ret.append((bill_id, bill_dict))
		g = f.readline()

	return dict(ret)

def pair_clf(voter_a, voter_b, vote_property, voter_record):
	record_ab = []
	vote_a = set(voter_record[voter_a].keys())
	vote_b = set(voter_record[voter_b].keys())

	allvote = set(vote_property.keys())
	candidate_vote = vote_a.intersection(vote_b).intersection(allvote)
	cnt = 0
	for vote in candidate_vote:
		cnt = cnt + 1
		#if (voter_record[voter_a][vote] != -1 and voter_record[voter_b][vote] != -1):
		record_ab.append((vote_property[vote], voter_record[voter_a][vote], voter_record[voter_b][vote]))
		if (cnt > training_sample):
			break

	n = len(record_ab)
	#print voter_a, voter_b
	if n == 0:
		return 0 
	p = len(record_ab[0][0])

	#print n, p
	#X = np.zeros((n, p))
	X = np.zeros((n,p))
	y = np.zeros(n)

	for i in range(n):
		for j in range(p):
			X[i,j] = record_ab[i][0][j]
		y[i] = int(record_ab[i][1] == record_ab[i][2])
	
	clf = svm.SVC(kernel = 'rbf', gamma = v_gamma)
	#clf = RandomForestClassifier(n_estimators = 10, max_features = 5)
	
	flag = True
	for i in range(len(y)):
		if (y[i] != y[0]):
			flag = False
	if (not flag):
		clf.fit(X,y)
		return clf
	else:
		return y[0]

	return clf

def compute_result(voters, voter_record, x, frac):
	cnt = 0
	tot = 0
	for z in voters:
		#print x, z, voter_record[z].keys()
		if (x in voter_record[z].keys()):
			tot = tot + 1
			cnt = cnt + voter_record[z][x]
	
	if (tot < len(voters) * frac):
		return -1

	if (cnt >= tot * 3.0 / 5):
		return 1
	elif (cnt < tot * 2.0 / 5):
		return 0
	elif (cnt >= tot / 2):
		return 3
	else:
		return 2

def clf_party(voters, vote_property, voter_record):
	print len(voters), voters

	record = []
	frac = 2.0 / 3
	n = 0
	while (n < training_sample):
		record = []
		cnt = 0

		votes = vote_property.keys()
		if shuffle:
			random.shuffle(votes)
		for vote in votes:
			tmp = compute_result(voters, voter_record, vote, frac)
			if (tmp == 1 or tmp == 0):
				cnt = cnt + 1
				record.append((vote_property[vote], tmp)) 
				if (cnt > training_sample * 2):
					break
		
		n = len(record)
		frac = frac * 0.9

	print n, "frac:", frac
	p = len(record[0][0])

	X = np.zeros((n,p))
	y = np.zeros(n)

	for i in range(n):
		for j in range(p):
			X[i,j] = record[i][0][j]
		y[i] = int(record[i][1])

	clf = svm.SVC(kernel = 'rbf', gamma = v_gamma)
	#clf = RandomForestClassifier(n_estimators = 10, max_features = 5)
	
	flag = True
	for i in range(len(y)):
		if (y[i] != y[0]):
			flag = False
	if (not flag):
		clf.fit(X,y)
		return clf
	else:
		return y[0]


	return clf

def predict(vote, vote_property_x, voters, vote_property, voter_record):
	m = len(voters)
	corr = np.zeros((m, m))
	print len(voters)

	start = time.time()
	print "hello!!!"
	#for i in range(len(voters)):
	#	for j in range(len(voters)):
	for i in range(m):
		for j in range(m):
			if (i != j):
				clf_ij = pair_clf(voters[i], voters[j], vote_property, voter_record)
				if (clf_ij == 0 or clf_ij == 1):
					corr[i,j] = clf_ij
				else:
					corr[i,j] = clf_ij.predict(vote_property_x)
		corr[i,i] = 1
		end = time.time()
		print i, ':', end - start
	print corr
	
	
	#post-processing
	U, s, V = np.linalg.svd(corr, full_matrices = True)

	vote_partyA = []
	vote_partyB = []
	print s
	print V[0,:]
	print V[1,:]

	for i in range(m):
		if (abs(s[1]) < 1e-5 or abs(V[0,i]) > abs(V[1,i])):
			vote_partyA.append(voters[i])
		else:
			vote_partyB.append(voters[i])

	clfA = clf_party(vote_partyA, vote_property, voter_record)
	#clfB = clf_party(vote_partyB, vote_property, voter_record)
	if (clfA == 0 or clfA == 1):
		tmpA = clfA
	else:
		tmpA = clfA.predict(vote_property_x)
	
	#if (clfB == 0 or clfB == 1):
	#	tmpB = clfB
	#else:
	#	tmpB = clfB.predict(vote_property_x)

	return tmpA

	if (tmpA != tmpB):
		print "Good Case!"
		if (len(vote_partyA) < len(voters) / 2):
			tmpA = 1 - tmpA
		return tmpA
	else:
		print "Bad Case!"
		return 1 - tmpA	

def cross_validation(vote_property, vote_record, voter_record):
	num_voters = num_v

	tot = 0
	tot_m = np.zeros((2,2))
	cor = 0
	cor_m = np.zeros((2,2))

	start = time.time()
	votes = vote_record.keys()
	if shuffle:
		random.shuffle(votes)
	for x in votes:
		if (not (x in vote_property.keys())):
			continue

		print x
		new_voter_record = {}
		voters = []

		cnt = 0
		for z in vote_record[x].keys():
			voters.append(z)
			cnt = cnt + 1
			if (cnt > num_voters):
				break

		for voter in voters:
			tmp = {}
			for vote in voter_record[voter].keys():
				if (vote != x): 
					tmp[vote] = voter_record[voter][vote]
			new_voter_record[voter] = tmp	
		
		vote_p_x = np.zeros((1, len(vote_property[x])))
		for i in range(len(vote_property[x])):
			vote_p_x[0][i] = vote_property[x][i]
		pred = predict(x, vote_p_x, voters, vote_property, new_voter_record)
		real = compute_result(voters, voter_record, x, 0) 
		print "original real", real
		if (real == 2):
			real = 0
		if (real == 3):
			real = 1

		print int(pred), real, (int(pred) == real)
		cor = cor + int(int(pred) == real)
		tot = tot + 1
		tot_m[int(pred),int(real)] = tot_m[int(pred),int(real)] + 1
		print "Correctness:", cor, '/', tot, '%', float(cor) / tot
		print tot_m

		end = time.time()
		print 'Case ', tot, ':', end - start

def compute_vote_property(tmp_vote_property, vote_to_bill, bill_property):
	n = len(vote_to_bill.keys())
	p = len(bill_property.values()[0])
	q = len(tmp_vote_property.values()[0])

	X = np.zeros((n,p+q))
	y = np.zeros(n)

	votes = vote_to_bill.keys()
	ret = {}

	for i in range(n):
		vote = votes[i]
		bill = vote_to_bill[vote]
		for j in range(p):
			X[i,j] = bill_property[bill][j]
		for j in range(q):
			X[i,p+j] = tmp_vote_property[vote][j]

		ret[vote] = np.ndarray.tolist(X[i,:])

	return ret

bill_property = loadBillProperty()
vote_to_bill = loadVoteToBill()
vote_record = loadVoteRecord()
voter_record = loadVoterRecord()
tmp_vote_property = loadVoteProperty()

vote_property = compute_vote_property(tmp_vote_property, vote_to_bill, bill_property)

print len(bill_property)
print len(vote_record)
print len(voter_record)
print len(vote_property)

voter = voter_record.keys()
#for i in range(100):
#	for j in range(100):
#		print i, j
#		pair_clf(voter[i], voter[j], vote_to_bill, bill_property, voter_record)

#pair_clf('L000174', 'G000445', vote_property, voter_record)
#pair_clf('L000174', 'I000055', vote_property, voter_record)
#pair_clf('L000174', 'M000485', vote_to_bill, bill_property, voter_record)

cross_validation(vote_property, vote_record, voter_record)
